"""
Phase 2 — Steps 4–7: Preprocessing, Hyperparameter Tuning, Training, Ensemble.

V3 "Dynamic Duo" architecture:
- XGBoost (PRIMARY, weight 0.80): Semi-supervised PU Learning on heuristic labels.
- IsolationForest (SECONDARY, weight 0.20): Unsupervised anomaly detection.
- LOF and OCSVM removed — O(N²) bottleneck with insufficient weight/performance.
- Ensemble anomaly flag: composite_score >= 85 (high-confidence threshold).
"""

import logging
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# Heavy-tailed features whose raw distributions span many orders of magnitude.
# Without log-compression, RobustScaler still leaves extreme outliers that
# collapse PCA: a single component absorbs >92% of variance, destroying the
# multi-dimensional structure that LOF and OCSVM depend on.
_LOG_TRANSFORM_COLS = frozenset([
    "total_achievements",
    "max_achievements_per_day",
    "max_achievements_per_minute",
    "median_unlock_interval_sec",
    "min_unlock_interval_sec",
    "std_unlock_interval_sec",
    "avg_achievements_per_game",
    "library_size",
    "total_reviews",
    "avg_review_length",
])


def apply_log_transform(X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p(clip(x, 0)) to heavy-tailed features before the sklearn pipeline.

    Must be called on the raw feature DataFrame before both preprocess() and
    preprocessor.transform() (e.g. inside train_xgboost_semisupervised) so
    that the same transformation is applied consistently at fit and predict time.

    Returns a new DataFrame — does not modify X_raw in place.
    """
    X = X_raw.copy()
    cols = [c for c in _LOG_TRANSFORM_COLS if c in X.columns]
    if cols:
        log.info("  Applying log1p to %d heavy-tailed features: %s", len(cols), cols)
        X[cols] = np.log1p(X[cols].clip(lower=0))
    return X


# ---------------------------------------------------------------------------
# Step 4 — Preprocessing with PCA
# ---------------------------------------------------------------------------

def preprocess(X_raw: pd.DataFrame, save_path: str,
               pca_variance: float = 0.90) -> tuple[np.ndarray, Pipeline, list]:
    """
    Fit Imputer → StandardScaler → PCA pipeline and persist it.

    PCA retains `pca_variance` fraction of explained variance (default 90%).
    This typically reduces 25 features to 8-12 PCA components, removing noise
    and multicollinearity before model training.
    """
    log.info("Step 4 — Preprocessing (Imputer + StandardScaler + PCA) …")
    # StandardScaler replaces RobustScaler: after log1p the heavy-tailed
    # distributions are approximately symmetric, so z-score normalisation is
    # mathematically optimal.  RobustScaler uses IQR which collapses to ~0 for
    # the many sparse/ghost accounts (0 achievements), causing extreme scale
    # blowup for active players and PCA collapse.
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("pca",     PCA(n_components=pca_variance, random_state=42)),
    ])
    X_scaled = preprocessor.fit_transform(X_raw)

    pca_step = preprocessor.named_steps["pca"]
    n_components = pca_step.n_components_
    variance_explained = pca_step.explained_variance_ratio_.sum() * 100
    log.info("  PCA: %d features → %d components (%.1f%% variance explained)",
             X_raw.shape[1], n_components, variance_explained)
    # Log per-component ratios to verify PCA collapse is fixed.
    # Before log-transform: component 1 alone explains ~92% (collapse).
    # After log-transform: expect a smoother distribution across 8-15 components.
    log.info("  PCA explained variance per component: %s",
             np.round(pca_step.explained_variance_ratio_, 3).tolist())

    pca_feature_names = [f"PC{i+1}" for i in range(n_components)]

    joblib.dump(preprocessor, save_path)
    log.info("  Preprocessor saved → %s", save_path)
    return X_scaled, preprocessor, pca_feature_names


# ---------------------------------------------------------------------------
# Step 5 — Hyperparameter Tuning (IsolationForest)
# ---------------------------------------------------------------------------

def tune_models(X_scaled: np.ndarray,
                y_heuristic: pd.Series) -> tuple[dict, pd.DataFrame]:
    """
    Grid-search IsolationForest and return best params + tuning results.
    ROC-AUC measures fit to heuristic labels (pseudo ground truth).
    """
    log.info("Step 5 — Hyperparameter tuning (IsolationForest) …")
    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_samples":   ["auto", 0.8, 0.6],
        "contamination": [0.02, 0.05, 0.10],
        "max_features":  [0.8, 1.0],
        "random_state":  [42],
    }
    log.info("  Tuning IsolationForest (%d param combos) …",
             len(list(ParameterGrid(param_grid))))
    rows = []
    for params in ParameterGrid(param_grid):
        t0 = time.time()
        m = IsolationForest(**params)
        m.fit(X_scaled)
        scores = -m.score_samples(X_scaled)
        auc = roc_auc_score(y_heuristic, scores)
        rows.append({**params, "roc_auc": auc, "runtime_s": time.time() - t0})

    if_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    best_if = if_df.iloc[0].drop(["roc_auc", "runtime_s"]).to_dict()
    best_if["n_estimators"] = int(best_if["n_estimators"])
    best_if["random_state"]  = int(best_if["random_state"])
    log.info("    Best IF  ROC-AUC=%.4f  %s", if_df.iloc[0]["roc_auc"], best_if)

    tuning_results = if_df.assign(model="IsolationForest")
    return best_if, tuning_results


# ---------------------------------------------------------------------------
# Step 5b — XGBoost Hyperparameter Tuning + Training (PRIMARY model)
# ---------------------------------------------------------------------------

def train_xgboost_semisupervised(
        X_raw: pd.DataFrame,
        y_heuristic: pd.Series,
        y_normal: pd.Series,
        preprocessor: Pipeline,
        outputs_dir: str) -> tuple:
    """
    Train XGBoost using Positive-Unlabeled (PU) Learning logic.

    Training is restricted to the confident subset:
      - heuristic_bot == 1  (confirmed bots)
      - heuristic_normal == 1  (confirmed normals: > 10 achievements AND
                                median interval > 30 min)
    The ambiguous grey area (neither confirmed bot nor confirmed normal) is
    excluded from .fit() to avoid noisy labels polluting the decision boundary.

    Prediction (predict_proba) runs on the ENTIRE trimmed dataset so every
    player receives an anomaly score for ensemble scoring.

    Returns (best_xgb, xgb_proba_full, common_ids).
    """
    import os
    try:
        import xgboost as xgb
    except ImportError:
        log.error("xgboost not installed. Run: pip install xgboost")
        raise

    log.info("Step 5b — XGBoost PU-Learning training …")

    common_ids  = X_raw.index.intersection(y_heuristic.index)
    X_all       = X_raw.loc[common_ids]          # full trimmed set (for scoring)
    y_bot_all   = y_heuristic.loc[common_ids]
    y_norm_all  = y_normal.reindex(common_ids).fillna(0).astype(int)

    # ── PU filter: keep only rows confidently labelled bot OR normal ──────────
    pu_mask = (y_bot_all == 1) | (y_norm_all == 1)
    X_train = X_all.loc[pu_mask]
    y_train = y_bot_all.loc[pu_mask]   # 1=bot, 0=normal (grey area removed)

    pos_count = int(y_train.sum())
    neg_count = int((y_train == 0).sum())
    log.info("  Full set: %d players", len(X_all))
    log.info("  PU training set: %d players (bot=%d, confirmed_normal=%d)",
             len(y_train), pos_count, neg_count)
    log.info("  Grey area excluded: %d players", int((~pu_mask).sum()))

    scale_pos_weight = neg_count / max(pos_count, 1)
    log.info("  Class imbalance ratio (scale_pos_weight): %.2f", scale_pos_weight)

    X_train_scaled = preprocessor.transform(X_train)

    param_distributions = {
        "n_estimators":     [200, 500, 1000],
        "max_depth":        [3, 5, 7],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 5, 10],
        "reg_alpha":        [0, 0.1, 1.0],
        "reg_lambda":       [1, 5, 10],
    }

    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=50,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    log.info("  Running RandomizedSearchCV (n_iter=50, cv=5) …")
    search.fit(X_train_scaled, y_train)

    log.info("  Best CV PR-AUC: %.4f", search.best_score_)
    log.info("  Best params: %s", search.best_params_)

    best_xgb = search.best_estimator_

    # Score ALL players in the trimmed dataset (not just the training subset)
    X_all_scaled  = preprocessor.transform(X_all)
    xgb_proba_full = best_xgb.predict_proba(X_all_scaled)[:, 1]

    joblib.dump(best_xgb, os.path.join(outputs_dir, "best_xgb.pkl"))
    pd.DataFrame(search.cv_results_).to_csv(
        os.path.join(outputs_dir, "xgb_tuning_results.csv"), index=False
    )

    return best_xgb, xgb_proba_full, common_ids


# ---------------------------------------------------------------------------
# Step 6 — Train Final IsolationForest
# ---------------------------------------------------------------------------

def train_best_models(X_scaled: np.ndarray,
                      best_if_params: dict) -> tuple[dict, dict]:
    """
    Retrain IsolationForest on the full (scaled/PCA'd) data.
    Returns (models_dict, raw_scores_dict) — higher score = more anomalous.
    """
    log.info("Step 6 — Training IsolationForest …")
    best_if = IsolationForest(**best_if_params)
    best_if.fit(X_scaled)
    if_scores = -best_if.score_samples(X_scaled)

    models = {"IsolationForest": best_if}
    scores = {"IsolationForest": if_scores}
    return models, scores


# ---------------------------------------------------------------------------
# Step 7 — Ensemble (XGBoost primary + unsupervised secondary)
# ---------------------------------------------------------------------------

def build_ensemble(scores: dict,
                   common_ids: pd.Index,
                   y_heuristic: pd.Series,
                   xgb_proba: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Percentile-rank each model (0–100, higher = more suspicious).

    Weights: XGB=0.80 (primary), IF=0.20 (secondary).
    IF auto-flip: if ROC-AUC < 0.4, scores are inverted and flipped.
    Anomaly flag: composite_score >= 85 (high-confidence threshold).
    """
    log.info("Step 7 — Building ensemble (Dynamic Duo: XGB + IF) …")
    if_scores = scores["IsolationForest"]
    if_pct    = rankdata(if_scores) / len(if_scores) * 100

    # Auto-detect IF score inversion: if AUC < 0.4, flip scores.
    if_auc = roc_auc_score(y_heuristic, if_pct)
    log.info("  IF AUC check: %.4f", if_auc)
    if if_auc < 0.4:
        log.warning("  IF scores appear inverted (AUC=%.4f) — flipping!", if_auc)
        if_pct = 100.0 - if_pct

    if xgb_proba is not None:
        xgb_pct   = rankdata(xgb_proba) / len(xgb_proba) * 100
        composite = 0.80 * xgb_pct + 0.20 * if_pct
        xgb_flag  = (xgb_pct >= 95).astype(int)
    else:
        xgb_pct   = np.zeros(len(if_pct))
        composite = if_pct.copy()
        xgb_flag  = np.zeros(len(if_pct), dtype=int)

    if_flag    = (if_pct >= 95).astype(int)
    is_anomaly = (composite >= 85).astype(int)

    ensemble_results = pd.DataFrame({
        "playerid":        common_ids,
        "composite_score": composite,
        "xgb_proba":       xgb_proba if xgb_proba is not None else np.nan,
        "xgb_pct":         xgb_pct,
        "if_pct":          if_pct,
        "is_anomaly":      is_anomaly,
        "xgb_flag":        xgb_flag,
        "if_flag":         if_flag,
        "heuristic_bot":   y_heuristic.reindex(common_ids).values,
    })

    n = is_anomaly.sum()
    log.info("  Anomalies flagged: %d (%.2f%%)", n, n / len(is_anomaly) * 100)

    percentile_scores = {
        "xgb_pct":   xgb_pct,
        "if_pct":    if_pct,
        "composite": composite,
    }
    return ensemble_results, percentile_scores
