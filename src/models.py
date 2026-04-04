"""
Phase 2 — Steps 4–7: Preprocessing, Hyperparameter Tuning, Training, Ensemble.

V2 changes:
- preprocess() now includes PCA (90% variance retained) after RobustScaler.
  PCA reduces the 25-feature space to ~8-12 components, removing noise and
  multicollinearity that hurt LOF and OCSVM distance calculations.
- train_xgboost_semisupervised() is the new PRIMARY model.  XGBoost is trained
  on the refined heuristic labels (V2 AND logic) via RandomizedSearchCV.
- build_ensemble() is updated:
    * XGBoost score gets weight 0.50 (new primary).
    * LOF keeps weight 0.30.
    * IF drops to 0.15 (was 0.50) — IF ROC-AUC was 0.0052, indicating an
      inverted score; the ensemble now auto-detects and flips IF if needed.
    * OCSVM keeps weight 0.05.
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
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

log = logging.getLogger(__name__)

# LOF and OCSVM are O(N²) — cap training size to avoid OOM
_LOF_SVM_SAMPLE = 20_000

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


def _draw_sample(X: np.ndarray, n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_sample, sample_indices) of size min(n, len(X))."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    return X[idx], idx


# ---------------------------------------------------------------------------
# Step 4 — Preprocessing with PCA
# ---------------------------------------------------------------------------

def preprocess(X_raw: pd.DataFrame, save_path: str,
               pca_variance: float = 0.90) -> tuple[np.ndarray, Pipeline, list]:
    """
    Fit Imputer → RobustScaler → PCA pipeline and persist it.

    PCA retains `pca_variance` fraction of explained variance (default 90%).
    This typically reduces 25 features to 8-12 PCA components, which:
    - Removes noise and multicollinearity (correlated speed/concentration features)
    - Improves LOF/OCSVM distance accuracy in lower-dimensional space
    - Cuts LOF/OCSVM compute time
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
# Step 5 — Hyperparameter Tuning (unsupervised models)
# ---------------------------------------------------------------------------

def _tune_isolation_forest(X_scaled: np.ndarray,
                            y: pd.Series) -> tuple[pd.DataFrame, dict]:
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
        auc = roc_auc_score(y, scores)
        rows.append({**params, "roc_auc": auc, "runtime_s": time.time() - t0})

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    best = df.iloc[0].drop(["roc_auc", "runtime_s"]).to_dict()
    best["n_estimators"] = int(best["n_estimators"])
    best["random_state"]  = int(best["random_state"])
    log.info("    Best IF  ROC-AUC=%.4f  %s", df.iloc[0]["roc_auc"], best)
    return df, best


def _tune_lof(X_scaled: np.ndarray, X_sample: np.ndarray,
              y: pd.Series) -> tuple[pd.DataFrame, dict]:
    log.info("  Tuning LOF (novelty=True, sample=%d) …", len(X_sample))
    param_grid = {
        "n_neighbors":   [10, 20, 30, 50],
        "contamination": [0.02, 0.05, 0.10],
        "metric":        ["euclidean", "manhattan"],
    }
    rows = []
    for params in ParameterGrid(param_grid):
        t0 = time.time()
        m = LocalOutlierFactor(novelty=True, **params)
        m.fit(X_sample)
        scores = -m.score_samples(X_scaled)
        auc = roc_auc_score(y, scores)
        rows.append({**params, "roc_auc": auc, "runtime_s": time.time() - t0})

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    best = df.iloc[0].drop(["roc_auc", "runtime_s"]).to_dict()
    best["n_neighbors"] = int(best["n_neighbors"])
    log.info("    Best LOF ROC-AUC=%.4f  %s", df.iloc[0]["roc_auc"], best)
    return df, best


def _tune_ocsvm(X_scaled: np.ndarray, X_sample: np.ndarray,
                y: pd.Series) -> tuple[pd.DataFrame, dict]:
    log.info("  Tuning OneClassSVM (sample=%d) …", len(X_sample))
    param_grid = {
        "kernel": ["rbf"],
        "gamma":  ["scale", "auto", 0.001, 0.01],
        "nu":     [0.01, 0.05, 0.10],
    }
    rows = []
    for params in ParameterGrid(param_grid):
        t0 = time.time()
        m = OneClassSVM(**params)
        m.fit(X_sample)
        scores = -m.score_samples(X_scaled)
        auc = roc_auc_score(y, scores)
        rows.append({**params, "roc_auc": auc, "runtime_s": time.time() - t0})

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    best = df.iloc[0].drop(["roc_auc", "runtime_s"]).to_dict()
    log.info("    Best SVM ROC-AUC=%.4f  %s", df.iloc[0]["roc_auc"], best)
    return df, best


def tune_models(X_scaled: np.ndarray,
                y_heuristic: pd.Series) -> tuple[dict, dict, dict, pd.DataFrame]:
    """
    Grid-search IF / LOF / OCSVM and return best params + tuning results.
    ROC-AUC here measures fit to heuristic labels (pseudo ground truth), not
    true anomaly detection accuracy.
    """
    log.info("Step 5 — Hyperparameter tuning …")
    X_sample, _ = _draw_sample(X_scaled, _LOF_SVM_SAMPLE)

    if_df,  best_if  = _tune_isolation_forest(X_scaled, y_heuristic)
    lof_df, best_lof = _tune_lof(X_scaled, X_sample, y_heuristic)
    svm_df, best_svm = _tune_ocsvm(X_scaled, X_sample, y_heuristic)

    tuning_results = pd.concat([
        if_df.assign(model="IsolationForest"),
        lof_df.assign(model="LOF"),
        svm_df.assign(model="OCSVM"),
    ], ignore_index=True)

    return best_if, best_lof, best_svm, tuning_results


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
# Step 6 — Train Final Unsupervised Models
# ---------------------------------------------------------------------------

def train_best_models(X_scaled: np.ndarray,
                      best_if_params: dict,
                      best_lof_params: dict,
                      best_svm_params: dict) -> tuple[dict, dict]:
    """
    Retrain each unsupervised model on the full (scaled/PCA'd) data.
    LOF and OCSVM still train on a 20k sample to stay within memory.
    Returns (models_dict, raw_scores_dict) — higher score = more anomalous.
    """
    log.info("Step 6 — Training final models …")
    X_sample, _ = _draw_sample(X_scaled, _LOF_SVM_SAMPLE)

    log.info("  IsolationForest …")
    best_if = IsolationForest(**best_if_params)
    best_if.fit(X_scaled)
    if_scores = -best_if.score_samples(X_scaled)

    log.info("  LOF …")
    best_lof = LocalOutlierFactor(novelty=True, **best_lof_params)
    best_lof.fit(X_sample)
    lof_scores = -best_lof.score_samples(X_scaled)

    log.info("  OneClassSVM …")
    best_svm = OneClassSVM(**best_svm_params)
    best_svm.fit(X_sample)
    svm_scores = -best_svm.score_samples(X_scaled)

    models = {"IsolationForest": best_if, "LOF": best_lof, "OCSVM": best_svm}
    scores = {"IsolationForest": if_scores, "LOF": lof_scores, "OCSVM": svm_scores}
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
    XGBoost is the primary model (weight 0.50).
    IF weight is reduced to 0.15; auto-flip if AUC < 0.4 indicates inversion.

    Weights: XGB=0.50, LOF=0.30, IF=0.15, OCSVM=0.05
    Flag threshold: top 5% per model; anomaly if ≥ 2 models agree.
    """
    log.info("Step 7 — Building ensemble …")
    if_scores  = scores["IsolationForest"]
    lof_scores = scores["LOF"]
    svm_scores = scores["OCSVM"]

    if_pct  = rankdata(if_scores)  / len(if_scores)  * 100
    lof_pct = rankdata(lof_scores) / len(lof_scores) * 100
    svm_pct = rankdata(svm_scores) / len(svm_scores) * 100

    # Auto-detect IF score inversion (Bug #1): if AUC < 0.4, flip scores.
    if_auc = roc_auc_score(y_heuristic, if_pct)
    log.info("  IF AUC check: %.4f", if_auc)
    if if_auc < 0.4:
        log.warning("  IF scores appear inverted (AUC=%.4f) — flipping!", if_auc)
        if_pct = 100.0 - if_pct

    if xgb_proba is not None:
        xgb_pct  = rankdata(xgb_proba) / len(xgb_proba) * 100
        composite = (
            0.50 * xgb_pct +
            0.30 * lof_pct +
            0.15 * if_pct  +
            0.05 * svm_pct
        )
        xgb_flag = (xgb_pct >= 95).astype(int)
    else:
        xgb_pct   = np.zeros(len(if_pct))
        composite = 0.50 * lof_pct + 0.35 * if_pct + 0.15 * svm_pct
        xgb_flag  = np.zeros(len(if_pct), dtype=int)

    if_flag  = (if_pct  >= 95).astype(int)
    lof_flag = (lof_pct >= 95).astype(int)
    svm_flag = (svm_pct >= 95).astype(int)
    vote_count = xgb_flag + lof_flag + if_flag + svm_flag
    is_anomaly = (vote_count >= 2).astype(int)

    ensemble_results = pd.DataFrame({
        "playerid":        common_ids,
        "composite_score": composite,
        "xgb_proba":       xgb_proba if xgb_proba is not None else np.nan,
        "xgb_pct":         xgb_pct,
        "if_pct":          if_pct,
        "lof_pct":         lof_pct,
        "svm_pct":         svm_pct,
        "vote_count":      vote_count,
        "is_anomaly":      is_anomaly,
        "xgb_flag":        xgb_flag,
        "if_flag":         if_flag,
        "lof_flag":        lof_flag,
        "svm_flag":        svm_flag,
        "heuristic_bot":   y_heuristic.reindex(common_ids).values,
    })

    n = is_anomaly.sum()
    log.info("  Anomalies flagged: %d (%.2f%%)", n, n / len(is_anomaly) * 100)

    percentile_scores = {
        "xgb_pct":   xgb_pct,
        "if_pct":    if_pct,
        "lof_pct":   lof_pct,
        "svm_pct":   svm_pct,
        "composite": composite,
    }
    return ensemble_results, percentile_scores
