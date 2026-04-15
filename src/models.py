"""
Phase 2 — Steps 4–7: Preprocessing, Hyperparameter Tuning, Training, Ensemble.

V4 "Dynamic Duo — No PCA" architecture:
- XGBoost (PRIMARY, weight 0.70): Semi-supervised PU Learning on heuristic labels.
- IsolationForest (SECONDARY, weight 0.30): Unsupervised anomaly detection.
- PCA removed: both models are tree-based and handle collinearity natively.
  Keeping PCA would (1) replace interpretable feature names with anonymous PC
  components, breaking SHAP explainability, and (2) compress low-variance anomaly
  signals (rare bot archetypes) into discarded components, reducing recall.
- Ensemble anomaly flag: composite_score >= 85 (high-confidence threshold).
"""

import logging
import time
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

log = logging.getLogger(__name__)

# Heavy-tailed features whose raw distributions span many orders of magnitude.
# Without log-compression, StandardScaler still leaves extreme outliers that
# dominate inter-feature variance and destabilise IsolationForest path lengths.
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

    **Mathematical justification (thesis reference)**
    Achievement-timing and count features follow power-law (Pareto) distributions:
    a small number of bots produce astronomically large values (e.g.
    `std_unlock_interval_sec` spans 6+ orders of magnitude).

    This transform is retained after PCA removal for one concrete reason:
    IsolationForest stability.  IF builds random trees by repeatedly choosing a
    feature and a random split point drawn *uniformly* from [min, max].  When a
    feature spans many orders of magnitude almost all random splits land in the
    sparse high-value tail, so the algorithm assigns anomalously short path lengths
    to outliers not because of genuine multi-feature isolation but purely because
    of skewed marginal density.  log1p compresses the range to roughly log-normal,
    making the uniform split distribution representative of actual data density and
    restoring path-length comparisons to their intended meaning.

    XGBoost is tree-based and learns its own thresholds, so it is invariant to
    any monotonic feature transformation; this step has no effect on XGBoost
    beyond ensuring consistent preprocessing between fit and predict.

    clip(lower=0) guards against negative values produced by median imputation
    or floating-point rounding before the pipeline runs.

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
# Step 4 — Preprocessing (no PCA)
# ---------------------------------------------------------------------------

def preprocess(X_raw: pd.DataFrame, save_path: str) -> tuple[pd.DataFrame, Pipeline, list]:
    """
    Fit Imputer → StandardScaler pipeline and persist it.

    **Why PCA was removed**
    Both XGBoost and IsolationForest are tree-based algorithms. Trees split on
    individual feature thresholds, so they are inherently invariant to linear
    rotations and handle correlated features without dimensionality reduction.
    PCA was inherited from earlier LOF/OCSVM models that *do* require a low-
    dimensional, spherical input space.  Keeping it caused two regressions:

      1. SHAP interpretability loss: feature names became 'PC1', 'PC2', …,
         hiding which real behaviours (playtime, unlock speed, …) drive each
         prediction.
      2. Anomaly signal compression: PCA maximises retained variance, which
         represents *normal* behaviour.  Rare-bot signals (low-variance by
         definition) risk being folded into discarded components.

    StandardScaler is still applied so that the median imputer's fill values
    remain on the same scale as real observations, and IsolationForest path-
    length comparisons are not biased by feature magnitude differences.

    Returns a DataFrame (not a numpy array) so that feature names propagate
    automatically into XGBoost's booster, SHAP plots, and feature-importance
    charts without any manual name tracking.
    """
    log.info("Step 4 — Preprocessing (Imputer + StandardScaler) …")
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    X_arr    = preprocessor.fit_transform(X_raw)
    X_scaled = pd.DataFrame(X_arr, index=X_raw.index, columns=X_raw.columns)
    log.info("  Scaled: %d players × %d features", *X_scaled.shape)

    feature_names = X_raw.columns.tolist()

    joblib.dump(preprocessor, save_path)
    log.info("  Preprocessor saved → %s", save_path)
    return X_scaled, preprocessor, feature_names


# ---------------------------------------------------------------------------
# Step 5 — Hyperparameter Tuning (IsolationForest)
# ---------------------------------------------------------------------------

def tune_models(X_scaled: pd.DataFrame,
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

    X_train_scaled = pd.DataFrame(
        preprocessor.transform(X_train),
        index=X_train.index, columns=X_train.columns,
    )

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
    X_all_scaled = pd.DataFrame(
        preprocessor.transform(X_all),
        index=X_all.index, columns=X_all.columns,
    )
    xgb_proba_full = best_xgb.predict_proba(X_all_scaled)[:, 1]

    joblib.dump(best_xgb, os.path.join(outputs_dir, "best_xgb.pkl"))
    pd.DataFrame(search.cv_results_).to_csv(
        os.path.join(outputs_dir, "xgb_tuning_results.csv"), index=False
    )

    return best_xgb, xgb_proba_full, common_ids


# ---------------------------------------------------------------------------
# Step 6 — Train Final IsolationForest
# ---------------------------------------------------------------------------

def train_best_models(X_scaled: pd.DataFrame,
                      best_if_params: dict) -> tuple[dict, dict]:
    """
    Retrain IsolationForest on the full scaled data.
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

    Weights: XGB=0.70 (primary), IF=0.30 (secondary).
    Anomaly flag: composite_score >= 85 (high-confidence threshold).
    """
    log.info("Step 7 — Building ensemble (Dynamic Duo: XGB + IF) …")
    if_scores = scores["IsolationForest"]
    # Negate score_samples so higher value = more anomalous (matches XGBoost convention).
    if_pct    = rankdata(if_scores) / len(if_scores) * 100

    if_auc = roc_auc_score(y_heuristic, if_pct)
    log.info("  IF ROC-AUC: %.4f", if_auc)

    if xgb_proba is not None:
        xgb_pct   = rankdata(xgb_proba) / len(xgb_proba) * 100
        composite = 0.70 * xgb_pct + 0.30 * if_pct
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


# ---------------------------------------------------------------------------
# Ensemble Weight Tuning — empirical search over (w_xgb, w_if) pairs
# ---------------------------------------------------------------------------

def tune_ensemble_weights(
    xgb_pct: np.ndarray,
    if_pct: np.ndarray,
    y_heuristic: pd.Series,
    outputs_dir: str,
    step: float = 0.05,
    anomaly_threshold: float = 85.0,
) -> pd.DataFrame:
    """
    Sweep XGBoost ensemble weight from 0.0 → 1.0 in increments of `step` and
    measure three metrics at each operating point:

      - Precision@100        : fraction of heuristic bots in the top-100 ranked players
      - PR-AUC               : average precision of composite_score vs. y_heuristic
      - High_Conflict_Cases  : stealth bots missed by heuristics but caught by ensemble
                               (is_anomaly == 1) & (y_heuristic == 0)

    The "optimal sweet spot" is defined as the weight that maximises Precision@100
    among all candidates where High_Conflict_Cases ≥ median(High_Conflict_Cases).
    This operationalises the thesis requirement: maximise bot-ranking precision
    *without* sacrificing too many stealth detections.

    Outputs
    -------
    outputs/ensemble_weight_metrics.csv  — full metric table
    outputs/plots/ensemble_weight_tuning.png — dual-axis line chart

    Parameters
    ----------
    xgb_pct           : percentile-ranked XGBoost probabilities (0–100), shape (N,)
    if_pct            : percentile-ranked IsolationForest scores (0–100), shape (N,)
    y_heuristic       : ground-truth heuristic bot labels (0/1), shape (N,)
    outputs_dir       : root outputs directory
    step              : grid step for xgb_weight (default 0.05)
    anomaly_threshold : composite_score cut-off for is_anomaly flag (default 85)

    Returns
    -------
    metrics_df : DataFrame with columns
                 [xgb_weight, if_weight, precision_at_100, pr_auc, high_conflict_cases]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score

    log.info("Ensemble weight tuning — sweeping xgb_weight 0.0 → 1.0 (step=%.2f) …", step)

    y_vals = y_heuristic.values  # numpy view for vectorised ops

    # ── Vectorised sweep ────────────────────────────────────────────────────
    # Build a 2-D matrix: rows = weight configs, cols = players.
    # xgb_weights[i] * xgb_pct[j] + if_weights[i] * if_pct[j]
    weights = np.arange(0.0, 1.0 + step / 2, step).clip(0.0, 1.0)  # shape (W,)
    weights = np.round(weights, 10)  # avoid floating-point fringe values

    # composite matrix: (W, N)
    composite_matrix = (
        weights[:, None] * xgb_pct[None, :]
        + (1.0 - weights)[:, None] * if_pct[None, :]
    )  # fully vectorised — no Python loop over rows

    # is_anomaly matrix: (W, N)
    is_anomaly_matrix = (composite_matrix >= anomaly_threshold).astype(int)

    # ── Metrics (one pass over weight dimension) ────────────────────────────
    rows = []
    for i, w_xgb in enumerate(weights):
        composite = composite_matrix[i]        # shape (N,)
        is_anom   = is_anomaly_matrix[i]       # shape (N,)

        # Precision@100 — top-100 by composite_score
        top100_idx = np.argpartition(composite, -100)[-100:]
        p_at_100   = float(y_vals[top100_idx].mean())

        # PR-AUC
        pr_auc = average_precision_score(y_vals, composite)

        # High-conflict: ensemble flags bot, heuristic says normal
        hcc = int(((is_anom == 1) & (y_vals == 0)).sum())

        rows.append({
            "xgb_weight":       round(float(w_xgb), 4),
            "if_weight":        round(float(1.0 - w_xgb), 4),
            "precision_at_100": round(p_at_100, 4),
            "pr_auc":           round(pr_auc, 4),
            "high_conflict_cases": hcc,
        })

    metrics_df = pd.DataFrame(rows)

    # ── Optimal sweet spot ──────────────────────────────────────────────────
    # Among weights where HCC >= median(HCC), choose argmax Precision@100.
    hcc_median  = metrics_df["high_conflict_cases"].median()
    candidates  = metrics_df[metrics_df["high_conflict_cases"] >= hcc_median]
    best_idx    = candidates["precision_at_100"].idxmax()
    best_weight = metrics_df.loc[best_idx, "xgb_weight"]
    best_p100   = metrics_df.loc[best_idx, "precision_at_100"]
    best_hcc    = metrics_df.loc[best_idx, "high_conflict_cases"]

    log.info(
        "  Optimal XGB weight: %.2f  (Precision@100=%.4f, HCC=%d)",
        best_weight, best_p100, best_hcc,
    )

    # ── Save CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(outputs_dir, "ensemble_weight_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    log.info("  Saved metric table → %s", csv_path)

    # ── Dual-axis chart ─────────────────────────────────────────────────────
    plots_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(11, 6))

    x = metrics_df["xgb_weight"].values

    # Left axis — precision metrics
    color_p100  = "#1f77b4"   # blue
    color_prauc = "#2ca02c"   # green
    ax1.plot(x, metrics_df["precision_at_100"].values,
             color=color_p100,  linewidth=2.0, marker="o", markersize=4,
             label="Precision@100")
    ax1.plot(x, metrics_df["pr_auc"].values,
             color=color_prauc, linewidth=2.0, marker="s", markersize=4,
             label="PR-AUC")
    ax1.set_xlabel("XGBoost Weight  ($w_{\\mathrm{XGB}}$,  $w_{\\mathrm{IF}} = 1 - w_{\\mathrm{XGB}}$)",
                   fontsize=12)
    ax1.set_ylabel("Score (0 – 1)", fontsize=12, color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Right axis — high-conflict cases
    ax2 = ax1.twinx()
    color_hcc = "#d62728"   # red
    ax2.plot(x, metrics_df["high_conflict_cases"].values,
             color=color_hcc, linewidth=2.0, linestyle="--", marker="^", markersize=4,
             label="High-Conflict Cases (stealth bots)")
    ax2.set_ylabel("High-Conflict Cases  (count)", fontsize=12, color=color_hcc)
    ax2.tick_params(axis="y", labelcolor=color_hcc)

    # Optimal sweet-spot marker
    ax1.axvline(best_weight, color="darkorange", linestyle=":", linewidth=2.0,
                label=f"Optimal: $w_{{\\mathrm{{XGB}}}}={best_weight:.2f}$")
    ax1.annotate(
        f"$w_{{\\mathrm{{XGB}}}}={best_weight:.2f}$\nP@100={best_p100:.3f}\nHCC={best_hcc}",
        xy=(best_weight, best_p100),
        xytext=(best_weight + 0.05, best_p100 - 0.12),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="darkorange"),
        color="darkorange",
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="lower left", fontsize=9, framealpha=0.85)

    plt.title(
        "Ensemble Weight Sensitivity Analysis\n"
        r"$\mathrm{composite} = w_{\mathrm{XGB}}\cdot\mathrm{xgb\_pct} "
        r"+ (1-w_{\mathrm{XGB}})\cdot\mathrm{if\_pct}$",
        fontsize=13,
    )
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, "ensemble_weight_tuning.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log.info("  Saved dual-axis chart → %s", plot_path)

    return metrics_df
