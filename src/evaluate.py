"""
Phase 2 — Steps 8 & 9: Evaluation metrics and SHAP explanations.

V2 changes:
- Model comparison table now includes XGBoost and reports PR-AUC alongside ROC-AUC.
- SHAP now uses XGBoost (TreeExplainer on XGBClassifier) instead of IsolationForest.
  TreeExplainer is more stable and interpretable for gradient-boosted trees.
- evaluate() signature updated: accepts best_xgb and xgb_proba, drops best_if.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: Precision@K
# ---------------------------------------------------------------------------

def precision_at_k(y_true: pd.Series, scores: np.ndarray, k: int) -> float:
    """Fraction of heuristic bots in the top-K highest-score accounts."""
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(y_true.values[top_k_idx].mean())


# ---------------------------------------------------------------------------
# Step 8 — Evaluation
# ---------------------------------------------------------------------------

def _model_comparison(y_heuristic: pd.Series,
                       percentile_scores: dict,
                       xgb_proba: np.ndarray | None,
                       ensemble_is_anomaly: np.ndarray) -> pd.DataFrame:
    score_map = {
        "XGBoost":         percentile_scores.get("xgb_pct", np.zeros(len(y_heuristic))),
        "IsolationForest": percentile_scores["if_pct"],
        "Ensemble":        percentile_scores["composite"],
    }

    rows = {}
    for model, scores in score_map.items():
        raw_prob = xgb_proba if (model == "XGBoost" and xgb_proba is not None) else None
        rows[model] = {
            "ROC-AUC":        roc_auc_score(y_heuristic, scores),
            "PR-AUC":         (average_precision_score(y_heuristic, raw_prob)
                               if raw_prob is not None
                               else average_precision_score(y_heuristic, scores / 100)),
            "Flagged Rate %": (ensemble_is_anomaly.mean() * 100
                               if model == "Ensemble"
                               else (scores >= 95).mean() * 100),
            "Precision@100":  precision_at_k(y_heuristic, scores, 100),
            "Precision@500":  precision_at_k(y_heuristic, scores, 500),
            "Precision@1000": precision_at_k(y_heuristic, scores, 1000),
        }

    return pd.DataFrame(rows).T.round(4)


# ---------------------------------------------------------------------------
# XGBoost-specific evaluation
# ---------------------------------------------------------------------------

def _evaluate_xgboost(best_xgb, X_scaled: np.ndarray,
                       y_heuristic: pd.Series,
                       original_feature_names: list,
                       plots_dir: str) -> None:
    """PR-AUC, optimal threshold, feature importance, and PR curve plot."""
    xgb_proba = best_xgb.predict_proba(X_scaled)[:, 1]

    pr_auc = average_precision_score(y_heuristic, xgb_proba)
    log.info("  XGBoost PR-AUC: %.4f", pr_auc)

    precisions, recalls, thresholds = precision_recall_curve(y_heuristic, xgb_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx   = f1_scores[:-1].argmax()
    best_thresh = thresholds[best_idx]
    log.info("  Optimal threshold (F1): %.4f", best_thresh)

    y_pred = (xgb_proba >= best_thresh).astype(int)
    log.info("\n%s", classification_report(y_heuristic, y_pred,
                                           target_names=["Normal", "Bot"]))

    # PR curve plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, label=f"XGBoost (PR-AUC={pr_auc:.3f})")
    ax.axvline(recalls[best_idx], color="red", linestyle="--",
               label=f"Optimal threshold={best_thresh:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — XGBoost")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "xgb_pr_curve.png"), dpi=150)
    plt.close()

    # Feature importance plot (XGBoost built-in)
    importances = best_xgb.feature_importances_
    
    # PCA removed — importances always align 1-to-1 with original_feature_names.
    fi_df = pd.DataFrame({
        "feature":    original_feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    log.info("\nTop-15 XGBoost Feature Importance:\n%s",
             fi_df.head(15).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 8))
    fi_df.head(15).plot.barh(x="feature", y="importance", ax=ax)
    ax.set_title("XGBoost Feature Importance (Top 15)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "xgb_feature_importance.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Step 9 — SHAP (XGBoost — TreeExplainer)
# ---------------------------------------------------------------------------

def _shap_plots(best_xgb, X_scaled: pd.DataFrame, feature_names: list,
                composite: np.ndarray, plots_dir: str) -> None:
    """
    SHAP summary, waterfall, and per-feature scatter for XGBoost.

    X_scaled is now a DataFrame (PCA removed), so feature_names are the real
    column names ('playtime_per_achievement', 'median_unlock_interval_sec', …).
    SHAP plots therefore display human-readable axes instead of 'PC1', 'PC2'.

    Uses native XGBoost pred_contribs to bypass TreeExplainer parser issues.
    """
    import xgboost as xgb

    SHAP_SAMPLE = 5_000
    rng = np.random.default_rng(42)
    shap_idx = rng.choice(len(X_scaled), min(SHAP_SAMPLE, len(X_scaled)), replace=False)
    # iloc preserves the DataFrame type and its column names
    X_shap = X_scaled.iloc[shap_idx]

    log.info("  Computing SHAP values natively via XGBoost on %d samples …", len(X_shap))

    try:
        booster = best_xgb.get_booster()
        # X_shap is a DataFrame — XGBoost DMatrix reads column names automatically.
        dmatrix = xgb.DMatrix(X_shap, feature_names=feature_names)

        # pred_contribs returns (n_samples, n_features + 1):
        # columns 0..n-1 are SHAP values, last column is the base score.
        contribs = booster.predict(dmatrix, pred_contribs=True)
        shap_values   = contribs[:, :-1]
        expected_value = contribs[0, -1]

        log.info("  [v] Successfully computed SHAP values via native XGBoost")

        # Plot 1: Global feature importance (bar/dot summary)
        fig, _ = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "shap_summary.png"), dpi=150)
        plt.close()
        log.info("  Saved shap_summary.png")

        # Plot 2: Waterfall for the most suspicious account in the SHAP sample
        top1_in_sample = int(np.argmax(composite[shap_idx]))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[top1_in_sample],
                base_values=expected_value,
                data=X_shap.iloc[top1_in_sample].values,
                feature_names=feature_names,
            ),
            show=False,
        )
        plt.savefig(
            os.path.join(plots_dir, "shap_waterfall.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
        log.info("  Saved shap_waterfall.png")

        # Plot 3: SHAP scatter for top-3 most important features
        importances = np.abs(shap_values).mean(0)
        top3_idx    = np.argsort(importances)[::-1][:3]
        top3_names  = []

        for fi in top3_idx:
            feat = feature_names[fi]
            top3_names.append(feat)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X_shap.iloc[:, fi].values, shap_values[:, fi], alpha=0.3, s=5)
            ax.set_xlabel(feat)
            ax.set_ylabel("SHAP value")
            ax.set_title(f"SHAP scatter: {feat}")
            plt.tight_layout()
            safe = feat.replace("/", "_").replace(" ", "_")
            plt.savefig(os.path.join(plots_dir, f"shap_scatter_{safe}.png"), dpi=150)
            plt.close()

        log.info("  Saved SHAP scatter plots for: %s", top3_names)

    except Exception as e:
        log.warning("  SHAP explanation failed: %s — skipping SHAP plots.", str(e))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate(ensemble_results: pd.DataFrame,
             percentile_scores: dict,
             y_heuristic: pd.Series,
             feature_matrix: pd.DataFrame,
             feature_names: list,
             X_scaled: np.ndarray,
             best_xgb,
             xgb_proba: np.ndarray | None,
             original_feature_names: list,
             outputs_dir: str) -> pd.DataFrame:
    """
    Steps 8 & 9: compute all evaluation metrics, save artefacts, generate plots.

    Parameters
    ----------
    best_xgb              : Trained XGBClassifier (used for SHAP + PR curve).
    xgb_proba             : XGBoost predicted probabilities on the common_ids set.
    original_feature_names: Raw feature column names for XGB importance chart.
    feature_names         : Feature names for SHAP plots (same as original after PCA removal).
    """
    plots_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    composite  = percentile_scores["composite"]
    is_anomaly = ensemble_results["is_anomaly"].values

    # ── 8.1 ROC-AUC and PR-AUC ───────────────────────────────────────────────
    log.info("Step 8 — Evaluation …")
    score_map = {
        "XGBoost":         percentile_scores.get("xgb_pct"),
        "IsolationForest": percentile_scores["if_pct"],
        "Ensemble":        composite,
    }
    for model, scores in score_map.items():
        if scores is None:
            continue
        auc = roc_auc_score(y_heuristic, scores)
        log.info("  ROC-AUC %-16s %.4f", model, auc)

    # ── 8.2 Precision@K (Ensemble) ────────────────────────────────────────────
    for k in [100, 500, 1000]:
        p = precision_at_k(y_heuristic, composite, k)
        log.info("  Precision@%-5d (Ensemble) %.4f", k, p)

    # ── 8.3 Model comparison table ────────────────────────────────────────────
    comparison = _model_comparison(y_heuristic, percentile_scores, xgb_proba, is_anomaly)
    log.info("\n%s", comparison.to_string())
    comparison.to_csv(os.path.join(outputs_dir, "model_comparison.csv"))

    # ── 8.4 XGBoost-specific metrics ─────────────────────────────────────────
    if best_xgb is not None and xgb_proba is not None:
        log.info("  XGBoost detailed evaluation …")
        _evaluate_xgboost(best_xgb, X_scaled, y_heuristic,
                          original_feature_names, plots_dir)

    # ── 8.5 Top-50 flagged profile analysis ──────────────────────────────────
    top50_ids     = set(ensemble_results.nlargest(50, "composite_score")["playerid"])
    flagged_feats = feature_matrix.loc[feature_matrix.index.isin(top50_ids)]
    normal_feats  = feature_matrix.loc[~feature_matrix.index.isin(top50_ids)]
    profile = pd.DataFrame({
        "Flagged (mean)": flagged_feats.mean(),
        "Normal (mean)":  normal_feats.mean(),
        "Ratio":          flagged_feats.mean() / normal_feats.mean().replace(0, np.nan),
    }).round(3)
    log.info("\nTop-50 Flagged vs Normal:\n%s", profile.to_string())
    profile.to_csv(os.path.join(outputs_dir, "top50_flagged_profiles.csv"))

    # ── Step 9: SHAP ──────────────────────────────────────────────────────────
    log.info("Step 9 — SHAP explanations …")
    if best_xgb is not None:
        try:
            _shap_plots(best_xgb, X_scaled, feature_names, composite, plots_dir)
        except Exception as e:
            log.warning("  SHAP explanation failed: %s — skipping SHAP plots.", str(e))
    else:
        log.warning("  XGBoost not available — skipping SHAP.")

    return comparison
