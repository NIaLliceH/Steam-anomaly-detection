"""
Phase 2 — Steps 8 & 9: Evaluation metrics and SHAP explanations.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 8.2 helper
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
                       ensemble_is_anomaly: np.ndarray) -> pd.DataFrame:
    if_pct    = percentile_scores["if_pct"]
    lof_pct   = percentile_scores["lof_pct"]
    svm_pct   = percentile_scores["svm_pct"]
    composite = percentile_scores["composite"]

    score_map = {
        "IsolationForest": if_pct,
        "LOF":             lof_pct,
        "OCSVM":           svm_pct,
        "Ensemble":        composite,
    }
    metrics = {m: roc_auc_score(y_heuristic, s) for m, s in score_map.items()}
    flagged_rates = {
        "IsolationForest": (if_pct  >= 95).mean(),
        "LOF":             (lof_pct >= 95).mean(),
        "OCSVM":           (svm_pct >= 95).mean(),
        "Ensemble":        ensemble_is_anomaly.mean(),
    }
    comparison = pd.DataFrame({
        "ROC-AUC":        metrics,
        "Flagged Rate %": {k: v * 100 for k, v in flagged_rates.items()},
        "Precision@100":  {m: precision_at_k(y_heuristic, score_map[m], 100)  for m in score_map},
        "Precision@500":  {m: precision_at_k(y_heuristic, score_map[m], 500)  for m in score_map},
        "Precision@1000": {m: precision_at_k(y_heuristic, score_map[m], 1000) for m in score_map},
    })
    return comparison


# ---------------------------------------------------------------------------
# Step 9 — SHAP
# ---------------------------------------------------------------------------

def _shap_plots(best_if, X_scaled: np.ndarray, feature_names: list,
                composite: np.ndarray, plots_dir: str) -> None:
    SHAP_SAMPLE = 10_000
    rng = np.random.default_rng(42)
    shap_idx = rng.choice(len(X_scaled), min(SHAP_SAMPLE, len(X_scaled)), replace=False)
    X_shap = X_scaled[shap_idx]

    log.info("  Computing SHAP values on %d samples …", len(X_shap))
    explainer   = shap.TreeExplainer(best_if)
    shap_values = explainer.shap_values(X_shap)

    # Plot 1: Global feature importance (bar)
    fig, _ = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        pd.DataFrame(X_shap, columns=feature_names),
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary.png"), dpi=150)
    plt.close()
    log.info("  Saved shap_summary.png")

    # Plot 2: Waterfall for the most suspicious account within the shap sample
    top1_in_sample = int(np.argmax(composite[shap_idx]))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[top1_in_sample],
            base_values=explainer.expected_value,
            data=X_shap[top1_in_sample],
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
    top3_idx = np.argsort(importances)[::-1][:3]
    for fi in top3_idx:
        feat = feature_names[fi]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X_shap[:, fi], shap_values[:, fi], alpha=0.3, s=5)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP scatter: {feat}")
        plt.tight_layout()
        safe = feat.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(plots_dir, f"shap_scatter_{safe}.png"), dpi=150)
        plt.close()
    top3_names = [feature_names[i] for i in top3_idx]
    log.info("  Saved SHAP scatter plots for: %s", top3_names)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate(ensemble_results: pd.DataFrame,
             percentile_scores: dict,
             y_heuristic: pd.Series,
             feature_matrix: pd.DataFrame,
             feature_names: list,
             X_scaled: np.ndarray,
             best_if,
             outputs_dir: str) -> pd.DataFrame:
    """
    Steps 8 & 9: compute all evaluation metrics, save artefacts, generate plots.

    Returns the model comparison DataFrame.
    """
    plots_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    composite  = percentile_scores["composite"]
    is_anomaly = ensemble_results["is_anomaly"].values

    # ── 8.1 ROC-AUC ──────────────────────────────────────────────────────────
    log.info("Step 8 — Evaluation …")
    score_map = {
        "IsolationForest": percentile_scores["if_pct"],
        "LOF":             percentile_scores["lof_pct"],
        "OCSVM":           percentile_scores["svm_pct"],
        "Ensemble":        composite,
    }
    for model, scores in score_map.items():
        auc = roc_auc_score(y_heuristic, scores)
        log.info("  ROC-AUC %-16s %.4f", model, auc)

    # ── 8.2 Precision@K ──────────────────────────────────────────────────────
    for k in [100, 500, 1000]:
        p = precision_at_k(y_heuristic, composite, k)
        log.info("  Precision@%-5d (Ensemble) %.4f", k, p)

    # ── 8.3 Model comparison table ───────────────────────────────────────────
    comparison = _model_comparison(y_heuristic, percentile_scores, is_anomaly)
    log.info("\n%s", comparison.round(4).to_string())
    comparison.to_csv(os.path.join(outputs_dir, "model_comparison.csv"))

    # ── 8.4 Top-50 flagged profile analysis ──────────────────────────────────
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

    # ── Step 9: SHAP ─────────────────────────────────────────────────────────
    log.info("Step 9 — SHAP explanations …")
    _shap_plots(best_if, X_scaled, feature_names, composite, plots_dir)

    return comparison
