"""
Steam Anomaly Detection — Main Pipeline Orchestrator (V2)
Executes Steps 2–9. Step 1 (data prep) is handled by src/data_prep.py.

V3 "Dynamic Duo" pipeline:
  2. Heuristic Labels V2    (AND logic, 3 bot archetypes)
  3. Feature Engineering    (int-safe library lookup, NaN for missing library)
  4. Preprocessing          (Imputer + StandardScaler + PCA 90%)
  5a. IF tuning             (grid search on IsolationForest only)
  5b. XGBoost training      (semi-supervised PU Learning, PRIMARY)
  6. Final IF training      (retrain on full data)
  7. Ensemble V3            (XGB=0.70, IF=0.30+auto-flip, threshold=85)
  8-9. Evaluation + SHAP    (XGBoost-based SHAP, PR-AUC, feature importance)
"""

import logging
import os
import sys
import joblib
import pandas as pd
from datetime import datetime


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from active_learning import generate_review_sample, integrate_human_labels
from features import (
    add_time_components,
    build_feature_matrix,
    build_heuristic_labels,
)
from models import (
    apply_log_transform,
    build_ensemble,
    preprocess,
    train_best_models,
    train_xgboost_semisupervised,
    tune_ensemble_weights,
    tune_models,
)
from evaluate import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
REVIEWED_CSV = os.path.join(os.path.dirname(__file__), "data", "reviewed.csv")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parquets() -> tuple:
    log.info("Loading processed parquet files from %s …", PROCESSED_DIR)
    history   = pd.read_parquet(os.path.join(PROCESSED_DIR, "history.parquet"))
    players   = pd.read_parquet(os.path.join(PROCESSED_DIR, "players.parquet"))
    reviews   = pd.read_parquet(os.path.join(PROCESSED_DIR, "reviews.parquet"))
    purchased = pd.read_parquet(os.path.join(PROCESSED_DIR, "purchased.parquet"))
    log.info("  history:   %d rows  |  columns: %s", len(history),   history.columns.tolist())
    log.info("  players:   %d rows  |  columns: %s", len(players),   players.columns.tolist())
    log.info("  reviews:   %d rows  |  columns: %s", len(reviews),   reviews.columns.tolist())
    log.info("  purchased: %d rows  |  columns: %s", len(purchased), purchased.columns.tolist())
    return history, players, reviews, purchased


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUTS_DIR, "plots"), exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    history, players, reviews, purchased = load_parquets()
    history = add_time_components(history)
    feature_reference_time = pd.to_datetime(history["date_acquired"], errors="coerce").max()
    log.info("Feature reference timestamp (last achievement record): %s", feature_reference_time)

    # ── Step 3: Feature Engineering (FIRST — heuristic labels reuse results) ──
    feature_matrix = build_feature_matrix(
        history, reviews, players, purchased, feature_reference_time,
    )
    fm_path = os.path.join(OUTPUTS_DIR, "feature_matrix.csv")
    feature_matrix.to_csv(fm_path)
    log.info("Saved → %s", fm_path)

    # ── Step 2: Heuristic Labels (consumes pre-computed feature_matrix) ───────
    heuristic_df = build_heuristic_labels(feature_matrix)

    # ── Active Learning: integrate human overrides (if reviewed.csv exists) ───
    reviewed_csv = os.path.join(REVIEWED_CSV)  # updated by auto_label.py
    heuristic_df = integrate_human_labels(heuristic_df, reviewed_csv)

    heuristic_path = os.path.join(OUTPUTS_DIR, "heuristic_labels.csv")
    heuristic_df.to_csv(heuristic_path)
    log.info("Saved → %s", heuristic_path)

    # ── Step 4: Align ─────────────────────────────────────────────────────────
    common_ids  = feature_matrix.index.intersection(heuristic_df.index)
    X_raw       = feature_matrix.loc[common_ids]
    y_heuristic = heuristic_df.loc[common_ids, "heuristic_bot"]
    y_normal    = heuristic_df.loc[common_ids, "heuristic_normal"]
    original_feature_names = X_raw.columns.tolist()
    log.info("Common players for modelling: %d", len(common_ids))

    # ── Path A: IsolationForest — log-transform → impute → scale ─────────────
    # IF uses random splits on individual features; extreme outliers (6+ orders
    # of magnitude) make uniform split sampling unrepresentative of true density.
    # log1p re-balances the marginal distributions.  SimpleImputer + StandardScaler
    # are then applied so path-length comparisons are not biased by magnitude.
    X_log = apply_log_transform(X_raw)
    X_if, preprocessor_if, feature_names = preprocess(
        X_log, save_path=os.path.join(OUTPUTS_DIR, "preprocessor.pkl")
    )

    # ── Path B: XGBoost — raw features, no imputer, no scaler ────────────────
    # XGBoost handles NaN natively via sparsity-aware split finding; imputing
    # destroys the missing-data signal (e.g. NaN review features for players
    # with no reviews is informative, not a gap to fill).  XGBoost is also
    # invariant to monotonic transforms, so neither log1p nor scaling helps.
    # X_raw is passed directly to train_xgboost_semisupervised below.

    # ── Step 5a: Hyperparameter Tuning (IsolationForest) ─────────────────────
    best_if_params, tuning_results = tune_models(X_if, y_heuristic)
    tuning_path = os.path.join(OUTPUTS_DIR, "tuning_results.csv")
    tuning_results.to_csv(tuning_path, index=False)
    log.info("Saved → %s", tuning_path)

    # ── Step 6: Train Final IsolationForest ──────────────────────────────────
    models, scores = train_best_models(X_if, best_if_params)
    joblib.dump(models["IsolationForest"], os.path.join(OUTPUTS_DIR, "best_if.pkl"))
    log.info("Saved → %s", os.path.join(OUTPUTS_DIR, "best_if.pkl"))

    # ── Step 5b: XGBoost — raw features (Path B) ─────────────────────────────
    best_xgb, xgb_proba, _ = train_xgboost_semisupervised(
        X_raw, y_heuristic, y_normal, OUTPUTS_DIR
    )

    # ── Step 7: Ensemble V2 ───────────────────────────────────────────────────
    ensemble_results, percentile_scores = build_ensemble(
        scores, common_ids, y_heuristic, xgb_proba=xgb_proba
    )
    ensemble_path = os.path.join(OUTPUTS_DIR, "ensemble_results.csv")
    ensemble_results.to_csv(ensemble_path, index=False)
    log.info("Saved → %s", ensemble_path)

    # ── Step 7b: Ensemble Weight Tuning ──────────────────────────────────────
    tune_ensemble_weights(
        xgb_pct     = percentile_scores["xgb_pct"],
        if_pct      = percentile_scores["if_pct"],
        y_heuristic = y_heuristic,
        outputs_dir = OUTPUTS_DIR,
    )

    # ── Active Learning: export high-conflict cases for human review ──────────
    generate_review_sample(ensemble_results, feature_matrix.loc[common_ids], OUTPUTS_DIR)

    # ── Steps 8 & 9: Evaluate + SHAP ──────────────────────────────────────────
    evaluate(
        ensemble_results       = ensemble_results,
        percentile_scores      = percentile_scores,
        y_heuristic            = y_heuristic,
        feature_matrix         = feature_matrix.loc[common_ids],
        feature_names          = feature_names,
        X_raw                  = X_raw,          # unscaled, for SHAP interpretability
        best_xgb               = best_xgb,
        xgb_proba              = xgb_proba,
        original_feature_names = original_feature_names,
        outputs_dir            = OUTPUTS_DIR,
    )

    # --- Save model memory for Streamlit app ---
    model_memory = {
        "feature_columns": original_feature_names,
        "baseline_size": int(len(common_ids)),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_reference_time": None if pd.isna(feature_reference_time) else pd.Timestamp(feature_reference_time).isoformat(),
        "sorted_raw_scores": {
            "IsolationForest": list(sorted(scores["IsolationForest"])) if "IsolationForest" in scores else [],
            "XGBoost": list(sorted(xgb_proba)) if xgb_proba is not None else [],
        },
        "raw_scores": {
            "IsolationForest": list(scores["IsolationForest"]) if "IsolationForest" in scores else [],
            "XGBoost": list(xgb_proba) if xgb_proba is not None else [],
        },
    }
    joblib.dump(model_memory, os.path.join(OUTPUTS_DIR, "model_memory.pkl"))
    log.info("Saved → %s", os.path.join(OUTPUTS_DIR, "model_memory.pkl"))
    
    log.info("Pipeline complete. All outputs in %s/", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
