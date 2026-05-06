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
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split


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
    # reviewed_csv = os.path.join(REVIEWED_CSV)  # updated by auto_label.py
    # heuristic_df = integrate_human_labels(heuristic_df, reviewed_csv)

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

    # ── Train-Test Split (80-20) for reliable evaluation ──────────────────────
    train_ids, test_ids = train_test_split(
        common_ids.tolist(),
        test_size=0.20,
        random_state=42,
        stratify=y_heuristic.values
    )
    train_ids = pd.Index(train_ids)
    test_ids = pd.Index(test_ids)
    log.info("Train-Test split: %d train (80%%), %d test (20%%)", len(train_ids), len(test_ids))

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

    # ── Step 5a: Hyperparameter Tuning (IsolationForest) — on training set only ──
    X_if_train = X_if.loc[train_ids]
    y_heuristic_train = y_heuristic.loc[train_ids]
    best_if_params, tuning_results = tune_models(X_if_train, y_heuristic_train)
    tuning_path = os.path.join(OUTPUTS_DIR, "tuning_results.csv")
    tuning_results.to_csv(tuning_path, index=False)
    log.info("Saved → %s", tuning_path)

    # ── Step 6: Train Final IsolationForest — on full training set ──────────────
    models, scores_train = train_best_models(X_if_train, best_if_params)
    # Score both train and test
    if_model_full = models["IsolationForest"]
    if_scores_train = -if_model_full.score_samples(X_if_train)
    if_scores_test = -if_model_full.score_samples(X_if.loc[test_ids])
    log.info("  IF scoring: train=%d, test=%d", len(if_scores_train), len(if_scores_test))
    joblib.dump(models["IsolationForest"], os.path.join(OUTPUTS_DIR, "best_if.pkl"))
    log.info("Saved → %s", os.path.join(OUTPUTS_DIR, "best_if.pkl"))

    # ── Step 5b: XGBoost — raw features on training set (Path B) ────────────────
    X_raw_train = X_raw.loc[train_ids]
    X_raw_test = X_raw.loc[test_ids]
    y_heuristic_train = y_heuristic.loc[train_ids]
    y_heuristic_test = y_heuristic.loc[test_ids]
    y_normal_train = y_normal.loc[train_ids]
    best_xgb, xgb_proba_train, xgb_proba_test = train_xgboost_semisupervised(
        X_raw_train, X_raw_test, y_heuristic_train, y_heuristic_test, y_normal_train, OUTPUTS_DIR
    )
    log.info("  XGBoost scoring: train=%d, test=%d", len(xgb_proba_train), len(xgb_proba_test))

    # ── Step 7: Ensemble V2 (on test set only for evaluation) ──────────────────────────
    ensemble_results_test, percentile_scores_test = build_ensemble(
        {"IsolationForest": if_scores_test},
        test_ids,
        y_heuristic_test,
        xgb_proba=xgb_proba_test,
        dataset_type="test"
    )
    ensemble_path = os.path.join(OUTPUTS_DIR, "ensemble_results_test.csv")
    ensemble_results_test.to_csv(ensemble_path, index=False)
    log.info("Saved test ensemble → %s", ensemble_path)

    # Also build full ensemble for reporting (train + test combined)
    ensemble_results_full, percentile_scores_full = build_ensemble(
        {"IsolationForest": np.concatenate([if_scores_train, if_scores_test])},
        common_ids,
        y_heuristic,
        xgb_proba=np.concatenate([xgb_proba_train, xgb_proba_test]),
        dataset_type="full"
    )

    # ── Step 7b: Ensemble Weight Tuning (on test set only) ────────────────────────────────
    tune_ensemble_weights(
        xgb_pct     = percentile_scores_test["xgb_pct"],
        if_pct      = percentile_scores_test["if_pct"],
        y_heuristic = y_heuristic_test,
        outputs_dir = OUTPUTS_DIR,
        dataset_type="test"
    )

    # ── Active Learning: export high-conflict cases for human review ──────────
    # generate_review_sample(ensemble_results, feature_matrix.loc[common_ids], OUTPUTS_DIR)

    # ── Steps 8 & 9: Evaluate + SHAP (on test set for honest metrics) ──────────────
    evaluate(
        ensemble_results       = ensemble_results_test,
        percentile_scores      = percentile_scores_test,
        y_heuristic            = y_heuristic_test,
        feature_matrix         = feature_matrix.loc[test_ids],
        feature_names          = feature_names,
        X_raw                  = X_raw_test,          # unscaled, for SHAP interpretability
        best_xgb               = best_xgb,
        xgb_proba              = xgb_proba_test,
        original_feature_names = original_feature_names,
        outputs_dir            = OUTPUTS_DIR,
    )

    # --- Save model memory for Streamlit app ---
    # Use full dataset scores for model_memory (for Streamlit app)
    model_memory = {
        "feature_columns": original_feature_names,
        "baseline_size": int(len(common_ids)),
        "train_size": int(len(train_ids)),
        "test_size": int(len(test_ids)),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_reference_time": None if pd.isna(feature_reference_time) else pd.Timestamp(feature_reference_time).isoformat(),
        "sorted_raw_scores": {
            "IsolationForest": list(sorted(np.concatenate([if_scores_train, if_scores_test]))),
            "XGBoost": list(sorted(np.concatenate([xgb_proba_train, xgb_proba_test]))) if xgb_proba_test is not None else [],
        },
        "raw_scores": {
            "IsolationForest": list(np.concatenate([if_scores_train, if_scores_test])),
            "XGBoost": list(np.concatenate([xgb_proba_train, xgb_proba_test])) if xgb_proba_test is not None else [],
        },
    }
    joblib.dump(model_memory, os.path.join(OUTPUTS_DIR, "model_memory.pkl"))
    log.info("Saved → %s", os.path.join(OUTPUTS_DIR, "model_memory.pkl"))
    
    log.info("Pipeline complete. All outputs in %s/", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
