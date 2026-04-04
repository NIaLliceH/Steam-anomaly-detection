"""
Steam Anomaly Detection — Main Pipeline Orchestrator (V2)
Executes Steps 2–9. Step 1 (data prep) is handled by src/data_prep.py.

V2 pipeline order:
  2. Heuristic Labels V2    (AND logic, 3 bot archetypes, Bug #3 fix)
  3. Feature Engineering    (Bug #2 fix: int-safe library lookup)
  4. Preprocessing          (Imputer + RobustScaler + PCA 90%)
  5a. Unsupervised tuning   (IF / LOF / OCSVM grid search)
  5b. XGBoost training      (semi-supervised, RandomizedSearchCV, PRIMARY)
  6. Final unsupervised     (retrain on full data)
  7. Ensemble V2            (XGB=0.50, LOF=0.30, IF=0.15+flip-if-inverted, SVM=0.05)
  8-9. Evaluation + SHAP    (XGBoost-based SHAP, PR-AUC, feature importance)
"""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from features import (
    add_time_components,
    build_feature_matrix,
    build_heuristic_labels,
    build_player_library,
)
from models import (
    apply_log_transform,
    build_ensemble,
    preprocess,
    train_best_models,
    train_xgboost_semisupervised,
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

    log.info("Building player library lookup …")
    player_library = build_player_library(purchased)

    # ── Step 2: Heuristic Labels V2 ───────────────────────────────────────────
    heuristic_df = build_heuristic_labels(history, reviews, player_library)
    heuristic_path = os.path.join(OUTPUTS_DIR, "heuristic_labels.csv")
    heuristic_df.to_csv(heuristic_path)
    log.info("Saved → %s", heuristic_path)

    # ── Step 3: Feature Engineering ──────────────────────────────────────────
    feature_matrix = build_feature_matrix(
        history, reviews, players, purchased, player_library
    )
    fm_path = os.path.join(OUTPUTS_DIR, "feature_matrix.csv")
    feature_matrix.to_csv(fm_path)
    log.info("Saved → %s", fm_path)

    # ── Data Trimming — remove sparse/ghost accounts before modelling ─────────
    # Ghost accounts (0-1 achievements, no library) dominate the 196k dataset.
    # Their presence causes IQR ≈ 0 for most features → StandardScaler (and the
    # old RobustScaler) blows up the scale for real players → 98%+ variance in
    # PC1 and ~6% Precision.  Trimming to players with ≥ 10 achievements AND
    # ≥ 1 owned game retains meaningful signal while removing noise.
    n_before = len(feature_matrix)
    feature_matrix = feature_matrix[
        (feature_matrix["total_achievements"] >= 10)
        & (feature_matrix["library_size"] >= 1)
    ]
    log.info("Data trimming: %d → %d players (removed %d ghost accounts)",
             n_before, len(feature_matrix), n_before - len(feature_matrix))

    # ── Step 4: Align + Preprocess (with PCA) ────────────────────────────────
    common_ids  = feature_matrix.index.intersection(heuristic_df.index)
    X_raw       = feature_matrix.loc[common_ids]
    y_heuristic = heuristic_df.loc[common_ids, "heuristic_bot"]
    y_normal    = heuristic_df.loc[common_ids, "heuristic_normal"]
    original_feature_names = X_raw.columns.tolist()
    log.info("Common players for modelling: %d", len(common_ids))

    # Log-transform heavy-tailed features before entering the sklearn pipeline.
    # Without this, std_unlock_interval_sec (reaches millions) causes PCA to
    # collapse: PC1 absorbs >92% of variance, leaving LOF/OCSVM with 1D data.
    X_log = apply_log_transform(X_raw)

    X_scaled, preprocessor, pca_feature_names = preprocess(
        X_log, save_path=os.path.join(OUTPUTS_DIR, "preprocessor.pkl")
    )

    # ── Step 5a: Hyperparameter Tuning (unsupervised) ─────────────────────────
    best_if_params, best_lof_params, best_svm_params, tuning_results = tune_models(
        X_scaled, y_heuristic
    )
    tuning_path = os.path.join(OUTPUTS_DIR, "tuning_results.csv")
    tuning_results.to_csv(tuning_path, index=False)
    log.info("Saved → %s", tuning_path)

    # ── Step 5b: XGBoost (PRIMARY model) ─────────────────────────────────────
    best_xgb, xgb_proba, _ = train_xgboost_semisupervised(
        X_log, y_heuristic, y_normal, preprocessor, OUTPUTS_DIR
    )

    # ── Step 6: Train Final Unsupervised Models ───────────────────────────────
    _, scores = train_best_models(
        X_scaled, best_if_params, best_lof_params, best_svm_params
    )

    # ── Step 7: Ensemble V2 ───────────────────────────────────────────────────
    ensemble_results, percentile_scores = build_ensemble(
        scores, common_ids, y_heuristic, xgb_proba=xgb_proba
    )
    ensemble_path = os.path.join(OUTPUTS_DIR, "ensemble_results.csv")
    ensemble_results.to_csv(ensemble_path, index=False)
    log.info("Saved → %s", ensemble_path)

    # ── Steps 8 & 9: Evaluate + SHAP ──────────────────────────────────────────
    evaluate(
        ensemble_results       = ensemble_results,
        percentile_scores      = percentile_scores,
        y_heuristic            = y_heuristic,
        feature_matrix         = feature_matrix.loc[common_ids],
        feature_names          = pca_feature_names,   # PCA components for SHAP
        X_scaled               = X_scaled,
        best_xgb               = best_xgb,
        xgb_proba              = xgb_proba,
        original_feature_names = original_feature_names,  # raw names for XGB importance
        outputs_dir            = OUTPUTS_DIR,
    )

    log.info("Pipeline complete. All outputs in %s/", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
