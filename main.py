"""
Steam Anomaly Detection — Main Pipeline Orchestrator
Executes Steps 2–9. Step 1 (data prep) is handled by src/data_prep.py.
"""

import logging
import os
import sys

import pandas as pd

# Make src/ importable without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from features import (
    add_time_components,
    build_feature_matrix,
    build_heuristic_labels,
    build_player_library,
)
from models import (
    build_ensemble,
    preprocess,
    train_best_models,
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

    # ── Load ─────────────────────────────────────────────────────────────────
    history, players, reviews, purchased = load_parquets()

    # Derive time components that feature engineering depends on
    history = add_time_components(history)

    # Build {playerid: set(gameids)} lookup — reused by both Steps 2 and 3
    log.info("Building player library lookup …")
    player_library = build_player_library(purchased)

    # ── Step 2: Heuristic Labels ──────────────────────────────────────────────
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

    # ── Step 4: Align + Preprocess ───────────────────────────────────────────
    # Only keep players present in both the feature matrix and heuristic labels
    common_ids  = feature_matrix.index.intersection(heuristic_df.index)
    X_raw       = feature_matrix.loc[common_ids]
    y_heuristic = heuristic_df.loc[common_ids, "heuristic_bot"]
    feature_names = X_raw.columns.tolist()
    log.info("Common players for modelling: %d", len(common_ids))

    X_scaled, _ = preprocess(
        X_raw, save_path=os.path.join(OUTPUTS_DIR, "preprocessor.pkl")
    )

    # ── Step 5: Hyperparameter Tuning ────────────────────────────────────────
    best_if_params, best_lof_params, best_svm_params, tuning_results = tune_models(
        X_scaled, y_heuristic
    )
    tuning_path = os.path.join(OUTPUTS_DIR, "tuning_results.csv")
    tuning_results.to_csv(tuning_path, index=False)
    log.info("Saved → %s", tuning_path)

    # ── Step 6: Train Final Models ───────────────────────────────────────────
    models, scores = train_best_models(
        X_scaled, best_if_params, best_lof_params, best_svm_params
    )

    # ── Step 7: Ensemble ─────────────────────────────────────────────────────
    ensemble_results, percentile_scores = build_ensemble(
        scores, common_ids, y_heuristic
    )
    ensemble_path = os.path.join(OUTPUTS_DIR, "ensemble_results.csv")
    ensemble_results.to_csv(ensemble_path, index=False)
    log.info("Saved → %s", ensemble_path)

    # ── Steps 8 & 9: Evaluate + SHAP ─────────────────────────────────────────
    evaluate(
        ensemble_results  = ensemble_results,
        percentile_scores = percentile_scores,
        y_heuristic       = y_heuristic,
        feature_matrix    = feature_matrix.loc[common_ids],
        feature_names     = feature_names,
        X_scaled          = X_scaled,
        best_if           = models["IsolationForest"],
        outputs_dir       = OUTPUTS_DIR,
    )

    log.info("Pipeline complete. All outputs in %s/", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
