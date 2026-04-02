"""
Phase 2 — Steps 4–7: Preprocessing, Hyperparameter Tuning, Training, Ensemble.
"""

import logging
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

log = logging.getLogger(__name__)

# LOF and OCSVM are O(N²) — cap training size to avoid OOM
_LOF_SVM_SAMPLE = 20_000


def _draw_sample(X: np.ndarray, n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_sample, sample_indices) of size min(n, len(X))."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    return X[idx], idx


# ---------------------------------------------------------------------------
# Step 4 — Preprocessing
# ---------------------------------------------------------------------------

def preprocess(X_raw: pd.DataFrame, save_path: str) -> tuple[np.ndarray, Pipeline]:
    """
    Fit a median-impute → RobustScaler pipeline and persist it.
    RobustScaler is preferred over StandardScaler because anomaly feature
    distributions are heavy-tailed.
    """
    log.info("Step 4 — Preprocessing (MedianImputer + RobustScaler) …")
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])
    X_scaled = preprocessor.fit_transform(X_raw)
    joblib.dump(preprocessor, save_path)
    log.info("  Preprocessor saved → %s", save_path)
    return X_scaled, preprocessor


# ---------------------------------------------------------------------------
# Step 5 — Hyperparameter Tuning
# ---------------------------------------------------------------------------

def _tune_isolation_forest(X_scaled: np.ndarray,
                            y: pd.Series) -> tuple[pd.DataFrame, dict]:
    log.info("  Tuning IsolationForest (%d param combos) …",
             len(list(ParameterGrid({
                 "n_estimators":  [100, 200, 300],
                 "max_samples":   ["auto", 0.8, 0.6],
                 "contamination": [0.02, 0.05, 0.10],
                 "max_features":  [0.8, 1.0],
             }))))
    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_samples":   ["auto", 0.8, 0.6],
        "contamination": [0.02, 0.05, 0.10],
        "max_features":  [0.8, 1.0],
        "random_state":  [42],
    }
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
    # ParameterGrid converts int params to numpy ints — cast back for sklearn
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
        # novelty=True is mandatory to call score_samples on unseen data
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
    Grid-search all three models and return:
      (best_if_params, best_lof_params, best_svm_params, tuning_results_df)

    Note: ROC-AUC here measures how well the model mimics the heuristic rules,
    NOT true anomaly detection accuracy. This is a known, accepted trade-off.
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
# Step 6 — Train Final Models
# ---------------------------------------------------------------------------

def train_best_models(X_scaled: np.ndarray,
                      best_if_params: dict,
                      best_lof_params: dict,
                      best_svm_params: dict) -> tuple[dict, dict]:
    """
    Retrain each model with its best params on the full (scaled) data.
    LOF and OCSVM still train on a fixed 20k sample to stay within memory.
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
# Step 7 — Ensemble
# ---------------------------------------------------------------------------

def build_ensemble(scores: dict,
                   common_ids: pd.Index,
                   y_heuristic: pd.Series) -> tuple[pd.DataFrame, dict]:
    """
    Percentile-rank each model's scores (0–100, higher = more suspicious),
    compute a weighted composite, and apply a 2-of-3 majority vote.

    Weights: IF=0.5, LOF=0.3, OCSVM=0.2
    Flag threshold: top 5% per model; anomaly if ≥ 2 models agree.
    """
    log.info("Step 7 — Building ensemble …")
    if_scores  = scores["IsolationForest"]
    lof_scores = scores["LOF"]
    svm_scores = scores["OCSVM"]

    if_pct  = rankdata(if_scores)  / len(if_scores)  * 100
    lof_pct = rankdata(lof_scores) / len(lof_scores) * 100
    svm_pct = rankdata(svm_scores) / len(svm_scores) * 100

    composite = 0.5 * if_pct + 0.3 * lof_pct + 0.2 * svm_pct

    if_flag  = (if_pct  >= 95).astype(int)
    lof_flag = (lof_pct >= 95).astype(int)
    svm_flag = (svm_pct >= 95).astype(int)
    vote_count = if_flag + lof_flag + svm_flag
    is_anomaly = (vote_count >= 2).astype(int)

    ensemble_results = pd.DataFrame({
        "playerid":        common_ids,
        "composite_score": composite,
        "if_pct":          if_pct,
        "lof_pct":         lof_pct,
        "svm_pct":         svm_pct,
        "vote_count":      vote_count,
        "is_anomaly":      is_anomaly,
        "if_flag":         if_flag,
        "lof_flag":        lof_flag,
        "svm_flag":        svm_flag,
        "heuristic_bot":   y_heuristic.values,
    })

    n = is_anomaly.sum()
    log.info("  Anomalies flagged: %d (%.2f%%)", n, n / len(is_anomaly) * 100)

    percentile_scores = {
        "if_pct": if_pct, "lof_pct": lof_pct,
        "svm_pct": svm_pct, "composite": composite,
    }
    return ensemble_results, percentile_scores
