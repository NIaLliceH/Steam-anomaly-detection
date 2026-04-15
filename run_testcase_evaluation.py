"""Standalone evaluator for a unified testcase CSV.

This file requires the trained artifacts already saved in outputs/:
- model_memory.pkl
- preprocessor.pkl
- best_xgb.pkl
- best_if.pkl

Expected testcase columns:
- playerid
- human_label (0/1)
- the same 27 feature columns stored in model_memory.pkl

Behavior:
- Uses the testcase player set exactly as provided; no re-sampling.
- Scores testcase rows directly from the 27 feature columns embedded in the CSV.
- Exports predictions, confusion matrix, and summary files into outputs/.

Default input: data/test/testcase_40_unified.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))


def parse_playerid_series(series: pd.Series) -> pd.Series:
    """Parse SteamID values safely from string-like input to int64 without float precision loss."""
    s = series.astype("string").str.strip()
    s = s.where(s.notna() & (s != ""))
    s = s.where(s.str.fullmatch(r"\d+"), other=pd.NA)
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def load_model_bundle() -> dict:
    """Load the saved model artifacts required for standalone inference."""
    required = {
        "memory": OUTPUTS_DIR / "model_memory.pkl",
        "preprocessor": OUTPUTS_DIR / "preprocessor.pkl",
        "xgb": OUTPUTS_DIR / "best_xgb.pkl",
        "iforest": OUTPUTS_DIR / "best_if.pkl",
    }

    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing) + ". Run main.py first."
        )

    return {
        "memory": joblib.load(required["memory"]),
        "preprocessor": joblib.load(required["preprocessor"]),
        "xgb": joblib.load(required["xgb"]),
        "iforest": joblib.load(required["iforest"]),
    }


def bot_evidence_text(row: pd.Series) -> str:
    """Return a concise explanation string for why a profile looks bot-like."""
    reasons: list[str] = []

    median_i = row.get("median_unlock_interval_sec")
    min_i = row.get("min_unlock_interval_sec")
    top1 = row.get("top1_game_concentration")
    max_day = row.get("max_achievements_per_day")
    night_r = row.get("night_activity_ratio")
    total_ach = row.get("total_achievements")
    review_unowned = row.get("review_unowned_ratio")
    total_reviews = row.get("total_reviews")

    if pd.notna(median_i) and pd.notna(min_i) and pd.notna(top1):
        if (median_i < 10) and (min_i < 1) and (top1 > 0.85):
            reasons.append("Speed pattern: unlock interval extremely short + high single-game concentration")

    if pd.notna(max_day) and pd.notna(night_r) and pd.notna(total_ach):
        if (max_day > 500) and (night_r > 0.40) and (total_ach > 1000):
            reasons.append("Volume pattern: very high daily unlock volume + night-heavy activity")

    if pd.notna(total_reviews) and pd.notna(total_ach) and pd.notna(review_unowned):
        if (total_reviews > 5) and (total_ach == 0) and (review_unowned > 0.70):
            reasons.append("Review pattern: many unowned-game reviews with no achievement history")

    xgb_pct = row.get("xgb_pct")
    if_pct = row.get("if_pct")
    comp = row.get("composite_score")
    if pd.notna(xgb_pct) and pd.notna(if_pct) and pd.notna(comp):
        reasons.append(f"Model signal: xgb_pct={xgb_pct:.1f}, if_pct={if_pct:.1f}, composite={comp:.1f}")

    if not reasons:
        return "Weak explicit rule evidence; flagged mainly by model score distribution."
    return " | ".join(reasons)


def percentile_from_sorted(sorted_scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Convert raw scores into percentile ranks against baseline scores."""
    if sorted_scores.size == 0:
        return np.full(shape=len(values), fill_value=np.nan, dtype=np.float32)
    idx = np.searchsorted(sorted_scores, values, side="right")
    return (idx / sorted_scores.size * 100).astype(np.float32)


def infer_and_score(features: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Run preprocessing + models and return per-player predictions."""
    from models import apply_log_transform

    feature_columns = bundle["memory"].get("feature_columns")
    if not feature_columns:
        raise ValueError("model_memory.pkl does not contain feature_columns")

    X = features.reindex(columns=feature_columns)
    X_log = apply_log_transform(X)
    X_scaled = bundle["preprocessor"].transform(X_log)

    xgb_model = bundle["xgb"]
    if_model = bundle["iforest"]
    memory = bundle["memory"]

    xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]
    if_score = -if_model.score_samples(X_scaled)

    xgb_sorted = np.asarray(memory.get("sorted_raw_scores", {}).get("XGBoost", []), dtype=np.float32)
    if_sorted = np.asarray(memory.get("sorted_raw_scores", {}).get("IsolationForest", []), dtype=np.float32)

    xgb_pct = percentile_from_sorted(xgb_sorted, xgb_proba)
    if_pct = percentile_from_sorted(if_sorted, if_score)
    if memory.get("if_flipped", False):
        if_pct = 100.0 - if_pct

    composite = 0.80 * xgb_pct + 0.20 * if_pct
    pred = (composite >= 85).astype(int)

    return pd.DataFrame(
        {
            "playerid": features.index.astype("int64"),
            "xgb_proba": xgb_proba,
            "xgb_pct": xgb_pct,
            "if_pct": if_pct,
            "composite_score": composite,
            "pred_label": pred,
        }
    )


def confusion_matrix_df(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Build a 2x2 confusion matrix with anomaly as positive class."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=pd.Index(["Actual Positive", "Actual Negative"], name="actual"),
        columns=pd.Index(["Predicted Positive", "Predicted Negative"], name="predicted"),
    )


def load_unified_testcase(path: Path) -> pd.DataFrame:
    """Load testcase with exact player set and labels."""
    if not path.exists():
        raise FileNotFoundError(f"Missing testcase file: {path}")

    df = pd.read_csv(path, dtype={"playerid": "string"})
    required = {"playerid", "human_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Testcase file missing required columns: {sorted(missing)}")

    ids = parse_playerid_series(df["playerid"])
    labels = pd.to_numeric(df["human_label"], errors="coerce")

    valid = ids.notna() & labels.isin([0, 1])
    df = df[valid].copy()
    df["playerid"] = ids[valid].astype("int64")
    df["human_label"] = labels[valid].astype("int64")

    if df.empty:
        raise ValueError("No valid rows after parsing playerid/human_label")

    df = df.drop_duplicates(subset=["playerid"], keep="first")

    if set(df["human_label"].unique()) != {0, 1}:
        raise ValueError("Testcase must contain both classes human_label=0 and human_label=1")

    return df.reset_index(drop=True)


def build_feature_frame_from_testcase(testcase_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Build the inference frame directly from the testcase's embedded feature columns."""
    if not feature_columns:
        raise ValueError("model_memory.pkl does not contain feature_columns")

    missing_columns = [col for col in feature_columns if col not in testcase_df.columns]
    if missing_columns:
        raise ValueError(
            "Testcase is missing required metric columns: " + ", ".join(missing_columns)
        )

    features = testcase_df[["playerid", *feature_columns]].copy()
    for col in feature_columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features[feature_columns] = features[feature_columns].replace([np.inf, -np.inf], np.nan)

    return features.set_index("playerid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone testcase evaluation from one unified CSV")
    parser.add_argument(
        "--testcase-input",
        type=Path,
        default=ROOT / "data" / "test" / "testcase_40_unified.csv",
        help="Unified testcase CSV containing playerid, human_label, and the 27 model features",
    )
    parser.add_argument("--threshold", type=float, default=85.0, help="Composite-score threshold used to convert score into pred_label")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="testcase_eval",
        help="Prefix for generated output files inside outputs/",
    )
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    pred_out = OUTPUTS_DIR / f"{args.output_prefix}_predictions.csv"
    cm_out = OUTPUTS_DIR / f"{args.output_prefix}_confusion_matrix.csv"
    summary_out = OUTPUTS_DIR / f"{args.output_prefix}_summary.json"

    testcase = load_unified_testcase(args.testcase_input)
    bundle = load_model_bundle()
    feature_columns = bundle["memory"].get("feature_columns")
    features = build_feature_frame_from_testcase(testcase, feature_columns)
    scored = infer_and_score(features, bundle)

    result = testcase[["playerid", "human_label"]].copy()
    result = result.merge(scored, on="playerid", how="left")
    feature_out = features.reset_index()
    result = result.merge(feature_out, on="playerid", how="left")

    result["pred_label"] = (result["composite_score"] >= float(args.threshold)).astype(int)
    result["result"] = np.where(
        (result["human_label"] == 1) & (result["pred_label"] == 1),
        "TP",
        np.where(
            (result["human_label"] == 0) & (result["pred_label"] == 0),
            "TN",
            np.where(
                (result["human_label"] == 0) & (result["pred_label"] == 1),
                "FP",
                "FN",
            ),
        ),
    )

    result["bot_evidence"] = result.apply(bot_evidence_text, axis=1)

    cm = confusion_matrix_df(result["human_label"], result["pred_label"])
    tp = int(cm.loc["Actual Positive", "Predicted Positive"])
    fp = int(cm.loc["Actual Negative", "Predicted Positive"])
    fn = int(cm.loc["Actual Positive", "Predicted Negative"])
    tn = int(cm.loc["Actual Negative", "Predicted Negative"])

    accuracy = (tp + tn) / max(len(result), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    result_out = result.sort_values(["human_label", "composite_score"], ascending=[False, False])

    result_out.to_csv(pred_out, index=False)
    cm.to_csv(cm_out)

    summary = {
        "n_total": int(len(result_out)),
        "n_normal": int((result_out["human_label"] == 0).sum()),
        "n_anomaly": int((result_out["human_label"] == 1).sum()),
        "threshold": float(args.threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "testcase_input": str(args.testcase_input),
        "predictions_output": str(pred_out),
        "confusion_matrix_output": str(cm_out),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Unified testcase evaluation ===")
    print(f"Input: {args.testcase_input}")
    print(f"Rows: {len(result_out)} | Normal: {(result_out['human_label'] == 0).sum()} | Anomaly: {(result_out['human_label'] == 1).sum()}")
    print(f"Threshold: {float(args.threshold):.2f}")
    print("\nConfusion matrix (anomaly = positive class):")
    print(cm.to_string())
    print("\nMetrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"\nSaved predictions -> {pred_out}")
    print(f"Saved confusion matrix -> {cm_out}")
    print(f"Saved summary -> {summary_out}")


if __name__ == "__main__":
    main()
