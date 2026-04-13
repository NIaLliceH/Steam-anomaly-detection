"""Evaluator for a unified testcase CSV.

Expected input columns:
- playerid
- human_label (0/1)
Optional columns:
- 27 feature columns

Behavior:
- Uses playerid from testcase as the exact evaluation set (no re-sampling).
- Runs model inference and computes confusion matrix + metrics.
- Exports predictions, behavior lines, confusion matrix, and summary.

Use the standardized testcase_40_unified.csv in data/test/ with columns: playerid, human_label, and 27 metric columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from independent_40_test import (
    OUTPUTS_DIR,
    bot_evidence_text,
    confusion_matrix_df,
    infer_and_score,
    load_model_bundle,
    parse_playerid_series,
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
    """Get feature vectors directly from the testcase's 27 metric columns."""
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
    parser = argparse.ArgumentParser(description="Run testcase evaluation from one unified CSV")
    parser.add_argument(
        "--testcase-input",
        type=Path,
        default=ROOT / "data" / "test" / "testcase_40_unified.csv",
        help="Unified testcase CSV with playerid + human_label",
    )
    parser.add_argument("--threshold", type=float, default=85.0, help="Composite-score threshold")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="testcase_eval",
        help="Prefix for output files in outputs/",
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
