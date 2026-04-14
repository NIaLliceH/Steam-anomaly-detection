"""
Active Learning & Human-in-the-Loop (HITL) workflow.

Provides two functions to close the feedback loop between model predictions
and ground-truth labels:

  generate_review_sample  — exports high-conflict cases for human review
  integrate_human_labels  — merges human decisions back into heuristic_df

Workflow:
  1. Run the full pipeline (main.py).
  2. Inspect outputs/to_review.csv and fill in the `human_label` column
     (1 = confirmed bot, 0 = confirmed normal, leave blank to skip).
  3. Save the file as outputs/reviewed.csv.
  4. Re-run main.py — human labels override heuristic labels automatically.
"""

import logging
import os

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Function A — Smart Sampling: export high-conflict cases for review
# ---------------------------------------------------------------------------

def generate_review_sample(
    ensemble_results: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    outputs_dir: str,
    top_k: int = 50,
) -> pd.DataFrame:
    """
    Find the highest-conflict players and export them for human review.

    "High-conflict" means the model strongly suspects a bot (composite_score >= 85)
    while the heuristic labelled the player as normal (heuristic_bot == 0).
    These are exactly the stealth bots the static rules miss.

    The export includes all feature columns plus an empty `human_label` column.
    A human reviewer fills in:
        1  → confirmed bot
        0  → confirmed normal
      (blank) → unsure / skip

    Parameters
    ----------
    ensemble_results : DataFrame returned by build_ensemble()
    feature_matrix   : raw feature DataFrame, indexed by playerid
    outputs_dir      : path to the outputs directory
    top_k            : maximum number of rows to export (sorted by composite_score desc)

    Returns
    -------
    conflict_df : the exported DataFrame (empty if no conflicts found)
    """
    # High-conflict mask: model says bot, heuristic says normal
    conflict_mask = (
        (ensemble_results["heuristic_bot"] == 0)
        & (ensemble_results["composite_score"] >= 85)
    )
    conflict = ensemble_results.loc[conflict_mask].copy()

    if conflict.empty:
        log.info("Active Learning: no high-conflict cases found — skipping export.")
        return conflict

    # Sort descending by composite_score, keep top_k most suspicious
    conflict = conflict.sort_values("composite_score", ascending=False).head(top_k)

    log.info(
        "Active Learning: %d high-conflict cases found (heuristic=normal, model=bot). "
        "Exporting top %d …",
        conflict_mask.sum(), len(conflict),
    )

    # Merge with feature columns so the reviewer has context
    feature_cols = feature_matrix.reindex(conflict["playerid"].values)
    feature_cols = feature_cols.reset_index(drop=False)  # playerid becomes a column

    review_df = conflict.merge(feature_cols, on="playerid", how="left")

    # Add empty human_label column at the front for easy editing
    review_df.insert(0, "human_label", "")

    out_path = os.path.join(outputs_dir, "to_review.csv")
    review_df.to_csv(out_path, index=False)
    log.info("  Saved → %s", out_path)
    log.info(
        "  ACTION REQUIRED: open to_review.csv, fill `human_label` (1=bot / 0=normal), "
        "then save as reviewed.csv in the same directory and re-run the pipeline."
    )

    return review_df


# ---------------------------------------------------------------------------
# Function B — Integrate human decisions back into heuristic_df
# ---------------------------------------------------------------------------

def integrate_human_labels(
    heuristic_df: pd.DataFrame,
    reviewed_csv_path: str,
) -> pd.DataFrame:
    """
    Override heuristic labels with human decisions from reviewed.csv.

    For every playerid in reviewed.csv that has a non-null `human_label`:
      - heuristic_bot    ← human_label  (1 = bot, 0 = normal)
      - heuristic_normal ← 1 - human_label  (complement)

    Players without a human_label (blank / NaN) are left unchanged.

    Parameters
    ----------
    heuristic_df      : original heuristic label DataFrame (indexed by playerid)
    reviewed_csv_path : path to the manually reviewed CSV file

    Returns
    -------
    Updated heuristic_df with human overrides applied.
    """
    if not os.path.exists(reviewed_csv_path):
        log.info("Active Learning: no reviewed.csv found — using heuristic labels as-is.")
        return heuristic_df

    reviewed = pd.read_csv(reviewed_csv_path)

    # Validate expected columns
    if "playerid" not in reviewed.columns or "human_label" not in reviewed.columns:
        log.warning(
            "Active Learning: reviewed.csv is missing 'playerid' or 'human_label' "
            "columns — skipping integration."
        )
        return heuristic_df

    # Keep only rows where human actually filled in a label
    labelled = reviewed.dropna(subset=["human_label"]).copy()
    labelled = labelled[labelled["human_label"].astype(str).str.strip() != ""]

    if labelled.empty:
        log.info("Active Learning: reviewed.csv has no filled labels — no overrides applied.")
        return heuristic_df

    # Cast to int (handles both 0/1 stored as float or string)
    labelled["human_label"] = labelled["human_label"].astype(float).astype(int)

    # Only accept valid labels (0 or 1)
    invalid = labelled[~labelled["human_label"].isin([0, 1])]
    if not invalid.empty:
        log.warning(
            "Active Learning: %d rows in reviewed.csv have invalid human_label values "
            "(not 0 or 1) — these will be skipped: %s",
            len(invalid), invalid["playerid"].tolist(),
        )
        labelled = labelled[labelled["human_label"].isin([0, 1])]

    labelled = labelled.set_index("playerid")[["human_label"]]

    # Find which playerids are actually present in heuristic_df
    common = heuristic_df.index.intersection(labelled.index)
    missing = labelled.index.difference(heuristic_df.index)
    if not missing.empty:
        log.warning(
            "Active Learning: %d playerid(s) in reviewed.csv not found in heuristic_df "
            "— these will be ignored: %s",
            len(missing), missing.tolist()[:10],
        )

    if common.empty:
        log.warning("Active Learning: no matching playerids — overrides not applied.")
        return heuristic_df

    heuristic_df = heuristic_df.copy()
    heuristic_df.loc[common, "heuristic_bot"]    = labelled.loc[common, "human_label"].values
    heuristic_df.loc[common, "heuristic_normal"] = (1 - labelled.loc[common, "human_label"]).values

    log.info(
        "Active Learning: applied %d human label override(s) "
        "(%d bot, %d normal) from %s",
        len(common),
        int(labelled.loc[common, "human_label"].sum()),
        int((1 - labelled.loc[common, "human_label"]).sum()),
        reviewed_csv_path,
    )

    return heuristic_df
