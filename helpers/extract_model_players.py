"""
extract_model_players.py

Trích xuất dữ liệu purchased_games của 3157 player được đưa vào model
từ file targeted_purchased_games.csv ra file riêng.

Input:
  - outputs/ensemble_results.csv  → danh sách player IDs
  - data/crawled/targeted_purchased_games.csv  → nguồn data crawl

Output:
  - data/crawled/model_players_purchased.csv   (dùng với merge_data.py)

Chạy:
  python3 extract_model_players.py
"""

import csv
import json
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ENSEMBLE_PATH = "outputs/ensemble_results.csv"
SOURCE_CSV    = "data/crawled/targeted_purchased_games.csv"
OUTPUT_CSV    = "data/crawled/model_players_purchased.csv"


def read_purchased_robust(path: str) -> pd.DataFrame:
    """Read purchased CSV, repairing rows with unescaped inner quotes."""
    csv.field_size_limit(10_000_000)
    rows = []
    bad_lines = set()

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for lineno, row in enumerate(reader, start=2):
            if len(row) == 2:
                rows.append({"playerid": int(row[0]), "library": row[1]})
            else:
                bad_lines.add(lineno)

    if bad_lines:
        log.warning("%d malformed rows — repairing…", len(bad_lines))
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for lineno, raw_line in enumerate(f, start=2):
                if lineno not in bad_lines:
                    continue
                raw_line = raw_line.rstrip("\n")
                comma_idx = raw_line.index(",")
                playerid  = int(raw_line[:comma_idx])
                rest      = raw_line[comma_idx + 1:]
                library   = rest[1:-1] if (rest.startswith('"') and rest.endswith('"')) else rest
                try:
                    json.loads(library)
                    rows.append({"playerid": playerid, "library": library})
                except json.JSONDecodeError:
                    log.warning("  Line %d: repair failed, skipping (playerid=%d)", lineno, playerid)

    return pd.DataFrame(rows)


# ── Load model player IDs ─────────────────────────────────────────────────────
log.info("Loading model player IDs from %s …", ENSEMBLE_PATH)
model_ids = set(pd.read_csv(ENSEMBLE_PATH, usecols=["playerid"])["playerid"].astype(int))
log.info("  %d players in model.", len(model_ids))

# ── Load & filter crawled data ────────────────────────────────────────────────
log.info("Reading %s …", SOURCE_CSV)
crawled_df = read_purchased_robust(SOURCE_CSV)
log.info("  %d total rows loaded.", len(crawled_df))

filtered_df = crawled_df[crawled_df["playerid"].isin(model_ids)].reset_index(drop=True)
log.info("  %d rows matched model players.", len(filtered_df))

missing = model_ids - set(filtered_df["playerid"])
if missing:
    log.warning("  %d model players not found in crawled data (private/not crawled).", len(missing))

# ── Save ──────────────────────────────────────────────────────────────────────
filtered_df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
log.info("Saved → %s  (%d rows)", OUTPUT_CSV, len(filtered_df))
