"""
Phase 1: Data ETL & Memory Optimization for Steam Anomaly Detection
Reads raw CSVs, applies memory optimization, cleans data, and exports as .parquet.
"""

import os
import json
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_list_fast(s: str) -> list:
    """Parse a Python-style list string using json.loads for speed."""
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        return json.loads(s.replace("'", '"'))
    except (ValueError, TypeError):
        return []


def _raw(filename: str) -> str:
    return os.path.join(RAW_DIR, filename)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_private_ids() -> set:
    log.info("Loading private Steam IDs …")
    df = pd.read_csv(
        _raw("private_steamids.csv"),
        usecols=["playerid"],
        dtype={"playerid": "int64"},
    )
    private = set(df["playerid"].tolist())
    log.info("  %d private player IDs loaded.", len(private))
    return private


def load_history(private_ids: set) -> pd.DataFrame:
    log.info("Loading history.csv (~647 MB) …")
    df = pd.read_csv(
        _raw("history.csv"),
        usecols=["playerid", "achievementid", "date_acquired"],
        dtype={"playerid": "int64", "achievementid": "string"},
    )
    log.info("  Raw rows: %d", len(df))

    # Extract gameid safely — achievement names may contain underscores
    df["gameid"] = (
        df["achievementid"]
        .str.extract(r"^(\d+)_")[0]
        .astype("Int32")  # nullable int to handle non-matching rows
    )

    # Fast datetime parse with explicit format
    df["date_acquired"] = pd.to_datetime(
        df["date_acquired"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    # Filter private players
    before = len(df)
    df = df[~df["playerid"].isin(private_ids)].reset_index(drop=True)
    log.info("  Removed %d rows belonging to private players.", before - len(df))

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["playerid", "achievementid", "date_acquired"], keep="last").reset_index(drop=True)
    log.info("  Removed %d duplicate rows.", before - len(df))
    log.info("  Final rows: %d", len(df))
    return df


def load_players(private_ids: set) -> pd.DataFrame:
    log.info("Loading players.csv …")
    df = pd.read_csv(
        _raw("players.csv"),
        usecols=["playerid", "country", "created"],
        dtype={"playerid": "int64", "country": "category"},
    )
    log.info("  Raw rows: %d", len(df))

    df["created"] = pd.to_datetime(
        df["created"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    before = len(df)
    df = df[~df["playerid"].isin(private_ids)].reset_index(drop=True)
    log.info("  Removed %d private players.", before - len(df))

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["playerid"], keep="last").reset_index(drop=True)
    log.info("  Removed %d duplicate rows.", before - len(df))
    log.info("  Final rows: %d", len(df))
    return df


def load_reviews(private_ids: set) -> pd.DataFrame:
    log.info("Loading reviews.csv (~551 MB) …")
    df = pd.read_csv(
        _raw("reviews.csv"),
        usecols=["reviewid", "playerid", "gameid", "review", "helpful", "funny", "awards", "posted"],
        dtype={
            "reviewid": "int32",
            "playerid": "int64",
            "gameid": "int32",
            "helpful": "int32",
            "funny": "int32",
            "awards": "int32",
            "review": "string",
        },
    )
    log.info("  Raw rows: %d", len(df))

    df["posted"] = pd.to_datetime(df["posted"], format="%Y-%m-%d", errors="coerce")

    before = len(df)
    df = df[~df["playerid"].isin(private_ids)].reset_index(drop=True)
    log.info("  Removed %d rows belonging to private players.", before - len(df))

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["reviewid"], keep="last").reset_index(drop=True)
    log.info("  Removed %d duplicate rows.", before - len(df))
    log.info("  Final rows: %d", len(df))
    return df


def load_purchased(private_ids: set) -> pd.DataFrame:
    log.info("Loading purchased_games.csv …")
    df = pd.read_csv(
        _raw("purchased_games.csv"),
        usecols=["playerid", "library"],
        dtype={"playerid": "int64", "library": "string"},
    )
    log.info("  Raw rows: %d", len(df))

    # Fast list parsing instead of ast.literal_eval
    df["library"] = df["library"].apply(_parse_list_fast)
    df["library_size"] = df["library"].apply(len).astype("int32")

    before = len(df)
    df = df[~df["playerid"].isin(private_ids)].reset_index(drop=True)
    log.info("  Removed %d private players.", before - len(df))

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["playerid"], keep="last").reset_index(drop=True)
    log.info("  Removed %d duplicate rows.", before - len(df))
    log.info("  Final rows: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    log.info("Output directory: %s", os.path.abspath(PROCESSED_DIR))

    private_ids = load_private_ids()

    datasets = {
        "history":   load_history(private_ids),
        "players":   load_players(private_ids),
        "reviews":   load_reviews(private_ids),
        "purchased": load_purchased(private_ids),
    }

    for name, df in datasets.items():
        out_path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
        df.to_parquet(out_path, index=False, compression="snappy")
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        log.info("Saved %s.parquet  (%.1f MB,  %d rows)", name, size_mb, len(df))

    log.info("Phase 1 complete.")


if __name__ == "__main__":
    main()
