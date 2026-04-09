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

RAW_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CRAWLED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "crawled")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_list_fast(s: str) -> list:
    """
    Parse a Python-style list string or JSON string into a structured list of dicts.
    Handles both the old format [10, 20, 30] and the new format [{"appid": 10, "playtime_mins": 0}].
    """
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        parsed_data = json.loads(s.replace("'", '"'))
        
        if not parsed_data:
            return []
            
        # Nếu dữ liệu là định dạng cũ (danh sách các số nguyên/chuỗi)
        if isinstance(parsed_data[0], (int, str)):
            return [{"appid": int(appid), "playtime_mins": -1} for appid in parsed_data]
            
        # Nếu dữ liệu là định dạng mới (danh sách các từ điển)
        elif isinstance(parsed_data[0], dict):
            return [{"appid": int(item.get("appid", -1)), "playtime_mins": int(item.get("playtime_mins", -1))} for item in parsed_data if "appid" in item]
            
        return []
    except (ValueError, TypeError):
        return []


def _raw(filename: str) -> str:
    return os.path.join(RAW_DIR, filename)


def _merge_with_crawled(df_raw: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    If data/crawled/<filename> exists, read it and concat with df_raw in RAM.
    The raw CSV files are never modified.
    """
    crawled_path = os.path.join(CRAWLED_DIR, filename)
    if not os.path.exists(crawled_path):
        return df_raw
    df_crawled = pd.read_csv(crawled_path)
    if df_crawled.empty:
        return df_raw
    log.info("  Found crawled/%s — merging %d rows in RAM.", filename, len(df_crawled))
    return pd.concat([df_raw, df_crawled], ignore_index=True)


def _read_purchased_robust(path: str) -> pd.DataFrame:
    """
    Read a purchased_games CSV that may contain malformed rows where the
    library JSON column has unescaped inner quotes.
    """
    import csv as _csv

    _csv.field_size_limit(10_000_000)
    rows = []
    bad_lines: set[int] = set()

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = _csv.reader(f)
        next(reader)  # skip header
        for lineno, row in enumerate(reader, start=2):
            if len(row) == 2:
                rows.append({"playerid": row[0], "library": row[1]})
            else:
                bad_lines.add(lineno)

    if bad_lines:
        log.warning("  %s: %d malformed rows — repairing…", os.path.basename(path), len(bad_lines))
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for lineno, raw_line in enumerate(f, start=2):
                if lineno not in bad_lines:
                    continue
                raw_line = raw_line.rstrip("\n")
                comma_idx = raw_line.index(",")
                playerid  = raw_line[:comma_idx]
                rest      = raw_line[comma_idx + 1:]
                library   = rest[1:-1] if (rest.startswith('"') and rest.endswith('"')) else rest
                try:
                    json.loads(library)
                    rows.append({"playerid": playerid, "library": library})
                except (ValueError, TypeError):
                    log.warning("  Line %d: repair failed, skipping.", lineno)

    return pd.DataFrame(rows, columns=["playerid", "library"])


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
    df = _merge_with_crawled(df, "history.csv")
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
    df = _merge_with_crawled(df, "players.csv")
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
    df = _merge_with_crawled(df, "reviews.csv")
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
    df = _read_purchased_robust(_raw("purchased_games.csv"))
    crawled_path = os.path.join(CRAWLED_DIR, "purchased_games.csv")
    if os.path.exists(crawled_path):
        df_crawled = _read_purchased_robust(crawled_path)
        if not df_crawled.empty:
            log.info("  Found crawled/purchased_games.csv — merging %d rows in RAM.", len(df_crawled))
            df = pd.concat([df, df_crawled], ignore_index=True)
    df["playerid"] = df["playerid"].astype("int64")
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
