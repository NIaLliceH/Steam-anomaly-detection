"""
model_players_enricher.py

Bổ sung purchased_games + playtime data cho tất cả 22583 model players vào
file data/crawled/model_purchased_games.csv.

Phase 1 — Copy:
    Trích dữ liệu đã có sẵn trong targeted_purchased_games.csv cho các player
    chưa xuất hiện trong model_purchased_games.csv.

Phase 2 — Crawl:
    Với các player chưa từng được attempt (không có trong processed_ids.txt),
    gọi Steam GetOwnedGames API và lưu kết quả vào model_purchased_games.csv.
    Checkpoint riêng: data/crawled/model_processed_ids.txt

Usage:
    python3 helpers/model_players_enricher.py
    python3 helpers/model_players_enricher.py --phase1-only
    python3 helpers/model_players_enricher.py --phase2-only
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()
API_KEY = os.getenv("API_KEY")

OUTPUTS_DIR    = "outputs"
CRAWLED_DIR    = "data/crawled"

ENSEMBLE_CSV        = os.path.join(OUTPUTS_DIR, "ensemble_results.csv")
TARGETED_CSV        = os.path.join(CRAWLED_DIR, "targeted_purchased_games.csv")
GLOBAL_CHECKPOINT   = os.path.join(CRAWLED_DIR, "processed_ids.txt")
OUTPUT_CSV          = os.path.join(CRAWLED_DIR, "model_purchased_games.csv")
MODEL_CHECKPOINT    = os.path.join(CRAWLED_DIR, "model_processed_ids.txt")

BATCH_SIZE       = 100
RATE_LIMIT_SLEEP = 1.1
RATE_429_SLEEP   = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

os.makedirs(CRAWLED_DIR, exist_ok=True)

csv.field_size_limit(10_000_000)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_model_ids() -> list[int]:
    df = pd.read_csv(ENSEMBLE_CSV, usecols=["playerid"])
    ids = df["playerid"].astype(int).tolist()
    log.info("Model players: %d", len(ids))
    return ids


def load_existing_model_pg_ids() -> set[int]:
    """Return playerids already in model_purchased_games.csv."""
    if not os.path.exists(OUTPUT_CSV):
        return set()
    ids = set()
    with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                try:
                    ids.add(int(row[0]))
                except ValueError:
                    pass
    return ids


def load_global_processed_ids() -> set[int]:
    """Return all player IDs ever attempted in the global crawler."""
    if not os.path.exists(GLOBAL_CHECKPOINT):
        return set()
    with open(GLOBAL_CHECKPOINT) as f:
        return {int(l.strip()) for l in f if l.strip()}


def load_model_processed_ids() -> set[int]:
    """Return player IDs attempted in phase-2 crawler."""
    if not os.path.exists(MODEL_CHECKPOINT):
        return set()
    with open(MODEL_CHECKPOINT) as f:
        return {int(l.strip()) for l in f if l.strip()}


def mark_model_processed(pid: int) -> None:
    with open(MODEL_CHECKPOINT, "a") as f:
        f.write(f"{pid}\n")


def flush_batch(batch: list[dict]) -> None:
    if not batch:
        return
    df = pd.DataFrame(batch)
    write_header = not os.path.exists(OUTPUT_CSV)
    df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False,
              quoting=csv.QUOTE_NONNUMERIC)
    log.info("  Flushed %d records → %s", len(batch), OUTPUT_CSV)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Copy from targeted_purchased_games.csv
# ──────────────────────────────────────────────────────────────────────────────

def read_targeted_robust() -> dict[int, str]:
    """Load targeted_purchased_games.csv as {playerid: library_json}."""
    result = {}
    bad_lines = set()

    with open(TARGETED_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for lineno, row in enumerate(reader, start=2):
            if len(row) == 2:
                try:
                    result[int(row[0])] = row[1]
                except ValueError:
                    pass
            else:
                bad_lines.add(lineno)

    if bad_lines:
        log.warning("  %d malformed rows in targeted CSV — repairing…", len(bad_lines))
        with open(TARGETED_CSV, "r", encoding="utf-8") as f:
            next(f)
            for lineno, raw in enumerate(f, start=2):
                if lineno not in bad_lines:
                    continue
                raw = raw.rstrip("\n")
                comma_idx = raw.index(",")
                pid  = int(raw[:comma_idx])
                rest = raw[comma_idx + 1:]
                lib  = rest[1:-1] if (rest.startswith('"') and rest.endswith('"')) else rest
                try:
                    json.loads(lib)
                    result[pid] = lib
                except json.JSONDecodeError:
                    log.warning("    Line %d: repair failed (pid=%d), skipping.", lineno, pid)

    return result


def phase1_copy(model_ids: list[int]) -> set[int]:
    """
    Copy rows from targeted_purchased_games.csv → model_purchased_games.csv
    for model players not yet present.

    Returns the set of playerids successfully copied.
    """
    log.info("=== Phase 1: Copy from targeted_purchased_games.csv ===")

    existing = load_existing_model_pg_ids()
    log.info("  Already in model_purchased_games: %d", len(existing))

    log.info("  Loading targeted_purchased_games.csv …")
    targeted = read_targeted_robust()
    log.info("  Targeted data loaded: %d players", len(targeted))

    to_copy = [pid for pid in model_ids if pid not in existing and pid in targeted]
    log.info("  To copy: %d players", len(to_copy))

    if not to_copy:
        log.info("  Nothing to copy.")
        return set()

    batch: list[dict] = []
    copied = set()
    for pid in to_copy:
        batch.append({"playerid": pid, "library": targeted[pid]})
        copied.add(pid)
        if len(batch) >= BATCH_SIZE:
            flush_batch(batch)
            batch.clear()
    flush_batch(batch)

    log.info("Phase 1 done. Copied %d players.", len(copied))
    return copied


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Crawl remaining players
# ──────────────────────────────────────────────────────────────────────────────

def fetch_owned_games(steam_id: int) -> list[dict] | None:
    url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": API_KEY,
        "steamid": steam_id,
        "include_appinfo": 0,
        "include_played_free_games": 1,
    }
    for _ in range(2):
        try:
            resp = requests.get(url, params=params, timeout=15)
        except requests.RequestException as e:
            log.warning("  Network error for %d: %s", steam_id, e)
            return None

        if resp.status_code == 429:
            log.warning("  HTTP 429 — sleeping %ds …", RATE_429_SLEEP)
            time.sleep(RATE_429_SLEEP)
            continue
        if resp.status_code in (403, 401):
            return None
        if resp.status_code != 200:
            log.warning("  HTTP %d for %d — skipping.", resp.status_code, steam_id)
            return None

        try:
            data = resp.json()
        except ValueError:
            log.warning("  Invalid JSON for %d — skipping.", steam_id)
            return None

        games = data.get("response", {}).get("games")
        if not games:
            return None
        return [{"appid": g["appid"], "playtime_mins": g.get("playtime_forever", 0)} for g in games]
    return None


def phase2_crawl(model_ids: list[int]) -> None:
    """Crawl players not yet in processed_ids.txt or model_processed_ids.txt."""
    log.info("=== Phase 2: Crawl missing players ===")

    if not API_KEY:
        log.error("API_KEY not set — skipping Phase 2.")
        return

    global_processed = load_global_processed_ids()
    model_processed  = load_model_processed_ids()
    all_attempted    = global_processed | model_processed

    existing = load_existing_model_pg_ids()

    pending = [pid for pid in model_ids if pid not in all_attempted and pid not in existing]
    log.info("  Players to crawl: %d", len(pending))

    if not pending:
        log.info("  Nothing left to crawl.")
        return

    batch: list[dict] = []
    success = skip = 0

    for i, pid in enumerate(pending, start=1):
        games = fetch_owned_games(pid)

        if games is not None:
            batch.append({"playerid": pid, "library": json.dumps(games, separators=(",", ":"))})
            success += 1
        else:
            skip += 1

        mark_model_processed(pid)

        if i % 500 == 0:
            log.info("  Progress: %d / %d  (saved=%d, private/empty=%d)",
                     i, len(pending), success, skip)

        if len(batch) >= BATCH_SIZE:
            flush_batch(batch)
            batch.clear()

        time.sleep(RATE_LIMIT_SLEEP)

    flush_batch(batch)
    log.info("Phase 2 done. Attempted: %d | Saved: %d | Skipped: %d",
             len(pending), success, skip)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich model_purchased_games.csv")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--phase1-only", action="store_true", help="Copy only, no crawling")
    group.add_argument("--phase2-only", action="store_true", help="Crawl only, skip copy phase")
    args = parser.parse_args()

    log.info("=== Model Players Enricher ===")
    model_ids = load_model_ids()

    if not args.phase2_only:
        phase1_copy(model_ids)

    if not args.phase1_only:
        phase2_crawl(model_ids)

    # Final stats
    final_ids = load_existing_model_pg_ids()
    log.info("=== Final model_purchased_games.csv: %d players ===", len(final_ids))
    missing = set(model_ids) - final_ids
    log.info("=== Still missing (private/not crawled): %d players ===", len(missing))


if __name__ == "__main__":
    main()
