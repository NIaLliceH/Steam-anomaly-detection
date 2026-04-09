"""
Targeted Data Enrichment Crawler
Crawls library + playtime for a high-priority subset of ~45,000 players.

Priority order:
  1. PU Training Set   — heuristic_bot==1 OR heuristic_normal==1
  2. Grey Area         — is_anomaly==1 OR composite_score>=80
  3. Random Baseline   — fill remaining quota from normal players

Output: data/crawled/targeted_purchased_games.csv
        data/crawled/processed_ids.txt  (checkpoint: all attempted IDs)
"""

import csv
import json
import logging
import os
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
OUTPUT_CSV     = os.path.join(CRAWLED_DIR, "targeted_purchased_games.csv")
CHECKPOINT_TXT = os.path.join(CRAWLED_DIR, "processed_ids.txt")

TARGET_QUOTA    = 10_000
BATCH_SIZE      = 100       # flush to disk every N successful records
RATE_LIMIT_SLEEP = 1.1      # seconds between requests
RATE_429_SLEEP   = 60       # seconds to wait on HTTP 429

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

os.makedirs(CRAWLED_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Targeted Sampling
# ──────────────────────────────────────────────────────────────────────────────

def select_target_players(quota: int = TARGET_QUOTA) -> list[int]:
    """
    Read heuristic_labels.csv and ensemble_results.csv and return up to
    `quota` playerids in strict priority order.
    """
    heuristic_path = os.path.join(OUTPUTS_DIR, "heuristic_labels.csv")
    ensemble_path  = os.path.join(OUTPUTS_DIR, "ensemble_results.csv")

    if not os.path.exists(heuristic_path):
        raise FileNotFoundError(f"Missing: {heuristic_path} — run main.py first.")
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Missing: {ensemble_path} — run main.py first.")

    heuristic = pd.read_csv(heuristic_path, usecols=["playerid", "heuristic_bot", "heuristic_normal"])
    ensemble  = pd.read_csv(ensemble_path,  usecols=["playerid", "is_anomaly", "composite_score"])

    heuristic["playerid"] = heuristic["playerid"].astype(int)
    ensemble["playerid"]  = ensemble["playerid"].astype(int)

    df = heuristic.merge(ensemble, on="playerid", how="outer")

    selected: list[int] = []
    remaining_mask = pd.Series(True, index=df.index)

    # ── Priority 1: PU Training Set ───────────────────────────────────────────
    p1_mask = (df["heuristic_bot"].fillna(0) == 1) | (df["heuristic_normal"].fillna(0) == 1)
    p1_ids  = df.loc[p1_mask, "playerid"].tolist()
    selected.extend(p1_ids)
    remaining_mask &= ~p1_mask
    log.info("Priority 1 (PU training set):   %d players", len(p1_ids))

    if len(selected) >= quota:
        log.info("Quota reached after Priority 1.")
        return [int(x) for x in selected[:quota]]

    # ── Priority 2: Grey Area / Suspected Bots ────────────────────────────────
    p2_mask = remaining_mask & (
        (df["is_anomaly"].fillna(0) == 1) |
        (df["composite_score"].fillna(0) >= 80)
    )
    p2_ids = df.loc[p2_mask, "playerid"].tolist()
    selected.extend(p2_ids)
    remaining_mask &= ~p2_mask
    log.info("Priority 2 (grey area bots):    %d players", len(p2_ids))

    if len(selected) >= quota:
        log.info("Quota reached after Priority 2.")
        return [int(x) for x in selected[:quota]]

    # ── Priority 3: Random Baseline ───────────────────────────────────────────
    slots_left = quota - len(selected)
    p3_pool    = df.loc[remaining_mask, "playerid"]
    p3_ids     = p3_pool.sample(n=min(slots_left, len(p3_pool)), random_state=42).tolist()
    selected.extend(p3_ids)
    log.info("Priority 3 (random baseline):   %d players", len(p3_ids))

    log.info("Total selected: %d / %d quota", len(selected), quota)
    return [int(x) for x in selected[:quota]]


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_processed_ids() -> set[int]:
    """Return the set of all already-attempted player IDs."""
    if not os.path.exists(CHECKPOINT_TXT):
        return set()
    with open(CHECKPOINT_TXT, "r") as f:
        ids = set()
        for line in f:
            line = line.strip()
            if line:
                ids.add(int(line))
    log.info("Checkpoint: %d IDs already processed — skipping.", len(ids))
    return ids


def mark_processed(pid: int) -> None:
    """Append a single player ID to the checkpoint file."""
    with open(CHECKPOINT_TXT, "a") as f:
        f.write(f"{pid}\n")


def flush_batch(batch: list[dict]) -> None:
    """Append a batch of rows to the output CSV."""
    if not batch:
        return
    df = pd.DataFrame(batch)
    write_header = not os.path.exists(OUTPUT_CSV)
    df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False,
              quoting=csv.QUOTE_NONNUMERIC)
    log.info("  Flushed %d records → %s", len(batch), OUTPUT_CSV)


# ──────────────────────────────────────────────────────────────────────────────
# Steam API — GetOwnedGames
# ──────────────────────────────────────────────────────────────────────────────

def fetch_owned_games(steam_id: int) -> list[dict] | None:
    """
    Call GetOwnedGames for one player.

    Returns a list of {"appid": int, "playtime_mins": int} dicts on success,
    None on private/empty profile, raises RuntimeError on unrecoverable error.

    Handles HTTP 429 by sleeping RATE_429_SLEEP seconds and retrying once.
    """
    url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": API_KEY,
        "steamid": steam_id,
        "include_appinfo": 0,
        "include_played_free_games": 1,
    }

    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, timeout=15)
        except requests.RequestException as e:
            log.warning("  Network error for %d: %s", steam_id, e)
            return None

        if resp.status_code == 429:
            log.warning("  HTTP 429 — sleeping %ds before retry …", RATE_429_SLEEP)
            time.sleep(RATE_429_SLEEP)
            continue

        if resp.status_code == 403:
            # Private profile
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
            # Public profile but no games (or empty library)
            return None

        return [
            {"appid": g["appid"], "playtime_mins": g.get("playtime_forever", 0)}
            for g in games
        ]

    return None   # Both attempts failed (429 twice)


# ──────────────────────────────────────────────────────────────────────────────
# Main crawler loop
# ──────────────────────────────────────────────────────────────────────────────

def crawl(target_ids: list[int]) -> None:
    processed = load_processed_ids()
    pending   = [pid for pid in target_ids if pid not in processed]
    log.info("Pending: %d players to crawl.", len(pending))

    batch: list[dict] = []
    success_total = 0
    skip_total    = 0

    for i, pid in enumerate(pending, start=1):
        games = fetch_owned_games(pid)

        if games is not None:
            library_json = json.dumps(games, separators=(",", ":"))
            batch.append({"playerid": pid, "library": library_json})
            success_total += 1
        else:
            skip_total += 1

        mark_processed(pid)

        if i % 100 == 0:
            log.info("Progress: %d / %d  (ok=%d, skipped=%d)",
                     i, len(pending), success_total, skip_total)

        if len(batch) >= BATCH_SIZE:
            flush_batch(batch)
            batch.clear()

        time.sleep(RATE_LIMIT_SLEEP)

    # Final flush
    flush_batch(batch)
    log.info("Done. Total attempted: %d | Saved: %d | Skipped/private: %d",
             len(pending), success_total, skip_total)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY:
        raise EnvironmentError("API_KEY not set — add it to your .env file.")

    log.info("=== Targeted Data Enrichment Crawler ===")
    target_ids = select_target_players(TARGET_QUOTA)
    crawl(target_ids)
