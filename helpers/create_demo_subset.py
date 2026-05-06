"""
Demo Dataset Selector:
- Chooses ~3.5k players from 424k raw dataset
- Maintains balanced bot/normal ratio with diversity across bot types
- Targets 5-10% filtering rate (after cleaning & trimming)
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Paths
BASE_DIR      = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "outputs")
DEMO_DIR      = os.path.join(BASE_DIR, "data", "demo")


def main():
    """
    Target: 3k-3.5k players after trimming
    Filter rate: 5-10% → need 3.2k-3.9k raw players (~3.5k optimal)
    
    Strategy:
    1. Load current heuristic labels to understand bot/normal distribution
    2. Load raw players.csv to get full population
    3. Sample players stratified by bot/normal status
    4. Create subset raw CSVs with selected playerids
    """
    
    os.makedirs(DEMO_DIR, exist_ok=True)
    log.info("Demo dataset directory: %s", os.path.abspath(DEMO_DIR))
    
    # ─────────────────────────────────────────────────────────────────────
    # Load heuristic labels from full run
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n=== PHASE 1: ANALYZE CURRENT BOT DISTRIBUTION ===")
    
    heuristic_path = os.path.join(OUTPUTS_DIR, "heuristic_labels.csv")
    if not os.path.exists(heuristic_path):
        log.error("  [!] heuristic_labels.csv not found in outputs/")
        log.error("      Please run the full pipeline first.")
        return
    
    heuristic_df = pd.read_csv(heuristic_path, dtype={"playerid": "int64"})
    
    bots = heuristic_df[heuristic_df["heuristic_bot"] == 1]["playerid"].tolist()
    normals = heuristic_df[heuristic_df["heuristic_normal"] == 1]["playerid"].tolist()
    unknown = heuristic_df[(heuristic_df["heuristic_bot"] == 0) & 
                           (heuristic_df["heuristic_normal"] == 0)]["playerid"].tolist()
    
    log.info("  Bots detected (heuristic):    %d (%.2f%%)", len(bots), 100*len(bots)/len(heuristic_df))
    log.info("  Normals confirmed (heuristic): %d (%.2f%%)", len(normals), 100*len(normals)/len(heuristic_df))
    log.info("  Unknown (no heuristic label):  %d (%.2f%%)", len(unknown), 100*len(unknown)/len(heuristic_df))
    log.info("  Total in feature matrix:       %d", len(heuristic_df))
    
    # ─────────────────────────────────────────────────────────────────────
    # Load raw players
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n=== PHASE 2: LOAD RAW PLAYERS ===")
    
    players_raw_path = os.path.join(RAW_DIR, "players.csv")
    players_raw = pd.read_csv(players_raw_path, dtype={"playerid": "int64"})
    log.info("  Raw players.csv: %d rows", len(players_raw))
    
    raw_playerids = set(players_raw["playerid"].tolist())
    
    # ─────────────────────────────────────────────────────────────────────
    # Target sample size
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n=== PHASE 3: STRATIFIED SAMPLING ===")
    
    TARGET_AFTER_TRIMMING = 3200  # Adjust target to ensure borderline quota
    EXPECTED_TRIM_RATE    = 0.10  # Target 10% filter rate (upper bound of 5-10%)
    TARGET_RAW            = int(TARGET_AFTER_TRIMMING / (1 - EXPECTED_TRIM_RATE))
    
    log.info("  Target after trimming: ~%d players", TARGET_AFTER_TRIMMING)
    log.info("  Expected trim rate: ~%.1f%%", 100*EXPECTED_TRIM_RATE)
    log.info("  Target raw sample: ~%d players", TARGET_RAW)
    
    # Ratio in full dataset
    current_bot_ratio = len(bots) / len(heuristic_df)
    log.info("  Current bot ratio in full set: %.2f%%", 100*current_bot_ratio)
    
    # Get available players
    available_bots = [pid for pid in bots if pid in raw_playerids]
    available_normals = [pid for pid in normals if pid in raw_playerids]
    available_unknown = [pid for pid in unknown if pid in raw_playerids]
    
    log.info("  Available in raw data:")
    log.info("    - Bots:    %d / %d", len(available_bots), len(bots))
    log.info("    - Normals: %d / %d", len(available_normals), len(normals))
    log.info("    - Unknown: %d / %d", len(available_unknown), len(unknown))
    
    # ─────────────────────────────────────────────────────────────────────
    # Strategy: Build demo with realistic filter rate (5-10%)
    # Include valid players + borderline players that will be filtered
    # ─────────────────────────────────────────────────────────────────────
    
    np.random.seed(42)
    
    sampled_bots = np.array(available_bots)  # Take ALL bots
    sampled_normals = np.array(available_normals)  # Take ALL confirmed normals
    
    labeled_total = len(sampled_bots) + len(sampled_normals)
    
    # Calculate quota for unknowns and borderline
    # Target: After filtering 10%, we want ~3,200 players
    # So we need: 3,200 / (1 - 0.10) ≈ 3,556 raw
    TARGET_FILTER_RATE = EXPECTED_TRIM_RATE
    target_raw_with_borderline = int(TARGET_AFTER_TRIMMING / (1 - TARGET_FILTER_RATE))
    unknown_quota = target_raw_with_borderline - labeled_total
    
    log.info("  Target with filter rate (%.1f%%): ~%d raw players",
             100*TARGET_FILTER_RATE, target_raw_with_borderline)
    log.info("  Unknown quota needed: %d", unknown_quota)
    
    # Add "unknown" players to reach target
    if unknown_quota > 0 and len(available_unknown) > 0:
        sampled_unknown = np.random.choice(available_unknown, 
                                          size=min(unknown_quota, len(available_unknown)), 
                                          replace=False)
    else:
        sampled_unknown = np.array([])
    
    selected_playerids = set(sampled_bots) | set(sampled_normals) | set(sampled_unknown)
    
    demo_bot_ratio = len(sampled_bots) / len(selected_playerids)
    
    log.info("\n  Selected:")
    log.info("    - Bots (all):        %d", len(sampled_bots))
    log.info("    - Normals (all):     %d", len(sampled_normals))
    log.info("    - Unknown (sampled): %d", len(sampled_unknown))
    log.info("    - Total:             %d", len(selected_playerids))
    log.info("    - Bot ratio:         %.2f%%", 100*demo_bot_ratio)
    log.info("\n  Expected after trimming:")
    log.info("    - Filter rate: ~%.1f%%", 100*TARGET_FILTER_RATE)
    log.info("    - Surviving players: ~%d", int(len(selected_playerids) * (1 - TARGET_FILTER_RATE)))
    
    # ─────────────────────────────────────────────────────────────────────
    # PHASE 3b: Add borderline players (will be filtered by trimming)
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n=== PHASE 3b: ADD BORDERLINE PLAYERS ===")
    
    # Load raw history and reviews to find borderline players
    history_raw_path = os.path.join(RAW_DIR, "history.csv")
    reviews_raw_path = os.path.join(RAW_DIR, "reviews.csv")
    purchased_raw_path = os.path.join(RAW_DIR, "purchased_games.csv")
    
    # Count achievements per player
    if os.path.exists(history_raw_path):
        history = pd.read_csv(history_raw_path, usecols=["playerid"], dtype={"playerid": "int64"})
        ach_counts = history.groupby("playerid").size()
        borderline_ach = set(ach_counts[(ach_counts >= 5) & (ach_counts < 10)].index.tolist())
        log.info("  Players with 5-9 achievements: %d", len(borderline_ach))
    else:
        borderline_ach = set()
    
    # Count reviews per player
    if os.path.exists(reviews_raw_path):
        reviews = pd.read_csv(reviews_raw_path, usecols=["playerid"], dtype={"playerid": "int64"})
        review_counts = reviews.groupby("playerid").size()
        borderline_reviews = set(review_counts[(review_counts >= 1) & (review_counts < 3)].index.tolist())
        log.info("  Players with 1-2 reviews (and <10 ach): %d", len(borderline_reviews))
    else:
        borderline_reviews = set()
    
    # Players with no library or small library
    if os.path.exists(purchased_raw_path):
        import json
        purchased = pd.read_csv(purchased_raw_path, usecols=["playerid", "library"], dtype={"playerid": "int64", "library": "string"})
        no_library = set()
        for pid, lib_str in zip(purchased["playerid"], purchased["library"]):
            try:
                if pd.isna(lib_str) or lib_str.strip() == "":
                    no_library.add(int(pid))
                else:
                    lib_data = json.loads(lib_str.replace("'", '"'))
                    if len(lib_data) == 0:
                        no_library.add(int(pid))
            except:
                pass
        log.info("  Players with no/empty library: %d", len(no_library))
    else:
        no_library = set()
    
    # Combine all borderline players
    all_borderline = (borderline_ach | borderline_reviews | no_library)
    # Remove those already selected
    borderline_candidates = all_borderline - selected_playerids
    
    log.info("  Borderline candidates (not yet selected): %d", len(borderline_candidates))
    
    # Add borderline players to reach filter rate target
    borderline_quota = max(0, target_raw_with_borderline - len(selected_playerids))
    
    if borderline_quota > 0 and len(borderline_candidates) > 0:
        added_borderline = np.random.choice(
            list(borderline_candidates),
            size=min(borderline_quota, len(borderline_candidates)),
            replace=False
        )
        log.info("  Adding %d borderline players to demo set", len(added_borderline))
        selected_playerids = selected_playerids | set(added_borderline)
    
    log.info("\n  Final selection:")
    log.info("    - Total players: %d", len(selected_playerids))
    log.info("    - Expected filter rate: ~%.1f%% (trimming removes borderline)",
             100*TARGET_FILTER_RATE)
    
    # ─────────────────────────────────────────────────────────────────────
    # Filter raw CSVs by selected playerids
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n=== PHASE 4: CREATE DEMO SUBSET CSVs ===")
    
    raw_files = ["players.csv", "history.csv", "reviews.csv", "purchased_games.csv"]
    
    for filename in raw_files:
        raw_path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(raw_path):
            log.warning("  [!] %s not found, skipping", filename)
            continue
        
        log.info("  Processing %s...", filename)
        
        try:
            df = pd.read_csv(raw_path)
        except Exception as e:
            log.warning("    Error reading %s: %s", filename, e)
            continue
        
        # Filter by playerid
        if "playerid" in df.columns:
            df_subset = df[df["playerid"].isin(selected_playerids)].copy()
        else:
            # Skip files without playerid
            log.warning("    Skipping %s (no playerid column)", filename)
            continue
        
        # Save to demo/
        demo_path = os.path.join(DEMO_DIR, filename)
        df_subset.to_csv(demo_path, index=False)
        
        reduction = (len(df) - len(df_subset)) / len(df) * 100
        log.info("    %d → %d rows (%.1f%% reduction) → %s", 
                 len(df), len(df_subset), reduction, demo_path)
    
    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n=== SUMMARY ===")
    log.info("Demo dataset created at: %s", DEMO_DIR)
    log.info("Selected players: %d (including borderline for realistic filter rate)", len(selected_playerids))
    log.info("  - Bots (heuristic):    %d (will survive trimming)", len(sampled_bots))
    log.info("  - Normals (heuristic): %d (will survive trimming)", len(sampled_normals))
    log.info("  - Unknown (sampled):   %d (mixed: some survive, some filtered)", len(sampled_unknown))
    borderline_count = len(selected_playerids) - len(sampled_bots) - len(sampled_normals) - len(sampled_unknown)
    log.info("  - Borderline (will be filtered): ~%d", max(0, borderline_count))
    log.info("\nExpected after trimming:")
    log.info("  Applying filter: (≥10 achievements OR ≥3 reviews) AND library_size ≥ 1")
    log.info("  With ~%.1f%% filter rate, expect ~%d players for modeling", 
             100*TARGET_FILTER_RATE, int(len(selected_playerids) * (1 - TARGET_FILTER_RATE)))
    log.info("\n[✓] Demo dataset ready with realistic filter behavior!")
    log.info("    Update data/raw → data/demo and re-run pipeline.")


if __name__ == "__main__":
    main()
