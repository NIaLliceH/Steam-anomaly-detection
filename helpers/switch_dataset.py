"""
Dataset Switcher Helper
Switch between demo and full datasets easily.

Usage:
  python helpers/switch_dataset.py --mode demo    # Use demo (3.3k players)
  python helpers/switch_dataset.py --mode full    # Use full (424k players)
  python helpers/switch_dataset.py --status       # Show current status
"""

import os
import sys
import shutil
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DEMO_DIR = os.path.join(BASE_DIR, "data", "demo")
RAW_BACKUP_DIR = os.path.join(BASE_DIR, "data", "raw_full")


def get_status():
    """Check which dataset is currently active."""
    if os.path.exists(RAW_DIR) and os.path.isdir(RAW_DIR):
        csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
        
        if not csv_files:
            return "empty"
        
        # Check if demo marker exists
        try:
            players_csv = os.path.join(RAW_DIR, "players.csv")
            with open(players_csv, 'r') as f:
                header = f.readline()
                for i, line in enumerate(f):
                    if i >= 10:  # Check first 10 lines
                        break
            
            # Count players in raw
            df_lines = sum(1 for line in open(players_csv)) - 1  # -1 for header
            
            if df_lines < 10000:
                return "demo"
            else:
                return "full"
        except Exception as e:
            log.warning("Could not determine status: %s", e)
            return "unknown"
    
    return "missing"


def switch_to_demo():
    """Activate demo dataset."""
    status = get_status()
    
    if status == "demo":
        log.info("[i] Already in demo mode")
        return True
    
    if status == "full":
        log.info("[*] Switching to demo mode...")
        log.info("    Backing up full dataset to data/raw_full/")
        
        if os.path.exists(RAW_BACKUP_DIR):
            shutil.rmtree(RAW_BACKUP_DIR)
        shutil.move(RAW_DIR, RAW_BACKUP_DIR)
        log.info("    [✓] Full dataset backed up")
    
    if not os.path.exists(DEMO_DIR):
        log.error("[!] Demo dataset not found at %s", DEMO_DIR)
        log.error("    Run: python helpers/create_demo_subset.py")
        return False
    
    log.info("    Activating demo dataset...")
    
    # Copy demo files to raw
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # Demo subset files (player-specific CSVs)
    demo_files = ["players.csv", "history.csv", "reviews.csv", "purchased_games.csv"]
    for fname in demo_files:
        src = os.path.join(DEMO_DIR, fname)
        dst = os.path.join(RAW_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    # Copy non-subset files from backup (if available) or from raw_full
    # These files are NOT subsetted (all players needed for metadata)
    backup_source = RAW_BACKUP_DIR if os.path.exists(RAW_BACKUP_DIR) else None
    
    if backup_source:
        other_files = ["private_steamids.csv", "achievements.csv", "games.csv"]
        for fname in other_files:
            src = os.path.join(backup_source, fname)
            dst = os.path.join(RAW_DIR, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                log.info("    Copied %s from backup", fname)
    
    log.info("    [✓] Demo dataset activated")
    
    log.info("\n[✓] Switched to DEMO mode")
    log.info("    Players: 3,322")
    log.info("    Expected after trimming: ~3,072")
    log.info("    Processing time: ~7-8 minutes")
    return True


def switch_to_full():
    """Activate full dataset."""
    status = get_status()
    
    if status == "full":
        log.info("[i] Already in full mode")
        return True
    
    if status == "demo":
        log.info("[*] Switching to full mode...")
        
        if os.path.exists(RAW_BACKUP_DIR):
            log.info("    Restoring full dataset from backup...")
            shutil.rmtree(RAW_DIR)
            shutil.move(RAW_BACKUP_DIR, RAW_DIR)
            log.info("    [✓] Full dataset restored")
        else:
            log.warning("[!] Full dataset backup not found")
            log.warning("    Cannot restore from demo. Please restore manually:")
            log.warning("    1. Check if data/raw_full/ exists")
            log.warning("    2. Run: mv data/raw_full data/raw")
            return False
    
    log.info("\n[✓] Switched to FULL mode")
    log.info("    Players: 424,692")
    log.info("    Processing time: ~30 minutes")
    return True


def show_status():
    """Display current dataset status."""
    status = get_status()
    
    if status == "demo":
        mode = "🔍 DEMO"
        players = "~3,322"
        expected = "~3,072 (after trim)"
        time = "7-8 min"
    elif status == "full":
        mode = "📊 FULL"
        players = "424,692"
        expected = "~22,583 (after trim)"
        time = "30 min"
    else:
        mode = "⚠️  UNKNOWN"
        players = "?"
        expected = "?"
        time = "?"
    
    print(f"\n{'='*60}")
    print(f"Current Mode:        {mode}")
    print(f"Players:             {players}")
    print(f"Expected (modeled):  {expected}")
    print(f"Est. Runtime:        {time}")
    print(f"{'='*60}\n")
    
    if status == "unknown" or status == "missing":
        print("⚠️  Could not determine status. Checking directories...\n")
        
        print(f"data/raw/:     {os.path.exists(RAW_DIR)} - ", end="")
        if os.path.exists(RAW_DIR):
            files = os.listdir(RAW_DIR)
            print(f"{len(files)} items")
        else:
            print("missing")
        
        print(f"data/demo/:    {os.path.exists(DEMO_DIR)} - ", end="")
        if os.path.exists(DEMO_DIR):
            files = os.listdir(DEMO_DIR)
            print(f"{len(files)} items")
        else:
            print("missing")
        
        print(f"data/raw_full/: {os.path.exists(RAW_BACKUP_DIR)} - ", end="")
        if os.path.exists(RAW_BACKUP_DIR):
            files = os.listdir(RAW_BACKUP_DIR)
            print(f"{len(files)} items (backup)")
        else:
            print("empty")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Switch between demo (fast) and full (comprehensive) datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python helpers/switch_dataset.py --mode demo       # Use demo dataset
  python helpers/switch_dataset.py --mode full       # Use full dataset
  python helpers/switch_dataset.py --status          # Check current status
        """
    )
    parser.add_argument("--mode", choices=["demo", "full"], 
                        help="Dataset mode to switch to")
    parser.add_argument("--status", action="store_true",
                        help="Show current dataset status")
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
    elif args.mode == "demo":
        switch_to_demo()
    elif args.mode == "full":
        switch_to_full()
    else:
        show_status()


if __name__ == "__main__":
    main()
