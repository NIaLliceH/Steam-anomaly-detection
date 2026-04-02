import os
import subprocess
import pandas as pd

# ==========================================
# CONFIGURATION OF TARGET ACCOUNTS TO CHECK
# ==========================================
TARGET_STEAM_IDS = [
    76561199761358443,
    76561198287996067
]

CRAWL_DIR = "data/crawled"
RAW_DIR = "data/raw"
OUTPUTS_DIR = "outputs"

def inject_crawled_data():
    """Inject data from crawled_data into raw-dataset, then delete crawl file to avoid duplicate injection"""
    print("=== 1. INJECTING NEW DATA ===")
    files = ["players.csv", "purchased_games.csv", "history.csv", "reviews.csv"]
    injected_any = False
    
    for file in files:
        crawl_path = os.path.join(CRAWL_DIR, file)
        raw_path = os.path.join(RAW_DIR, file)
        
        if os.path.exists(crawl_path):
            df_crawl = pd.read_csv(crawl_path)
            if not df_crawl.empty:
                df_crawl.to_csv(raw_path, mode='a', header=not os.path.exists(raw_path), index=False)
                print(f"[+] Injected {len(df_crawl)} records into {file}")
                injected_any = True
            # Delete file after injection to avoid duplicates on next run
            os.remove(crawl_path)
    
    if not injected_any:
        print("[-] No new data to inject.")

def run_ml_pipeline():
    """Automatically call Machine Learning scripts"""
    print("\n=== 2. RUNNING AI PIPELINE (Please wait...) ===")
    print("[*] Running Phase 1 (Data Prep)...")
    subprocess.run(["python", "src/data_prep.py"], check=True)
    
    print("[*] Running Phase 2 (Anomaly Detection Model)...")
    subprocess.run(["python", "main.py"], check=True)

def generate_report(target_ids):
    """Analyze results and print a user-friendly report"""
    print("\n" + "="*60)
    print("AI DETECTION REPORT FOR TARGET ACCOUNTS")
    print("="*60)
    
    try:
        results = pd.read_csv(os.path.join(OUTPUTS_DIR, "ensemble_results.csv"))
        features = pd.read_csv(os.path.join(OUTPUTS_DIR, "feature_matrix.csv"))
        
        # Calculate mean baseline for comparison
        normal_mean = features[~features['playerid'].isin(target_ids)].mean()
        
        for pid in target_ids:
            res = results[results['playerid'] == pid]
            feat = features[features['playerid'] == pid]
            
            if res.empty:
                print(f"[-] Steam ID: {pid} -> Not found in database!")
                continue
                
            score = res.iloc[0]['composite_score']
            is_bot = res.iloc[0]['is_anomaly'] == 1
            
            status = "ANOMALY DETECTED (BOT/FRAUD)" if is_bot else "NORMAL PLAYER"
            
            print(f"\nSTEAM ID: {pid}")
            print(f"   Status: {status}")
            print(f"   AI Suspicion Score: {score:.2f} / 100")
            print(f"   Model Votes: IF={res.iloc[0]['if_pct']:.1f}%, LOF={res.iloc[0]['lof_pct']:.1f}%, SVM={res.iloc[0]['svm_pct']:.1f}%")
            
            if is_bot and not feat.empty:
                print("   BEHAVIOR EVIDENCE (Why flagged?):")
                user_feat = feat.iloc[0]
                
                # Compare prominent metrics for reporting
                if user_feat['max_achievements_per_day'] > 500:
                    print(f"      - Abnormal speed: Unlocked {user_feat['max_achievements_per_day']:.0f} achievements/day (Average player only {normal_mean['max_achievements_per_day']:.0f})")
                
                if user_feat['review_unowned_ratio'] > 0.5:
                    print(f"      - Fake reviews: {user_feat['review_unowned_ratio']*100:.1f}% of reviews are for games not owned.")
                    
                if user_feat['cv_unlock_interval'] < 1.0 and user_feat['total_achievements'] > 100:
                    print(f"      - Tool/Macro usage: Unlock speed too uniform like a preset machine (Dispersion coefficient = {user_feat['cv_unlock_interval']:.2f})")

    except FileNotFoundError:
        print("[!] Results file not found. Make sure the Model ran successfully.")
    print("\n" + "="*60)

if __name__ == "__main__":
    inject_crawled_data()
    run_ml_pipeline()
    generate_report(TARGET_STEAM_IDS)