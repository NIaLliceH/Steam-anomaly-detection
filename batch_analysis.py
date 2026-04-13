import os
import subprocess
import pandas as pd
from datetime import datetime
import argparse
import sys

# ==========================================
# CONFIGURATION OF TARGET ACCOUNTS TO CHECK
# ==========================================
TARGET_STEAM_IDS = [
    76561198287996067,
    76561198399223263,
    76561198350357346,
    76561198405841744,
    76561198354838543,
    76561198391038255,
    76561198147116758
]

CRAWL_DIR = "data/crawled"
RAW_DIR = "data/raw"
OUTPUTS_DIR = "outputs"

def inject_crawled_data():
    """Check if any crawled data files are present in data/crawled/.

    The actual merging happens in-memory inside src/data_prep.py — raw CSV
    files are never modified and crawled files are never moved or deleted.

    Returns:
        bool: True if at least one crawled file exists and is non-empty.
    """
    print("=== 1. CHECKING FOR NEW CRAWLED DATA ===")
    files = ["players.csv", "purchased_games.csv", "history.csv", "reviews.csv"]
    found_any = False

    for file in files:
        crawl_path = os.path.join(CRAWL_DIR, file)
        if os.path.exists(crawl_path) and os.path.getsize(crawl_path) > 0:
            row_count = sum(1 for _ in open(crawl_path, encoding="utf-8")) - 1
            print(f"[+] Found crawled/{file}  ({row_count} rows — will be merged in-memory during preprocessing)")
            found_any = True

    if not found_any:
        print("[-] No crawled data found.")

    return found_any

def run_ml_pipeline():
    """Automatically call Machine Learning scripts"""
    print("\n=== 2. RUNNING AI PIPELINE (Please wait...) ===")
    print("[*] Running Phase 1 (Data Prep)...")
    subprocess.run(["python3", "src/data_prep.py"], check=True)
    
    print("[*] Running Phase 2 (Anomaly Detection Model)...")
    subprocess.run(["python3", "main.py"], check=True)

def _load_known_playerids():
    """
    Return a dict mapping playerid → reason_filtered for players that exist
    somewhere in the pipeline but are absent from ensemble_results.csv.

    Lookup order (most to least specific):
      1. feature_matrix.csv (pre-trimming): player had activity data but failed
         the trimming filter (< 10 achievements OR library_size < 1).
      2. data/processed/players.parquet: player account exists but had no
         achievements or reviews — never entered feature engineering.
      3. data/raw/players.csv: player in raw data but removed as private account
         during Phase 1 ETL.
    """
    sources = {}

    # Source 1 — feature matrix (saved before trimming in main.py)
    fm_path = os.path.join(OUTPUTS_DIR, "feature_matrix.csv")
    if os.path.exists(fm_path):
        try:
            fm_ids = pd.read_csv(fm_path, usecols=["playerid"])["playerid"]
            for pid in fm_ids:
                sources[int(pid)] = (
                    "Player exists in data but was excluded from the model during "
                    "trimming (< 10 achievements or no game library — likely a "
                    "sparse/inactive account)."
                )
        except Exception:
            pass

    # Source 2 — processed players parquet (after private-account filter)
    proc_path = os.path.join("data", "processed", "players.parquet")
    if os.path.exists(proc_path):
        try:
            proc_ids = pd.read_parquet(proc_path, columns=["playerid"])["playerid"]
            for pid in proc_ids:
                pid = int(pid)
                if pid not in sources:
                    sources[pid] = (
                        "Player account is known but had no achievement or review "
                        "history — not enough data to analyse."
                    )
        except Exception:
            pass

    # Source 3 — raw players CSV (includes private accounts)
    raw_path = os.path.join("data", "raw", "players.csv")
    if os.path.exists(raw_path):
        try:
            raw_ids = pd.read_csv(raw_path, usecols=["playerid"])["playerid"]
            for pid in raw_ids:
                pid = int(pid)
                if pid not in sources:
                    sources[pid] = (
                        "Player is in raw data but was removed during Phase 1 ETL "
                        "(private account filter)."
                    )
        except Exception:
            pass

    return sources


def build_report_data(target_ids):
    """Build report data structure from results and features."""
    try:
        results = pd.read_csv(os.path.join(OUTPUTS_DIR, "ensemble_results.csv"))
        features = pd.read_csv(os.path.join(OUTPUTS_DIR, "feature_matrix.csv"))
    except FileNotFoundError:
        return None

    # Build lookup for players filtered out during preprocessing/trimming
    known_ids = _load_known_playerids()

    # Calculate mean baseline for comparison (exclude target ids)
    normal_mean = features[~features['playerid'].isin(target_ids)].mean(numeric_only=True)

    report_data = []
    for pid in target_ids:
        res  = results[results['playerid'] == pid]
        feat = features[features['playerid'] == pid]

        if res.empty:
            filter_reason = known_ids.get(int(pid))
            report_data.append({
                'playerid': pid,
                'found': False,
                'filtered': filter_reason is not None,
                'filter_reason': filter_reason or "Player ID not found in any data source.",
            })
            continue

        score    = res.iloc[0]['composite_score']
        is_bot   = res.iloc[0]['is_anomaly'] == 1
        user_feat = feat.iloc[0] if not feat.empty else pd.Series(dtype=float)

        entry = {
            'playerid': pid,
            'found': True,
            'score': score,
            'is_bot': is_bot,
            'status': "ANOMALY DETECTED (BOT/FRAUD)" if is_bot else "NORMAL PLAYER",
            'if_pct': res.iloc[0]['if_pct'],
            'max_achievements_per_day': user_feat.get('max_achievements_per_day', 0),
            'normal_avg_achievements': normal_mean['max_achievements_per_day'],
            'review_unowned_ratio': user_feat.get('review_unowned_ratio', 0),
            'cv_unlock_interval': user_feat.get('cv_unlock_interval', 0),
            'total_achievements': user_feat.get('total_achievements', 0),
        }
        report_data.append(entry)

    return report_data

def generate_markdown_report(target_ids):
    """Generate Markdown formatted report"""
    report_data = build_report_data(target_ids)
    if report_data is None:
        return None
    
    md = "# AI Detection Report for Target Accounts\n\n"
    md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for entry in report_data:
        pid = entry['playerid']
        md += f"## Steam ID: {pid}\n\n"
        
        if not entry['found']:
            if entry['filtered']:
                md += "> ℹ️ Player excluded from model during preprocessing.\n\n"
                md += f"> **Reason:** {entry['filter_reason']}\n\n"
            else:
                md += "> ❌ Not found in any data source.\n\n"
            continue
        
        status_icon = "🚨" if entry['is_bot'] else "✅"
        md += f"**Status:** {status_icon} {entry['status']}\n\n"
        md += f"**AI Suspicion Score:** {entry['score']:.2f} / 100\n\n"
        md += f"**Model Scores:**\n"
        md += f"- Isolation Forest: {entry['if_pct']:.1f}%\n\n"
        
        if entry['is_bot']:
            md += "### Behavior Evidence (Why flagged?)\n\n"
            if entry['max_achievements_per_day'] > 500:
                md += f"- **Abnormal speed:** Unlocked {entry['max_achievements_per_day']:.0f} achievements/day (Average player only {entry['normal_avg_achievements']:.0f})\n"
            
            if entry['review_unowned_ratio'] > 0.5:
                md += f"- **Fake reviews:** {entry['review_unowned_ratio']*100:.1f}% of reviews are for games not owned.\n"
            
            if entry['cv_unlock_interval'] < 1.0 and entry['total_achievements'] > 100:
                md += f"- **Tool/Macro usage:** Unlock speed too uniform like a preset machine (Dispersion coefficient = {entry['cv_unlock_interval']:.2f})\n"
            md += "\n"
        
        md += "---\n\n"
    
    return md

def generate_html_report(target_ids):
    """Generate HTML formatted report"""
    report_data = build_report_data(target_ids)
    if report_data is None:
        return None
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Detection Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        .report-meta {{
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }}
        .account-card {{
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: #fafafa;
        }}
        .account-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .steam-id {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }}
        .status {{
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
        }}
        .status.anomaly {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .status.normal {{
            background-color: #d4edda;
            color: #155724;
        }}
        .score-section {{
            background-color: white;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 15px;
        }}
        .score-display {{
            font-size: 32px;
            font-weight: bold;
            color: #007bff;
        }}
        .score-label {{
            color: #666;
            font-size: 14px;
        }}
        .model-votes {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 10px;
        }}
        .vote-item {{
            background-color: white;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }}
        .vote-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        .vote-percent {{
            font-size: 20px;
            font-weight: bold;
            color: #007bff;
        }}
        .evidence-section {{
            background-color: #fff8e1;
            border-left: 4px solid #ffc107;
            padding: 15px;
            border-radius: 4px;
            margin-top: 15px;
        }}
        .evidence-title {{
            font-weight: bold;
            color: #856404;
            margin-bottom: 10px;
        }}
        .evidence-item {{
            color: #856404;
            margin-bottom: 8px;
            padding-left: 20px;
            position: relative;
        }}
        .evidence-item:before {{
            content: "⚠";
            position: absolute;
            left: 0;
        }}
        .not-found {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 4px;
        }}
        .not-found.filtered {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .footer {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 AI Detection Report for Target Accounts</h1>
        <div class="report-meta">Generated: {timestamp}</div>
""".format(timestamp=timestamp)
    
    for entry in report_data:
        pid = entry['playerid']
        html += f'<div class="account-card">\n'
        html += f'<div class="account-header">\n'
        html += f'<span class="steam-id">Steam ID: {pid}</span>\n'
        
        if not entry['found']:
            html += '</div>\n'
            if entry['filtered']:
                html += '<div class="not-found filtered">ℹ️ Player excluded from model during preprocessing.<br>'
                html += f'<small>{entry["filter_reason"]}</small></div>\n'
            else:
                html += '<div class="not-found">❌ Not found in any data source.</div>\n'
            html += '</div>\n'
            continue
        
        status_class = "anomaly" if entry['is_bot'] else "normal"
        status_icon = "🚨" if entry['is_bot'] else "✅"
        html += f'<span class="status {status_class}">{status_icon} {entry["status"]}</span>\n'
        html += '</div>\n'
        
        html += '<div class="score-section">\n'
        html += f'<div class="score-label">AI Suspicion Score</div>\n'
        html += f'<div class="score-display">{entry["score"]:.2f}</div>\n'
        html += '<div class="model-votes">\n'
        html += f'<div class="vote-item"><div class="vote-label">Isolation Forest</div><div class="vote-percent">{entry["if_pct"]:.1f}%</div></div>\n'
        html += '</div>\n</div>\n'
        
        if entry['is_bot']:
            html += '<div class="evidence-section">\n'
            html += '<div class="evidence-title">🔎 Behavior Evidence (Why flagged?)</div>\n'
            
            if entry['max_achievements_per_day'] > 500:
                html += f'<div class="evidence-item">Abnormal speed: Unlocked {entry["max_achievements_per_day"]:.0f} achievements/day (Average player only {entry["normal_avg_achievements"]:.0f})</div>\n'
            
            if entry['review_unowned_ratio'] > 0.5:
                html += f'<div class="evidence-item">Fake reviews: {entry["review_unowned_ratio"]*100:.1f}% of reviews are for games not owned.</div>\n'
            
            if entry['cv_unlock_interval'] < 1.0 and entry['total_achievements'] > 100:
                html += f'<div class="evidence-item">Tool/Macro usage: Unlock speed too uniform like a preset machine (Dispersion coefficient = {entry["cv_unlock_interval"]:.2f})</div>\n'
            
            html += '</div>\n'
        
        html += '</div>\n'
    
    html += """
        <div class="footer">
            <p>Report Generated by AI Detection Pipeline</p>
        </div>
    </div>
</body>
</html>
"""
    return html

def generate_report(target_ids, output_format='console'):
    """Analyze results and generate report in specified format
    
    Args:
        target_ids: List of Steam IDs to analyze
        output_format: 'console' (default), 'markdown', 'html', or 'all'
    """
    if output_format == 'console' or output_format == 'all':
        print("\n" + "="*60)
        print("AI DETECTION REPORT FOR TARGET ACCOUNTS")
        print("="*60)
    
        
        report_data = build_report_data(target_ids)
        if report_data is None:
            print("[!] Results file not found. Make sure the Model ran successfully.")
            return
        
        for entry in report_data:
            pid = entry['playerid']

            if not entry['found']:
                if entry['filtered']:
                    print(f"[~] Steam ID: {pid} -> Excluded from model during preprocessing.")
                    print(f"    Reason: {entry['filter_reason']}")
                else:
                    print(f"[-] Steam ID: {pid} -> Not found in any data source.")
                continue
            
            status = entry['status']
            score = entry['score']
            
            print(f"\nSTEAM ID: {pid}")
            print(f"   Status: {status}")
            print(f"   AI Suspicion Score: {score:.2f} / 100")
            print(f"   Model Scores: IF={entry['if_pct']:.1f}%")
            
            if entry['is_bot']:
                print("   BEHAVIOR EVIDENCE (Why flagged?):")
                
                if entry['max_achievements_per_day'] > 500:
                    print(f"      - Abnormal speed: Unlocked {entry['max_achievements_per_day']:.0f} achievements/day (Average player only {entry['normal_avg_achievements']:.0f})")
                
                if entry['review_unowned_ratio'] > 0.5:
                    print(f"      - Fake reviews: {entry['review_unowned_ratio']*100:.1f}% of reviews are for games not owned.")
                    
                if entry['cv_unlock_interval'] < 1.0 and entry['total_achievements'] > 100:
                    print(f"      - Tool/Macro usage: Unlock speed too uniform like a preset machine (Dispersion coefficient = {entry['cv_unlock_interval']:.2f})")
        
        print("\n" + "="*60)
    
    # Generate Markdown report
    if output_format == 'markdown' or output_format == 'all':
        md_content = generate_markdown_report(target_ids)
        if md_content:
            md_file = os.path.join(OUTPUTS_DIR, "detection_report.md")
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            print(f"[+] Markdown report saved: {md_file}")
    
    # Generate HTML report
    if output_format == 'html' or output_format == 'all':
        html_content = generate_html_report(target_ids)
        if html_content:
            html_file = os.path.join(OUTPUTS_DIR, "detection_report.html")
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"[+] HTML report saved: {html_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch analysis of Steam profiles for anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 batch_analysis.py                           # Full pipeline (if new data exists)
  python3 batch_analysis.py --query-only              # Skip injection, query existing results only
  python3 batch_analysis.py --query-only --steam-ids 123 456  # Query specific player IDs
  python3 batch_analysis.py --force-run               # Force run pipeline even without new data
        """
    )
    
    parser.add_argument(
        '--query-only',
        action='store_true',
        help='Skip data injection and pipeline; only generate reports from existing results'
    )
    
    parser.add_argument(
        '--steam-ids',
        nargs='+',
        type=int,
        help='Specific Steam IDs to query (overrides TARGET_STEAM_IDS)'
    )
    
    parser.add_argument(
        '--format',
        choices=['markdown', 'html', 'all'],
        default='all',
        help='Report output format (default: all)'
    )
    
    parser.add_argument(
        '--force-run',
        action='store_true',
        help='Force run the entire pipeline even if no new data is injected'
    )
    
    args = parser.parse_args()
    
    # Determine target IDs
    target_ids = args.steam_ids if args.steam_ids else TARGET_STEAM_IDS
    
    # Execute based on arguments
    if args.query_only:
        print("[*] QUERY-ONLY MODE: Skipping injection and pipeline.")
        print(f"[*] Generating reports for {len(target_ids)} player IDs...\n")
        generate_report(target_ids, output_format=args.format)
    else:
        # Inject data and check if anything was injected
        data_injected = inject_crawled_data()
        
        if data_injected or args.force_run:
            print("\n=== 2. RUNNING AI PIPELINE (Please wait...) ===")
            print("[*] Running Phase 1 (Data Prep)...")
            subprocess.run(["python3", "src/data_prep.py"], check=True)
            
            print("[*] Running Phase 2 (Anomaly Detection Model)...")
            subprocess.run(["python3", "main.py"], check=True)
        else:
            print("\n[*] No new data injected. Skipping pipeline and using existing results.")
        
        print(f"\n[*] Generating reports for {len(target_ids)} player IDs...\n")
        generate_report(target_ids, output_format=args.format)