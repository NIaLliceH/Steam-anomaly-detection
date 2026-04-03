import os
import subprocess
import pandas as pd
from datetime import datetime

# ==========================================
# CONFIGURATION OF TARGET ACCOUNTS TO CHECK
# ==========================================
TARGET_STEAM_IDS = [
    76561198287996067,
    76561199761358443,
    76561198399223263,
    76561198350357346
]

CRAWL_DIR = "data/crawled"
RAW_DIR = "data/raw"
OUTPUTS_DIR = "outputs"

def inject_crawled_data():
    """Inject data from data/crawled to data/raw, then delete crawl file to avoid duplicate injection"""
    print("=== 1. INJECTING NEW DATA ===")
    files = ["players.csv", "purchased_games.csv", "history.csv", "reviews.csv"]
    injected_any = False
    
    for file in files:
        crawl_path = os.path.join(CRAWL_DIR, file)
        raw_path = os.path.join(RAW_DIR, file)
        archive_path = os.path.join("data/archive/", f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
        
        if os.path.exists(crawl_path):
            df_crawl = pd.read_csv(crawl_path)
            if not df_crawl.empty:
                if os.path.exists(raw_path):
                    # Ensure file ends with newline before appending
                    with open(raw_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if content and not content.endswith('\n'):
                        with open(raw_path, 'a', encoding='utf-8') as f:
                            f.write('\n')
                    # Append without header
                    df_crawl.to_csv(raw_path, mode='a', header=False, index=False)
                else:
                    # Create new file with header
                    df_crawl.to_csv(raw_path, index=False)
                
                print(f"[+] Injected {len(df_crawl)} records into {file}")
                injected_any = True
                
                # move file to archive
                os.makedirs(archive_path, exist_ok=True)
                os.rename(crawl_path, os.path.join(archive_path, file))
                print(f"[+] Moved {file} to {archive_path}/")
    
    
    if not injected_any:
        print("[-] No new data to inject.")

def run_ml_pipeline():
    """Automatically call Machine Learning scripts"""
    print("\n=== 2. RUNNING AI PIPELINE (Please wait...) ===")
    print("[*] Running Phase 1 (Data Prep)...")
    subprocess.run(["python3", "src/data_prep.py"], check=True)
    
    print("[*] Running Phase 2 (Anomaly Detection Model)...")
    subprocess.run(["python3", "main.py"], check=True)

def build_report_data(target_ids):
    """Build report data structure from results and features"""
    try:
        results = pd.read_csv(os.path.join(OUTPUTS_DIR, "ensemble_results.csv"))
        features = pd.read_csv(os.path.join(OUTPUTS_DIR, "feature_matrix.csv"))
        
        # Calculate mean baseline for comparison
        normal_mean = features[~features['playerid'].isin(target_ids)].mean()
        
        report_data = []
        for pid in target_ids:
            res = results[results['playerid'] == pid]
            feat = features[features['playerid'] == pid]
            
            if res.empty:
                report_data.append({
                    'playerid': pid,
                    'found': False
                })
                continue
            
            score = res.iloc[0]['composite_score']
            is_bot = res.iloc[0]['is_anomaly'] == 1
            user_feat = feat.iloc[0]
            
            entry = {
                'playerid': pid,
                'found': True,
                'score': score,
                'is_bot': is_bot,
                'status': "ANOMALY DETECTED (BOT/FRAUD)" if is_bot else "NORMAL PLAYER",
                'if_pct': res.iloc[0]['if_pct'],
                'lof_pct': res.iloc[0]['lof_pct'],
                'svm_pct': res.iloc[0]['svm_pct'],
                'max_achievements_per_day': user_feat.get('max_achievements_per_day', 0),
                'normal_avg_achievements': normal_mean['max_achievements_per_day'],
                'review_unowned_ratio': user_feat.get('review_unowned_ratio', 0),
                'cv_unlock_interval': user_feat.get('cv_unlock_interval', 0),
                'total_achievements': user_feat.get('total_achievements', 0)
            }
            report_data.append(entry)
        
        return report_data
    except FileNotFoundError:
        return None

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
            md += "> ⚠️ Not found in database!\n\n"
            continue
        
        status_icon = "🚨" if entry['is_bot'] else "✅"
        md += f"**Status:** {status_icon} {entry['status']}\n\n"
        md += f"**AI Suspicion Score:** {entry['score']:.2f} / 100\n\n"
        md += f"**Model Votes:**\n"
        md += f"- Isolation Forest: {entry['if_pct']:.1f}%\n"
        md += f"- Local Outlier Factor: {entry['lof_pct']:.1f}%\n"
        md += f"- SVM: {entry['svm_pct']:.1f}%\n\n"
        
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
            html += '</div>\n<div class="not-found">⚠️ Not found in database!</div>\n'
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
        html += f'<div class="vote-item"><div class="vote-label">Local Outlier Factor</div><div class="vote-percent">{entry["lof_pct"]:.1f}%</div></div>\n'
        html += f'<div class="vote-item"><div class="vote-label">SVM</div><div class="vote-percent">{entry["svm_pct"]:.1f}%</div></div>\n'
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
                print(f"[-] Steam ID: {pid} -> Not found in database!")
                continue
            
            status = entry['status']
            score = entry['score']
            
            print(f"\nSTEAM ID: {pid}")
            print(f"   Status: {status}")
            print(f"   AI Suspicion Score: {score:.2f} / 100")
            print(f"   Model Votes: IF={entry['if_pct']:.1f}%, LOF={entry['lof_pct']:.1f}%, SVM={entry['svm_pct']:.1f}%")
            
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
    inject_crawled_data()
    run_ml_pipeline()
    generate_report(TARGET_STEAM_IDS, output_format='all')