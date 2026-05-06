#!/usr/bin/env python3
"""
Steam Anomaly Detection — Standalone EDA Script

Run this AFTER Phase 1 (data_prep.py) to explore processed parquets.
Can be run independently of the full pipeline.

Usage:
    python3 eda_standalone.py [--output-dir OUTPUT_DIR] [--no-plots]

Example:
    python3 eda_standalone.py --output-dir outputs/eda_results
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Config
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE_SINGLE = (14, 6)
FIGSIZE_WIDE = (16, 8)

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
OUTPUTS_DIR = Path('outputs')


def count_rows_fast(path, chunk_size=1024 * 1024):
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return max(
            0,
            sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(chunk_size), b"")) - 1,
        )


def sample_raw_csv(path, nrows=100000):
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception:
        return pd.read_csv(path, nrows=nrows, low_memory=False, engine="python", on_bad_lines="skip")


def get_core_ids(history_df, reviews_df, purchased_df):
    ach_counts = history_df.groupby("playerid").size()
    review_counts = reviews_df.groupby("playerid").size()

    if "library_size" in purchased_df.columns:
        lib_sizes = purchased_df.set_index("playerid")["library_size"].fillna(0)
    else:
        lib_sizes = purchased_df.set_index("playerid")["library"].map(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) and x else 0
        ).fillna(0)

    has_library = lib_sizes[lib_sizes >= 1].index
    core_ids = (
        ach_counts[ach_counts >= 10].index
        .union(review_counts[review_counts >= 3].index)
        .intersection(has_library)
    )
    return pd.Index(core_ids)


def load_data():
    """Load processed parquet files."""
    print("[*] Loading processed data...")
    try:
        history = pd.read_parquet(PROCESSED_DIR / 'history.parquet')
        players = pd.read_parquet(PROCESSED_DIR / 'players.parquet')
        reviews = pd.read_parquet(PROCESSED_DIR / 'reviews.parquet')
        purchased = pd.read_parquet(PROCESSED_DIR / 'purchased.parquet')
        
        # Parse datetimes
        history['date_acquired'] = pd.to_datetime(history['date_acquired'], errors='coerce')
        players['created'] = pd.to_datetime(players['created'], errors='coerce')
        reviews['posted'] = pd.to_datetime(reviews['posted'], errors='coerce')
        
        return history, players, reviews, purchased
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run Phase 1 first: python3 src/data_prep.py")
        sys.exit(1)


def profile_raw_data(sample_rows=100000, full_scan=True):
    """Profile raw CSVs to justify preprocessing steps."""
    print("\n" + "=" * 80)
    print("SECTION 0: RAW DATA PROFILING (PREPROCESSING RATIONALE)")
    print("=" * 80)

    raw_specs = [
        ("history.csv", ["playerid", "achievementid", "date_acquired"], ["date_acquired"]),
        ("reviews.csv", ["reviewid", "playerid", "gameid", "posted"], ["posted"]),
        ("players.csv", ["playerid", "country", "created"], ["created"]),
        ("purchased_games.csv", ["playerid", "library"], []),
        ("private_steamids.csv", ["playerid"], []),
    ]

    scan_label = "full" if full_scan else f"sample({sample_rows:,})"

    def read_raw_csv(path):
        if full_scan:
            try:
                return pd.read_csv(path, low_memory=False)
            except Exception:
                return pd.read_csv(path, low_memory=False, engine="python", on_bad_lines="skip")
        return sample_raw_csv(path, nrows=sample_rows)

    for fname, key_cols, date_cols in raw_specs:
        path = RAW_DIR / fname
        if not path.exists():
            print(f"- {fname}: MISSING")
            continue

        file_mb = path.stat().st_size / 1024**2
        total_rows = count_rows_fast(path)
        sample = read_raw_csv(path)
        sample_len = len(sample)

        key_cols_present = [c for c in key_cols if c in sample.columns]
        missing_key_pct = {}
        if key_cols_present:
            missing_key_pct = (sample[key_cols_present].isna().mean() * 100).round(2).to_dict()

        parse_error_pct = {}
        for col in date_cols:
            if col in sample.columns:
                dt = pd.to_datetime(sample[col], errors="coerce")
                bad = (dt.isna() & sample[col].notna()).sum()
                parse_error_pct[col] = round((bad / max(sample_len, 1)) * 100, 2)

        if key_cols_present:
            dup_key_pct = round(sample.duplicated(subset=key_cols_present).mean() * 100, 2)
        else:
            dup_key_pct = round(sample.duplicated().mean() * 100, 2)
        dup_row_pct = round(sample.duplicated().mean() * 100, 2) if sample_len else 0.0

        print(f"\n- {fname}")
        print(f"  size_mb: {file_mb:.2f}, rows: {total_rows:,}, scan_rows: {sample_len:,} ({scan_label})")
        for k, v in missing_key_pct.items():
            print(f"  missing_pct[{k}]: {v:.2f}% (sample)")
        for k, v in parse_error_pct.items():
            print(f"  parse_error_pct[{k}]: {v:.2f}% (sample)")
        print(f"  duplicate_key_pct: {dup_key_pct:.2f}%")
        print(f"  duplicate_row_pct: {dup_row_pct:.2f}%")

    print("\nPreprocessing actions supported by raw profiling:")
    print("- Fix data types: parse datetime columns and enforce numeric playerid/gameid")
    print("- Remove duplicate rows after key-based checks")
    print("- Handle malformed purchased_games rows via robust CSV reader")
    print("- Drop private steam IDs before any analysis")
    print("- Normalize library format: list of dicts with appid and playtime_mins")
    print("- Leave semantic outliers for EDA (bots are expected anomalies)")


def load_feature_matrix():
    """Load feature matrix if available."""
    path = OUTPUTS_DIR / 'feature_matrix.csv'
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


def load_heuristic_labels():
    """Load heuristic labels if available."""
    path = OUTPUTS_DIR / 'heuristic_labels.csv'
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


def section_1_data_overview(history, players, reviews, purchased, output_dir):
    """Section 1: Data Overview"""
    print("\n" + "=" * 80)
    print("SECTION 1: DATA OVERVIEW")
    print("=" * 80)
    
    tables = {'history': history, 'players': players, 'reviews': reviews, 'purchased': purchased}
    
    for name, df in tables.items():
        n_rows = len(df)
        n_players = df['playerid'].nunique()
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"\n{name.upper()}:")
        print(f"  Rows: {n_rows:,}")
        print(f"  Unique players: {n_players:,}")
        print(f"  Memory: {memory_mb:.2f} MB")
    
    # Ghost account estimate (same logic as build_feature_matrix)
    core_ids = get_core_ids(history, reviews, purchased)
    total_unique = len(set(history["playerid"].unique()) | set(reviews["playerid"].unique()))

    print("\nGhost Account Trimming (Expected):")
    print(f"  Before: {total_unique:,}")
    print(f"  After: {len(core_ids):,}")
    print(f"  Removed (ghost): {total_unique - len(core_ids):,} ({(1 - len(core_ids)/max(total_unique, 1))*100:.1f}%)")


def pipeline_stage_summary(
    history,
    players,
    reviews,
    purchased,
    feature_matrix,
    heuristic,
    raw_full_scan=True,
    raw_sample_rows=100000,
):
    """Summarize stats for raw -> processed -> trimmed -> features -> model stages."""
    print("\n" + "=" * 80)
    print("PIPELINE STAGE SUMMARY (RAW -> PROCESSED -> TRIMMED -> FEATURES -> MODEL)")
    print("=" * 80)
    scan_label = "full" if raw_full_scan else f"sample({raw_sample_rows:,})"
    print(f"Note: Raw stage uses {scan_label} for null and duplicate estimates.\n")

    key_cols_map = {
        "history": ["playerid", "achievementid", "date_acquired"],
        "reviews": ["reviewid"],
        "players": ["playerid"],
        "purchased": ["playerid"],
    }

    def summarize_table(stage, name, df, key_cols=None, rows_est=None):
        rows = len(df)
        unique_players = df["playerid"].nunique() if "playerid" in df.columns else np.nan
        if key_cols:
            dup_pct = round(df.duplicated(subset=key_cols).mean() * 100, 2)
        else:
            dup_pct = round(df.duplicated().mean() * 100, 2)

        null_pct = (df.isna().mean() * 100).round(2)
        null_max = null_pct.max() if len(null_pct) else 0.0
        null_mean = null_pct.mean() if len(null_pct) else 0.0

        return {
            "stage": stage,
            "table": name,
            "rows": rows,
            "rows_est": rows_est if rows_est is not None else rows,
            "unique_players": unique_players,
            "columns": len(df.columns),
            "dup_key_pct": dup_pct,
            "null_pct_max": round(float(null_max), 2),
            "null_pct_mean": round(float(null_mean), 2),
        }, null_pct

    def print_nulls(stage, name, null_pct):
        null_tbl = null_pct[null_pct > 0].sort_values(ascending=False)
        if null_tbl.empty:
            print(f"  - {stage}/{name}: no nulls")
            return
        null_df = pd.DataFrame({"column": null_tbl.index, "null_pct": null_tbl.values})
        print(f"  - {stage}/{name}:")
        print(null_df.to_string(index=False))

    summary_rows = []

    # Stage A: Raw (sample-based)
    raw_files = {
        "history": "history.csv",
        "reviews": "reviews.csv",
        "players": "players.csv",
        "purchased": "purchased_games.csv",
    }
    raw_tables = {}
    raw_rows_est = {}
    for name, fname in raw_files.items():
        path = RAW_DIR / fname
        if not path.exists():
            continue
        raw_rows_est[name] = count_rows_fast(path)
        if raw_full_scan:
            try:
                raw_tables[name] = pd.read_csv(path, low_memory=False)
            except Exception:
                raw_tables[name] = pd.read_csv(path, low_memory=False, engine="python", on_bad_lines="skip")
        else:
            raw_tables[name] = sample_raw_csv(path, nrows=raw_sample_rows)

    for name, df in raw_tables.items():
        row, null_pct = summarize_table(
            "raw_sample", name, df, key_cols=key_cols_map.get(name), rows_est=raw_rows_est.get(name)
        )
        summary_rows.append(row)

    print("[Null rates by column] RAW (sample-based)")
    for name, df in raw_tables.items():
        _, null_pct = summarize_table(
            "raw_sample", name, df, key_cols=key_cols_map.get(name), rows_est=raw_rows_est.get(name)
        )
        print_nulls("raw_sample", name, null_pct)

    # Stage B: Processed
    processed_tables = {
        "history": history,
        "reviews": reviews,
        "players": players,
        "purchased": purchased,
    }
    for name, df in processed_tables.items():
        row, null_pct = summarize_table("processed", name, df, key_cols=key_cols_map.get(name))
        summary_rows.append(row)

    print("\n[Null rates by column] PROCESSED")
    for name, df in processed_tables.items():
        _, null_pct = summarize_table("processed", name, df, key_cols=key_cols_map.get(name))
        print_nulls("processed", name, null_pct)

    # Stage C: Trimmed
    core_ids = get_core_ids(history, reviews, purchased)
    trimmed_tables = {
        "history": history[history["playerid"].isin(core_ids)],
        "reviews": reviews[reviews["playerid"].isin(core_ids)],
        "players": players[players["playerid"].isin(core_ids)],
        "purchased": purchased[purchased["playerid"].isin(core_ids)],
    }
    for name, df in trimmed_tables.items():
        row, null_pct = summarize_table("trimmed", name, df, key_cols=key_cols_map.get(name))
        summary_rows.append(row)

    print("\n[Null rates by column] TRIMMED")
    for name, df in trimmed_tables.items():
        _, null_pct = summarize_table("trimmed", name, df, key_cols=key_cols_map.get(name))
        print_nulls("trimmed", name, null_pct)

    # Stage D: Feature engineering output
    if feature_matrix is not None:
        fm_row, fm_null_pct = summarize_table("features", "feature_matrix", feature_matrix, key_cols=None)
        fm_row["columns"] = len(feature_matrix.columns)
        summary_rows.append(fm_row)

        print("\n[Null rates by column] FEATURES")
        print_nulls("features", "feature_matrix", fm_null_pct)

    # Stage E: Model input
    if heuristic is not None and feature_matrix is not None:
        model_ids = feature_matrix.index.intersection(heuristic.index)
        model_row = {
            "stage": "model_input",
            "table": "X_raw",
            "rows": len(model_ids),
            "rows_est": len(model_ids),
            "unique_players": len(model_ids),
            "columns": len(feature_matrix.columns),
            "dup_key_pct": round(pd.Index(model_ids).duplicated().mean() * 100, 2),
            "null_pct_max": round(float(feature_matrix.loc[model_ids].isna().mean().max()), 2),
            "null_pct_mean": round(float(feature_matrix.loc[model_ids].isna().mean().mean()), 2),
        }
        summary_rows.append(model_row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n[Summary Table]")
    print(summary_df.to_string(index=False))


def section_2_feature_analysis(feature_matrix, output_dir):
    """Section 2: Feature Analysis"""
    if feature_matrix is None:
        print("⚠ Feature matrix not available. Skipping Section 2.")
        return
    
    print("\n" + "=" * 80)
    print("SECTION 2: FEATURE ANALYSIS")
    print("=" * 80)
    
    print(f"\nActive players: {len(feature_matrix):,}")
    print(f"Features: {len(feature_matrix.columns)}")
    
    # Skewness analysis
    print(f"\nSkewness Analysis:")
    skew_data = []
    for col in feature_matrix.columns:
        val = feature_matrix[col].dropna()
        if len(val) > 0:
            skew = stats.skew(val)
            skew_data.append({'Feature': col, 'Skewness': skew})
    
    skew_df = pd.DataFrame(skew_data).sort_values('Skewness', ascending=False, key=abs)
    heavy_tail = skew_df[skew_df['Skewness'].abs() > 2]
    if not heavy_tail.empty:
        print(f"Heavy-tailed features (|skew| > 2): {len(heavy_tail)}")
        for _, row in heavy_tail.head(5).iterrows():
            print(f"  - {row['Feature']}: {row['Skewness']:.2f}")
    else:
        print("No heavy-tailed features (good)")
    
    # NaN rates
    print(f"\nMissing Data:")
    nan_rates = feature_matrix.isna().sum() / len(feature_matrix) * 100
    nan_rates = nan_rates[nan_rates > 0].sort_values(ascending=False)
    if not nan_rates.empty:
        for col, rate in nan_rates.head(5).items():
            print(f"  - {col}: {rate:.1f}%")
    else:
        print("No missing values")


def section_3_correlations(feature_matrix, output_dir):
    """Section 3: Correlation Analysis"""
    if feature_matrix is None:
        return
    
    print("\n" + "=" * 80)
    print("SECTION 3: CORRELATION ANALYSIS")
    print("=" * 80)
    
    corr = feature_matrix.corr(method='pearson')
    
    # Find high correlations
    high_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                high_pairs.append({
                    'Feat1': corr.columns[i],
                    'Feat2': corr.columns[j],
                    'r': r
                })
    
    print(f"\nHigh-correlation pairs (|r| > 0.7): {len(high_pairs)}")
    if high_pairs:
        high_df = pd.DataFrame(high_pairs).sort_values('r', ascending=False, key=abs)
        for _, row in high_df.head(5).iterrows():
            print(f"  {row['Feat1']} <-> {row['Feat2']}: {row['r']:.3f}")


def section_4_temporal(history, output_dir):
    """Section 4: Temporal Patterns"""
    print("\n" + "=" * 80)
    print("SECTION 4: TEMPORAL PATTERNS")
    print("=" * 80)
    
    if 'hour' not in history.columns:
        history['hour'] = history['date_acquired'].dt.hour
    
    hourly = history['hour'].value_counts().sort_index()
    night_mask = history['hour'] < 6
    night_ratio = night_mask.sum() / len(history) * 100
    
    print(f"\nAchievement unlocks by hour (UTC):")
    print(f"  Total achievements: {len(history):,}")
    print(f"  Night activity (UTC 00:00-05:59): {night_ratio:.1f}%")
    print(f"  Peak hour: {hourly.idxmax()}:00 ({hourly.max():,} achievements)")


def section_5_labels(feature_matrix, heuristic, output_dir):
    """Section 5: Heuristic Labels"""
    if heuristic is None:
        return
    
    print("\n" + "=" * 80)
    print("SECTION 5: HEURISTIC LABELS & BOT DISTRIBUTION")
    print("=" * 80)
    
    n_bot = heuristic['heuristic_bot'].sum()
    n_normal = heuristic['heuristic_normal'].sum()
    n_total = len(heuristic)
    
    print(f"\nLabel Distribution:")
    print(f"  Total: {n_total:,}")
    print(f"  Bot: {n_bot:,} ({n_bot/n_total*100:.2f}%)")
    print(f"  Normal: {n_normal:,} ({n_normal/n_total*100:.2f}%)")
    print(f"  Grey area: {n_total - n_bot - n_normal:,} ({(n_total - n_bot - n_normal)/n_total*100:.2f}%)")
    
    imbalance = n_bot / max(n_normal, 1)
    print(f"\nBot:Normal ratio: 1:{1/imbalance:.1f} (imbalanced)")


def create_visualizations(history, feature_matrix, heuristic, output_dir):
    """Create key visualizations."""
    if not output_dir:
        return
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[*] Creating visualizations in {plots_dir} ...")
    
    # Temporal patterns
    if 'hour' not in history.columns:
        history['hour'] = history['date_acquired'].dt.hour
    
    fig, ax = plt.subplots(figsize=(14, 5))
    history['hour'].value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title('Achievement Unlocks by Hour (UTC)')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(plots_dir / '01_temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 01_temporal_patterns.png")
    
    # Feature distributions
    if feature_matrix is not None:
        representative = [
            'median_unlock_interval_sec', 'max_achievements_per_day',
            'night_activity_ratio', 'total_achievements',
            'total_reviews', 'account_age_days'
        ]
        available = [f for f in representative if f in feature_matrix.columns]
        
        if available:
            n = len(available)
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            
            for idx, feat in enumerate(available):
                data = feature_matrix[feat].dropna()
                axes[idx].hist(data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'{feat}\n(μ={data.mean():.2f})')
                axes[idx].grid(alpha=0.3)
            
            for idx in range(len(available), 6):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(plots_dir / '02_feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 02_feature_distributions.png")
    
    # Heuristic labels
    if heuristic is not None:
        n_bot = heuristic['heuristic_bot'].sum()
        n_normal = heuristic['heuristic_normal'].sum()
        n_grey = len(heuristic) - n_bot - n_normal
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie
        sizes = [n_bot, n_normal, n_grey]
        labels = [f'Bot ({n_bot})', f'Normal ({n_normal})', f'Grey ({n_grey})']
        colors = ['#ff6b6b', '#51cf66', '#cccccc']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Label Distribution')
        
        # Bar
        ax2.bar(['Bot', 'Normal', 'Grey'], sizes, color=colors, edgecolor='black', alpha=0.8)
        ax2.set_ylabel('Count')
        ax2.set_title('Heuristic Labels')
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plots_dir / '03_heuristic_labels.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 03_heuristic_labels.png")


def main():
    parser = argparse.ArgumentParser(description='Standalone EDA for Steam Anomaly Detection')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'),
                       help='Output directory for plots')
    parser.add_argument('--no-plots', action='store_true', help='Skip visualization generation')
    parser.add_argument('--raw-sample', action='store_true',
                        help='Use 100k-row sample for raw profiling (faster)')
    parser.add_argument('--raw-sample-rows', type=int, default=100000,
                        help='Row count for raw profiling sample')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("STEAM ANOMALY DETECTION — STANDALONE EDA")
    print("=" * 80)

    # Step 0: Raw profiling to justify preprocessing
    raw_full_scan = not args.raw_sample
    profile_raw_data(sample_rows=args.raw_sample_rows, full_scan=raw_full_scan)
    
    # Load processed data (post-preprocessing)
    history, players, reviews, purchased = load_data()
    feature_matrix = load_feature_matrix()
    heuristic = load_heuristic_labels()

    # Stage summary (raw -> processed -> trimmed -> features -> model)
    pipeline_stage_summary(
        history,
        players,
        reviews,
        purchased,
        feature_matrix,
        heuristic,
        raw_full_scan=raw_full_scan,
        raw_sample_rows=args.raw_sample_rows,
    )
    
    # Run sections
    section_1_data_overview(history, players, reviews, purchased, args.output_dir)
    section_2_feature_analysis(feature_matrix, args.output_dir)
    section_3_correlations(feature_matrix, args.output_dir)
    section_4_temporal(history, args.output_dir)
    section_5_labels(feature_matrix, heuristic, args.output_dir)
    
    # Visualizations
    if not args.no_plots:
        create_visualizations(history, feature_matrix, heuristic, args.output_dir)
    
    print("\n" + "=" * 80)
    print("✓ EDA Complete")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run full pipeline: python3 main.py")
    print("  2. View dashboard: streamlit run streamlit_app.py")
    print("  3. Check outputs/plots/ for detailed visualizations")


if __name__ == '__main__':
    main()
