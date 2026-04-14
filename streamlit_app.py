"""
Steam Anomaly Detection - Streamlit app for Steam anomaly detection dashboard.
This app allows users to input Steam IDs and view their anomaly risk profiles based on the trained models.
The app loads pre-trained models and baseline data from the "outputs" directory, computes anomaly scores for input profiles, and provides visualizations and explanations of the results.

Run: `streamlit run streamlit_app.py`

"""

import io
import json
import os
import re
import time
from pathlib import Path
import subprocess
import sys
from glob import glob

import altair as alt
import numpy as np
import pandas as pd
import joblib
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models import apply_log_transform

st.set_page_config(page_title="Steam Anomaly Detection Dashboard", layout="wide")
st.title("Steam Anomaly Detection Dashboard")
st.caption("Search by Steam ID: Predict normal/anomaly status and analyze behavioral metrics.")

MODEL_RELATION_TOOLTIPS = {
    "xgb_flag": "1 = XGBoost indicates anomaly (xgb_pct >= 95), 0 = not.",
    "if_flag": "1 = IsolationForest indicates anomaly (if_pct >= 95), 0 = not.",
    "is_anomaly": "Final ensemble prediction: 1 if composite_score >= 85.",
    "combo 1-1": "Both XGB and IF indicate anomaly.",
    "combo 1-0": "Only XGB indicates anomaly, IF does not.",
    "combo 0-1": "Only IF indicates anomaly, XGB does not.",
    "combo 0-0": "Neither XGB nor IF indicates anomaly.",
}

METRIC_EN_LABELS = {
    "median_unlock_interval_sec": "Median Unlock Interval (s)",
    "min_unlock_interval_sec": "Min Unlock Interval (s)",
    "std_unlock_interval_sec": "Unlock Interval Std Dev",
    "cv_unlock_interval": "Unlock Interval CV",  # Coefficient of Variation
    "max_achievements_per_minute": "Max Achievements/Min",
    "max_achievements_per_day": "Max Achievements/Day",
    "night_activity_ratio": "Night Activity Ratio",
    "hour_entropy": "Activity Hour Entropy",
    "activity_density": "Activity Density Index",
    "weekend_ratio": "Weekend Activity Ratio",
    "total_achievements": "Total Achievements",
    "games_with_achievements": "Games w/ Achievements",
    "library_size": "Total Library Size",
    "achievement_game_ratio": "Achievement-to-Game Ratio",
    "top1_game_concentration": "Top 1 Game Concentration",
    "top3_game_concentration": "Top 3 Games Concentration",
    "game_hhi": "Game Concentration Index (HHI)",
    "avg_achievements_per_game": "Avg Achievements/Game",
    "total_reviews": "Total Reviews Submitted",
    "review_unowned_ratio": "Unowned Review Ratio",
    "review_duplication_rate": "Review Duplication Rate",
    "avg_review_length": "Avg Review Length (Chars)",
    "min_review_length": "Min Review Length (Chars)",
    "days_before_first_achievement": "Days to First Achievement",
    "account_age_days": "Account Age (Days)",
    "total_playtime_mins": "Total Playtime (Mins)",
    "playtime_per_achievement": "Playtime per Achievement",
}

# Convert raw metric keys to readable labels for UI tables and charts.
def metric_display_name(metric: str) -> str:
    if metric in METRIC_EN_LABELS:
        return METRIC_EN_LABELS[metric]
    return metric.replace("_", " ").strip().title()

# Convert a raw value into a baseline percentile for quick comparison.
def baseline_percentile(value: float, baseline_series: pd.Series) -> float:
    clean = pd.to_numeric(baseline_series, errors="coerce").dropna()
    if clean.empty or pd.isna(value):
        return np.nan
    return float((clean <= value).mean() * 100)

# Flip the percentile direction when low values are the suspicious ones.
def suspicious_percentile_by_cohen(
    value: float,
    baseline_series: pd.Series,
    cohen_d: float | None,
) -> float:
    clean = pd.to_numeric(baseline_series, errors="coerce").dropna()
    if clean.empty or pd.isna(value):
        return np.nan

    high_is_suspicious = True
    if cohen_d is not None and pd.notna(cohen_d):
        high_is_suspicious = float(cohen_d) >= 0

    if high_is_suspicious:
        return float((clean <= value).mean() * 100)
    return float((clean >= value).mean() * 100)

# Bucket the final score into a simple risk band for the UI.
def score_band(score: float) -> str:
    if pd.isna(score):
        return "no_data"
    if score >= 85:
        return "high"
    if score >= 65:
        return "medium"
    if score >= 40:
        return "neutral"
    return "low"

# Map the score band to a user-facing risk label.
def profile_risk_label(score: float) -> str:
    labels = {
        "no_data": "No Data",
        "high": "High Risk (Critical Suspicion)",
        "medium": "Medium Risk (Suspicious Behavior)",
        "neutral": "Neutral (Needs Monitoring)",
        "low": "Low Risk (Normal User)",
    }
    return labels[score_band(score)]

# Combine model agreement and score band into the online assessment label.
def online_assessment_label(score: float, is_anomaly: float) -> str:
    band = score_band(score)
    if band == "no_data":
        return "Insufficient Data"
    if pd.notna(is_anomaly) and int(is_anomaly) == 1:
        return "Confirmed Anomaly (Blacklisted)"
    if band == "high":
        return "High Risk (Critical Suspicion)"
    if band == "medium":
        return "Medium Risk (Suspicious Behavior)"
    if band == "neutral":
        return "Neutral (Needs Monitoring)"
    return "Low Risk (Normal User)"

# Translate score band into a compact risk bucket for tables.
def risk_bucket_from_score(score: float) -> str:
    labels = {
        "no_data": "No Data",
        "high": "High",
        "medium": "Medium",
        "neutral": "Neutral",
        "low": "Low",
    }
    return labels[score_band(score)]

# Derive confidence from how strongly the two models agree.
def confidence_from_votes(is_anomaly: float, xgb_flag: float, if_flag: float, score: float) -> str:
    if pd.isna(is_anomaly) or pd.isna(score):
        return "Insufficient Data"

    xgb_v = int(xgb_flag) if pd.notna(xgb_flag) else 0
    if_v = int(if_flag) if pd.notna(if_flag) else 0
    agree_votes = xgb_v + if_v

    if int(is_anomaly) == 1:
        if agree_votes == 2 and score >= 85:
            return "High"
        if agree_votes >= 1 and score >= 85:
            return "Medium"
        return "Needs Review"

    if agree_votes == 0 and score < 85:
        return "High"
    return "Needs Review"

# Downgrade confidence when the profile is missing key behavior signals.
def adjust_confidence_by_data_quality(confidence: str, data_quality_note: str) -> str:
    """Downgrade confidence when profile has known missing behavior signals."""
    if not data_quality_note or data_quality_note == "Sufficient data":
        return confidence
    if confidence == "High":
        return "Medium"
    if confidence == "Medium":
        return "Needs Review"
    return confidence

# Format tooltip dictionaries into a table for the model relation explainer.
def build_tooltip_df(tooltips: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": metric_display_name(key), "y_nghia": value} for key, value in tooltips.items()]
    )

# Calculate Cohen's d effect size between two groups represented as pandas Series.
def cohen_d_from_series(group_a: pd.Series, group_b: pd.Series) -> float:
    a = pd.to_numeric(group_a, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(group_b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_denom = (len(a) - 1) + (len(b) - 1)
    if pooled_denom <= 0:
        return np.nan
    pooled_std = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / pooled_denom)
    if pooled_std == 0 or np.isnan(pooled_std):
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled_std)

# Summarize which metrics separate flagged accounts from normal ones.
def build_behavior_reference_table(df: pd.DataFrame) -> pd.DataFrame:
    if "is_anomaly" not in df.columns:
        return pd.DataFrame()

    exclude_metrics = {
        "playerid",
        "is_anomaly",
        "xgb_flag",
        "if_flag",
        "heuristic_bot",
        "heuristic_normal",
        "xgb_proba",
        "xgb_pct",
        "if_pct",
        "composite_score",
        "anomaly_pct",
        "normal_pct",
        "created_year",
    }

    flagged_all = df[df["is_anomaly"] == 1].copy()
    normal_true = df[df["is_anomaly"] == 0].copy()

    numeric_cols = [
        col
        for col in df.columns
        if col not in exclude_metrics and pd.api.types.is_numeric_dtype(df[col])
    ]

    rows = []
    for metric in numeric_cols:
        flagged_col = pd.to_numeric(flagged_all[metric], errors="coerce")
        normal_col = pd.to_numeric(normal_true[metric], errors="coerce")
        flagged_mean = flagged_col.mean()
        normal_mean = normal_col.mean()
        ratio = np.nan
        if pd.notna(normal_mean) and normal_mean != 0:
            ratio = flagged_mean / normal_mean
        rows.append(
            {
                "metric": metric,
                "flagged_mean": flagged_mean,
                "normal_mean": normal_mean,
                "ratio_flagged_vs_normal": ratio,
                "cohen_d_flagged_vs_normal": cohen_d_from_series(flagged_col, normal_col),
            }
        )

    ref = pd.DataFrame(rows)
    if ref.empty:
        return ref

    ref = ref.sort_values("cohen_d_flagged_vs_normal", key=lambda s: s.abs(), ascending=False)
    return ref

# Build anomaly-rate statistics grouped by a categorical column.
def build_categorical_rate_table(
    df: pd.DataFrame,
    category_col: str,
    *,
    flag_col: str = "is_anomaly",
    score_col: str = "composite_score",
) -> pd.DataFrame:
    working = df[[category_col, flag_col, score_col]].copy()
    working[category_col] = working[category_col].fillna("Unknown").astype(str)
    grouped = (
        working.groupby(category_col, as_index=False)
        .agg(
            total_accounts=(flag_col, "count"),
            flagged_accounts=(flag_col, "sum"),
            mean_anomaly_score=(score_col, "mean"),
        )
    )
    grouped["flag_rate_pct"] = (
        grouped["flagged_accounts"] / grouped["total_accounts"].replace(0, np.nan) * 100
    )
    return grouped.sort_values(["flagged_accounts", "total_accounts"], ascending=[False, False])

# Build anomaly-rate statistics for pre-defined numeric bins.
def build_binned_rate_table(
    df: pd.DataFrame,
    value_col: str,
    bins: list[float],
    labels: list[str],
    *,
    flag_col: str = "is_anomaly",
    score_col: str = "composite_score",
) -> pd.DataFrame:
    working = df[[value_col, flag_col, score_col]].copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working["bin"] = pd.cut(
        working[value_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    grouped = (
        working.groupby("bin", as_index=False)
        .agg(
            total_accounts=(flag_col, "count"),
            flagged_accounts=(flag_col, "sum"),
            mean_anomaly_score=(score_col, "mean"),
        )
    )
    grouped["flag_rate_pct"] = (
        grouped["flagged_accounts"] / grouped["total_accounts"].replace(0, np.nan) * 100
    )
    return grouped

# Read CSV data safely when the file is present.
def load_csv(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# Convert bytes into human-readable units.
def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

# List all files under the outputs folder for the explorer panel.
def list_output_files(outputs_dir: str = "outputs") -> list[str]:
    if not os.path.exists(outputs_dir):
        return []
    paths = [p for p in Path(outputs_dir).rglob("*") if p.is_file()]
    return sorted(str(p).replace("\\", "/") for p in paths)

# Read size and modified-time metadata for a file card.
def get_file_meta(path: str) -> dict:
    stat = os.stat(path)
    return {
        "size": stat.st_size,
        "size_human": human_size(stat.st_size),
        "mtime": pd.to_datetime(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
    }

# Extract candidate Steam IDs from free-form text input.
def parse_ids_from_text(text: str) -> list[int]:
    if not text:
        return []
    ids = re.findall(r"\d{8,20}", text)
    return [int(x) for x in ids]

# Parse player IDs from uploaded CSV/TXT files.
def parse_ids_from_uploaded(file_obj) -> list[int]:
    if file_obj is None:
        return []

    name = file_obj.name.lower()
    raw = file_obj.getvalue()

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
        if df.empty:
            return []
        if "playerid" in df.columns:
            series = df["playerid"]
        else:
            series = df.iloc[:, 0]
        series = pd.to_numeric(series, errors="coerce").dropna().astype("int64")
        return series.tolist()

    text = raw.decode("utf-8", errors="ignore")
    return parse_ids_from_text(text)

# Merge ensemble output with feature and player metadata for profile rendering.
def build_profile_df(ensemble: pd.DataFrame, features: pd.DataFrame | None, players: pd.DataFrame | None) -> pd.DataFrame:
    base = ensemble.copy()

    if features is not None and "playerid" in features.columns:
        f = features.copy()
        f["playerid"] = pd.to_numeric(f["playerid"], errors="coerce")
        f = f.dropna(subset=["playerid"])
        f["playerid"] = f["playerid"].astype("int64")
        base = base.merge(f, on="playerid", how="left")

    if players is not None and "playerid" in players.columns:
        p = players.copy()
        p["playerid"] = pd.to_numeric(p["playerid"], errors="coerce")
        p = p.dropna(subset=["playerid"])
        p["playerid"] = p["playerid"].astype("int64")
        base = base.merge(p[["playerid", "country", "created"]], on="playerid", how="left")

    base["anomaly_pct"] = base["composite_score"].clip(0, 100)
    base["normal_pct"] = 100 - base["anomaly_pct"]
    return base

# Execute a subprocess command and return success flag plus combined logs.
def run_cmd(cmd: list[str], title: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            check=False,
        )
        log_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode == 0:
            return True, f"[{title}] Success\n\n{log_text}"
        return False, f"[{title}] Failed with exit code {proc.returncode}\n\n{log_text}"
    except Exception as exc:
        return False, f"[{title}] Error: {exc}"

# Read only the tail portion of a text file for compact log display.
def read_tail_text(path: str, max_chars: int = 12000) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return content[-max_chars:]

def get_preferred_python_executable() -> str:
    """Prefer project venv python to avoid missing-package issues in subprocesses."""
    windows_venv_python = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
    posix_venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python")
    if os.path.exists(windows_venv_python):
        return windows_venv_python
    if os.path.exists(posix_venv_python):
        return posix_venv_python
    return sys.executable

# Convert ID-like values to stable strings for table rendering.
def _stringify_id_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)

# Normalize DataFrame ID columns to string to avoid scientific notation.
def safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        col_name = str(col).lower()
        if "playerid" in col_name or col_name.endswith("_id") or col_name == "id":
            out[col] = out[col].map(_stringify_id_value)
    return out

# Render a plot image when it exists, otherwise show a friendly notice.
def show_plot_if_exists(path: str, caption: str, *, width: int | None = None) -> None:
    if os.path.exists(path):
        if width is not None:
            st.image(path, caption=caption, width=width)
        else:
            st.image(path, caption=caption, width="stretch")
    else:
        st.info(f"File not found: {path}")


def _safe_mtime(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    return os.path.getmtime(path)


def get_model_bundle_cache_key() -> tuple[float | None, ...]:
    return (
        _safe_mtime("outputs/model_memory.pkl"),
        _safe_mtime("outputs/preprocessor.pkl"),
        _safe_mtime("outputs/best_if.pkl"),
        _safe_mtime("outputs/best_xgb.pkl"),
    )


def get_feature_matrix_cache_key() -> float | None:
    return _safe_mtime("outputs/feature_matrix.csv")


@st.cache_resource(show_spinner=False)
# Load the saved models, preprocessor, and calibration memory from disk.
def load_model_bundle(_cache_key: tuple[float | None, ...]) -> dict | None:
    memory_path = "outputs/model_memory.pkl"
    preprocessor_path = "outputs/preprocessor.pkl"
    model_paths = {
        "IsolationForest": "outputs/best_if.pkl",
        "XGBoost": "outputs/best_xgb.pkl",
    }

    required = [memory_path, preprocessor_path, *model_paths.values()]
    if not all(os.path.exists(path) for path in required):
        return None

    memory = joblib.load(memory_path)
    preprocessor = joblib.load(preprocessor_path)
    models = {name: joblib.load(path) for name, path in model_paths.items()}
    return {
        "memory": memory,
        "preprocessor": preprocessor,
        "models": models,
    }


@st.cache_resource(show_spinner=False)
def load_feature_matrix_lookup(_cache_key: float | None) -> pd.DataFrame | None:
    """Load feature_matrix indexed by playerid for exact train-online parity."""
    path = "outputs/feature_matrix.csv"
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, dtype={"playerid": "string"})
    if "playerid" not in df.columns:
        return None

    ids = pd.to_numeric(df["playerid"], errors="coerce")
    df = df[ids.notna()].copy()
    df["playerid"] = ids[ids.notna()].astype("int64")
    return df.drop_duplicates(subset=["playerid"], keep="last").set_index("playerid")

# Fetch sorted baseline scores for a model, sorting raw scores when needed.
def _get_sorted_baseline(memory: dict, key: str) -> np.ndarray:
    sorted_map = memory.get("sorted_raw_scores", {})
    if key in sorted_map:
        return np.asarray(sorted_map[key], dtype=np.float32)
    raw_map = memory.get("raw_scores", {})
    if key not in raw_map:
        raise KeyError(f"Missing baseline score array: {key}")
    return np.sort(np.asarray(raw_map[key], dtype=np.float32))

def _percentile_from_sorted(sorted_scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    if sorted_scores.size == 0:
        return np.full(shape=len(values), fill_value=np.nan, dtype=np.float32)
    idx = np.searchsorted(sorted_scores, values, side="right")
    return (idx / sorted_scores.size * 100).astype(np.float32)

# Parse the purchased-games JSON cell and return the Steam app IDs.
def _parse_library_cell(cell) -> set[int]:
    if pd.isna(cell):
        return set()
    text = str(cell).strip()
    if not text:
        return set()
    try:
        values = json.loads(text.replace("'", '"'))
        library_ids = set()
        for item in values:
            if isinstance(item, dict):
                appid = item.get("appid")
                if pd.notna(appid):
                    library_ids.add(int(appid))
            elif str(item).strip():
                library_ids.add(int(item))
        return library_ids
    except Exception:
        return set()


def _parse_library_stats(cell) -> tuple[set[int], float, int]:
    """Parse app IDs plus aggregate playtime from a purchased-games library cell."""
    if pd.isna(cell):
        return set(), np.nan, 0

    text = str(cell).strip()
    if not text:
        return set(), np.nan, 0

    try:
        values = json.loads(text.replace("'", '"'))
    except Exception:
        return set(), np.nan, 0

    library_ids: set[int] = set()
    total_playtime_mins = 0
    games_with_playtime = 0

    for item in values:
        if isinstance(item, dict):
            appid = item.get("appid")
            if pd.notna(appid):
                library_ids.add(int(appid))

            playtime = item.get("playtime_mins")
            if playtime is not None and pd.notna(playtime):
                playtime_int = int(playtime)
                if playtime_int >= 0:
                    total_playtime_mins += playtime_int
                    games_with_playtime += 1
        elif str(item).strip():
            library_ids.add(int(item))

    if games_with_playtime == 0:
        return library_ids, np.nan, 0
    return library_ids, float(total_playtime_mins), games_with_playtime

# Collect crawled data files into a single lookup bundle for scoring.
def load_crawled_data() -> dict[str, pd.DataFrame | None]:
    crawled = {
        "players": load_csv("data/crawled/players.csv"),
        "purchased": load_csv("data/crawled/purchased_games.csv"),
        "history": load_csv("data/crawled/history.csv"),
        "reviews": load_csv("data/crawled/reviews.csv"),
    }
    return crawled

# Extract normalized player IDs from a crawled table.
def _extract_playerid_set(df: pd.DataFrame | None) -> set[int]:
    if df is None or "playerid" not in df.columns:
        return set()
    ids = pd.to_numeric(df["playerid"], errors="coerce").dropna().astype("int64")
    return set(ids.tolist())

# Build a table-to-id-set index for quick membership checks.
def build_crawled_id_index(crawled: dict[str, pd.DataFrame | None]) -> dict[str, set[int]]:
    return {name: _extract_playerid_set(df) for name, df in crawled.items()}

# Classify whether an ID exists in crawl data and whether it has behavior rows.
def classify_crawled_id_status(pid: int, id_index: dict[str, set[int]]) -> tuple[bool, bool, list[str]]:
    present_tables = [name for name, id_set in id_index.items() if pid in id_set]
    exists_in_crawled = len(present_tables) > 0
    has_behavior_rows = any(name in present_tables for name in ["purchased", "history", "reviews"])
    return exists_in_crawled, has_behavior_rows, present_tables

# Enforce the minimum behavior coverage expected by the training set.
def online_data_quality_gate(
    profile: dict,
    *,
    min_total_achievements: int = 10,
    min_library_size: int = 1,
) -> tuple[bool, str]:
    total_ach = pd.to_numeric(pd.Series([profile.get("total_achievements", np.nan)]), errors="coerce").iloc[0]
    library_size = pd.to_numeric(pd.Series([profile.get("library_size", np.nan)]), errors="coerce").iloc[0]

    if pd.isna(total_ach) or pd.isna(library_size):
        return False, "missing key behavior features"
    if total_ach < min_total_achievements and library_size < min_library_size:
        return False, f"total_achievements<{min_total_achievements} and library_size<{min_library_size}"
    if total_ach < min_total_achievements:
        return False, f"total_achievements<{min_total_achievements}"
    if library_size < min_library_size:
        return False, f"library_size<{min_library_size}"
    return True, "ok"

# Run the ensemble models on a batch of extracted profile features.
def infer_online_profiles_batch(feature_profiles: list[dict], bundle: dict) -> list[dict]:
    if not feature_profiles:
        return []

    feature_columns = bundle["memory"].get("feature_columns", [])
    if not feature_columns:
        raise ValueError("model_memory.pkl does not contain feature_columns")

    X_new = pd.DataFrame(feature_profiles).reindex(columns=feature_columns)
    X_new_log = apply_log_transform(X_new)
    X_new_scaled = bundle["preprocessor"].transform(X_new_log)

    models = bundle["models"]
    memory = bundle["memory"]
    if_flipped = bool(bundle["memory"].get("if_flipped", False))

    if_score = -models["IsolationForest"].score_samples(X_new_scaled)
    xgb_proba = models["XGBoost"].predict_proba(X_new_scaled)[:, 1]

    if_sorted = _get_sorted_baseline(memory, "IsolationForest")
    xgb_sorted = _get_sorted_baseline(memory, "XGBoost")

    if_pct = _percentile_from_sorted(if_sorted, if_score)
    if if_flipped:
        if_pct = 100.0 - if_pct
    xgb_pct = _percentile_from_sorted(xgb_sorted, xgb_proba)

    composite = 0.80 * xgb_pct + 0.20 * if_pct

    xgb_flag = (xgb_pct >= 95).astype(int)
    if_flag = (if_pct >= 95).astype(int)
    is_anomaly = (composite >= 85).astype(int)

    results = []
    for i, profile in enumerate(feature_profiles):
        results.append(
            {
                **profile,
                "if_score": float(if_score[i]),
                "xgb_proba": float(xgb_proba[i]),
                "if_pct": float(if_pct[i]),
                "xgb_pct": float(xgb_pct[i]),
                "composite_score": float(composite[i]),
                "normal_pct": float(100.0 - composite[i]),
                "is_anomaly": int(is_anomaly[i]),
                "xgb_flag": int(xgb_flag[i]),
                "if_flag": int(if_flag[i]),
                "baseline_size": int(bundle["memory"].get("baseline_size", 0)),
                "trained_at": bundle["memory"].get("trained_at"),
                "if_flipped": if_flipped,
            }
        )
    return results


def export_online_metrics_csv(
    feature_profiles: list[dict],
    bundle: dict,
) -> tuple[Path, pd.DataFrame, list[str]]:
    """Export online feature rows using the exact trained feature column order."""
    feature_columns = [str(c) for c in bundle.get("memory", {}).get("feature_columns", [])]
    if not feature_columns:
        raise ValueError("model_memory.pkl does not contain feature_columns")

    export_df = pd.DataFrame(feature_profiles).copy()
    if export_df.empty:
        raise ValueError("No online feature profiles available to export")

    ordered_cols = ["playerid", *feature_columns]
    for col in ordered_cols:
        if col not in export_df.columns:
            export_df[col] = np.nan

    export_df = export_df[ordered_cols]
    export_df["playerid"] = export_df["playerid"].astype("Int64").astype(str)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "online_27_metrics.csv"
    try:
        export_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        base = "online_27_metrics"
        suffix = ".csv"
        fallback_name = pd.Timestamp.now().strftime(f"{base}_%Y%m%d_%H%M%S{suffix}")
        out_path = out_dir / fallback_name
        export_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path, export_df, feature_columns

# Rebuild a temporary profile from crawl tables for one Steam ID.
def compute_temp_profile(
    pid: int,
    crawled: dict[str, pd.DataFrame | None],
    model_memory: dict | None = None,
) -> dict:
    # For IDs present in training feature matrix, use the exact same row as training.
    fm_lookup = load_feature_matrix_lookup(get_feature_matrix_cache_key())
    if fm_lookup is not None and pid in fm_lookup.index:
        row = fm_lookup.loc[pid].to_dict()
        row["playerid"] = int(pid)
        return row

    players_c = crawled.get("players")
    purchased_c = crawled.get("purchased")
    history_c = crawled.get("history")
    reviews_c = crawled.get("reviews")

    row = {"playerid": pid}

    if players_c is not None and "playerid" in players_c.columns:
        p = players_c.copy()
        p["playerid"] = pd.to_numeric(p["playerid"], errors="coerce")
        p = p[p["playerid"] == pid]
        p = p.drop_duplicates(subset=["playerid"], keep="last")
        if not p.empty:
            row["country"] = p.iloc[-1].get("country", np.nan)
            row["created"] = p.iloc[-1].get("created", np.nan)

    library = set()
    total_playtime_mins = np.nan
    if purchased_c is not None and "playerid" in purchased_c.columns:
        pu = purchased_c.copy()
        pu["playerid"] = pd.to_numeric(pu["playerid"], errors="coerce")
        pu = pu[pu["playerid"] == pid]
        pu = pu.drop_duplicates(subset=["playerid"], keep="last")
        if not pu.empty and "library" in pu.columns:
            library, total_playtime_mins, _ = _parse_library_stats(pu.iloc[-1]["library"])
    row["library_size"] = len(library)
    row["total_playtime_mins"] = total_playtime_mins

    h = pd.DataFrame()
    h_timed = pd.DataFrame()
    if history_c is not None and "playerid" in history_c.columns:
        h = history_c.copy()
        h["playerid"] = pd.to_numeric(h["playerid"], errors="coerce")
        h = h[h["playerid"] == pid].copy()

    if not h.empty:
        if "date_accquired" in h.columns:
            if "date_acquired" in h.columns:
                h["date_acquired"] = h["date_acquired"].fillna(h["date_accquired"])
            else:
                h["date_acquired"] = h["date_accquired"]
        h["date_acquired"] = pd.to_datetime(
            h.get("date_acquired"),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )

        if "gameid" not in h.columns and "achievementid" in h.columns:
            h["gameid"] = h["achievementid"].astype(str).str.extract(r"^(\d+)_")[0]
        h["gameid"] = pd.to_numeric(h.get("gameid"), errors="coerce")
        dedupe_cols = [c for c in ["playerid", "achievementid", "date_acquired"] if c in h.columns]
        if dedupe_cols:
            h = h.drop_duplicates(subset=dedupe_cols, keep="last")
        else:
            h = h.drop_duplicates()

        h = h.sort_values("date_acquired")
        h_timed = h.dropna(subset=["date_acquired"]).copy()

        row["total_achievements"] = float(len(h))
        row["games_with_achievements"] = float(h["gameid"].nunique())

        if pd.notna(row.get("total_playtime_mins", np.nan)):
            if row["total_achievements"] > 0:
                row["playtime_per_achievement"] = row["total_playtime_mins"] / row["total_achievements"]
            else:
                row["playtime_per_achievement"] = np.nan

        if row["library_size"] > 0:
            row["achievement_game_ratio"] = row["games_with_achievements"] / row["library_size"]
        else:
            row["achievement_game_ratio"] = np.nan

        h["hour"] = h["date_acquired"].dt.hour
        h["dow"] = h["date_acquired"].dt.dayofweek
        row["night_activity_ratio"] = float((h["hour"] < 6).mean())
        row["weekend_ratio"] = float(h["dow"].isin([5, 6]).mean())

        if not h_timed.empty:
            h_timed["minute"] = h_timed["date_acquired"].dt.floor("min")
            h_timed["day"] = h_timed["date_acquired"].dt.floor("D")
            h_timed["hour"] = h_timed["date_acquired"].dt.hour
            h_timed["dow"] = h_timed["date_acquired"].dt.dayofweek
            row["max_achievements_per_minute"] = float(h_timed.groupby("minute").size().max())
            row["max_achievements_per_day"] = float(h_timed.groupby("day").size().max())

            hour_counts = h_timed["hour"].value_counts(normalize=True)
            probs = hour_counts.values
            row["hour_entropy"] = float(-np.sum(probs * np.log(probs + 1e-12)))

            active_days = h_timed["day"].nunique()
            span_days = (h_timed["date_acquired"].max() - h_timed["date_acquired"].min()).days + 1
            row["activity_density"] = float(active_days / span_days) if span_days > 0 else np.nan

            h_timed["interval_sec"] = h_timed["date_acquired"].diff().dt.total_seconds()
            iv = h_timed["interval_sec"].dropna()
            row["median_unlock_interval_sec"] = float(iv.median()) if not iv.empty else np.nan
            row["min_unlock_interval_sec"] = float(iv.min()) if not iv.empty else np.nan
            row["std_unlock_interval_sec"] = float(iv.std()) if not iv.empty else np.nan
            iv_mean = float(iv.mean()) if not iv.empty else np.nan
            row["cv_unlock_interval"] = (
                float(row["std_unlock_interval_sec"] / iv_mean)
                if not np.isnan(iv_mean) and iv_mean > 0 and not np.isnan(row["std_unlock_interval_sec"])
                else np.nan
            )

        gcounts = h["gameid"].value_counts()
        if not gcounts.empty:
            gsum = gcounts.sum()
            p = gcounts / gsum
            row["top1_game_concentration"] = float(p.iloc[0])
            row["top3_game_concentration"] = float(p.iloc[:3].sum())
            row["game_hhi"] = float((p**2).sum())
            row["avg_achievements_per_game"] = float(gsum / max(1, gcounts.size))

    r = pd.DataFrame()
    if reviews_c is not None and "playerid" in reviews_c.columns:
        r = reviews_c.copy()
        r["playerid"] = pd.to_numeric(r["playerid"], errors="coerce")
        r = r[r["playerid"] == pid].copy()
        if "reviewid" in r.columns:
            r = r.drop_duplicates(subset=["reviewid"], keep="last")
        else:
            r = r.drop_duplicates()

    row["total_reviews"] = float(len(r))
    if not r.empty and "review" in r.columns:
        text_col = r["review"].astype(str)
        row["avg_review_length"] = float(text_col.str.len().mean())
        row["min_review_length"] = float(text_col.str.len().min())
        unique_n = text_col.nunique(dropna=True)
        row["review_duplication_rate"] = float(1 - (unique_n / len(text_col))) if len(text_col) > 0 else np.nan
    else:
        row["review_duplication_rate"] = 0.0
        row["avg_review_length"] = 0.0
        row["min_review_length"] = 0.0

    if not r.empty and "gameid" in r.columns and row["library_size"] > 0:
        rg = pd.to_numeric(r["gameid"], errors="coerce").dropna().astype("int64")
        if not rg.empty:
            row["review_unowned_ratio"] = float((~rg.isin(list(library))).mean())

    created_raw = row.get("created")
    created_dt = pd.to_datetime(created_raw, errors="coerce")
    ref_time = pd.NaT
    if model_memory is not None:
        ref_time = pd.to_datetime(model_memory.get("feature_reference_time"), errors="coerce")
    if pd.isna(ref_time):
        ref_time = pd.Timestamp.now()
    if pd.notna(created_dt):
        row["account_age_days"] = float((ref_time - created_dt).days)
        if not h_timed.empty:
            row["days_before_first_achievement"] = float((h_timed["date_acquired"].min() - created_dt).days)

    return row

# Render the per-player summary, model outputs, and baseline comparison.
def render_profile(
    profile: pd.Series,
    base_size: int,
    behavior_reference: pd.DataFrame | None = None,
    baseline_df: pd.DataFrame | None = None,
) -> None:
    pid = int(profile["playerid"])
    anomaly_pct = float(profile.get("anomaly_pct", np.nan))
    normal_pct = float(profile.get("normal_pct", np.nan))
    risk_label = profile_risk_label(anomaly_pct)

    st.subheader(f"Player {pid}")
    st.write(f"Quick assessment: **{risk_label}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Normal %", f"{normal_pct:.2f}%" if not np.isnan(normal_pct) else "N/A")
    c2.metric("Anomaly %", f"{anomaly_pct:.2f}%" if not np.isnan(anomaly_pct) else "N/A")
    c3.metric("Flag XGB", f"{int(profile.get('xgb_flag', 0))}")
    c4.metric("Flag IF", f"{int(profile.get('if_flag', 0))}")
    st.caption(f"XGB bot probability: {float(profile.get('xgb_proba', np.nan)):.4f}" if not np.isnan(profile.get("xgb_proba", np.nan)) else "N/A")

    model_cols = st.columns(3)
    model_cols[0].metric("XGB percentile", f"{float(profile.get('xgb_pct', np.nan)):.2f}" if pd.notna(profile.get("xgb_pct", np.nan)) else "N/A")
    model_cols[1].metric("IF percentile", f"{float(profile.get('if_pct', np.nan)):.2f}" if pd.notna(profile.get("if_pct", np.nan)) else "N/A")
    model_cols[2].metric("Composite score", f"{float(profile.get('composite_score', np.nan)):.2f}" if pd.notna(profile.get("composite_score", np.nan)) else "N/A")

    if behavior_reference is not None and not behavior_reference.empty:
        top_ref = behavior_reference.head(5)
        top_cols = st.columns(5)
        for idx, (_, row) in enumerate(top_ref.iterrows()):
            metric_name = str(row.get("metric", "N/A"))
            cohend = row.get("cohen_d_flagged_vs_normal", np.nan)
            player_value = pd.to_numeric(profile.get(metric_name, np.nan), errors="coerce")

            rank_text = "N/A"
            top_text = None

            if baseline_df is not None and metric_name in baseline_df.columns and pd.notna(player_value):
                series = pd.to_numeric(baseline_df[metric_name], errors="coerce").dropna()
                n = len(series)
                if n > 0:
                    high_suspicious = pd.isna(cohend) or float(cohend) >= 0
                    if high_suspicious:
                        rank = int((series > float(player_value)).sum() + 1)
                    else:
                        rank = int((series < float(player_value)).sum() + 1)
                    rank_text = f"#{rank}/{n}"
                    top_text = f"Top {rank / n * 100:.2f}%"

            top_cols[idx].metric(metric_display_name(metric_name), rank_text, top_text)

    st.subheader("Detailed Metrics")
    detail_rows = []
    base_info_fields = ["country", "created"]
    for metric in base_info_fields:
        if metric in profile.index:
            detail_rows.append(
                {
                    "metric": metric,
                    "value": profile.get(metric, np.nan),
                }
            )

    for metric in METRIC_EN_LABELS.keys():
        if metric in profile.index:
            detail_rows.append(
                {
                    "metric": metric,
                    "value": profile.get(metric, np.nan),
                }
            )

    for row in detail_rows:
        row["metric"] = metric_display_name(str(row.get("metric", "")))

    if detail_rows:
        st.dataframe(safe_dataframe(pd.DataFrame(detail_rows)), width="stretch", hide_index=True)
    else:
        st.info("No detailed information available.")

ensemble = load_csv("outputs/ensemble_results.csv")
features = load_csv("outputs/feature_matrix.csv")
players = load_csv("data/raw/players.csv")
top50 = load_csv("outputs/top50_flagged_profiles.csv")
loading = load_csv("outputs/pca_loading_matrix.csv")

if ensemble is None:
    st.error("No outputs/ensemble_results.csv found. Please run python3 batch_analysis.py first to generate the ensemble results.")
    st.stop()

ensemble["playerid"] = pd.to_numeric(ensemble["playerid"], errors="coerce")
ensemble = ensemble.dropna(subset=["playerid"]).copy()
ensemble["playerid"] = ensemble["playerid"].astype("int64")

profile_df = build_profile_df(ensemble, features, players)
behavior_reference_df = build_behavior_reference_table(profile_df)

st.header("1) Search by Steam ID")
st.write("Enter a single ID, multiple IDs, or drag and drop a CSV/TXT file containing a list of playerids.")

mode = st.radio(
    "Input Method",
    ["Manual Entry", "Upload File"],
    horizontal=True,
)

target_ids: list[int] = []
if mode == "Manual Entry":
    one_id = st.text_input("Enter 1 Steam ID", placeholder="Example: 76561197960272169")
    multi_text = st.text_area(
        "Or enter multiple IDs (one per line, or separated by commas)",
        placeholder="76561197960272169\n76561197962909864",
    )
    target_ids = parse_ids_from_text(one_id) + parse_ids_from_text(multi_text)
else:
    uploaded = st.file_uploader("Upload CSV or TXT file", type=["csv", "txt"])
    target_ids = parse_ids_from_uploaded(uploaded)

target_ids = sorted(set(target_ids))

if target_ids:
    found = profile_df[profile_df["playerid"].isin(target_ids)].copy()
    missing = [x for x in target_ids if x not in set(found["playerid"].tolist())]

    c1, c2, c3 = st.columns(3)
    c1.metric("Number of IDs entered", len(target_ids))
    c2.metric("Found", len(found))
    c3.metric("No data available", len(missing))

    if missing:
        st.warning("No data available for the following IDs: " + ", ".join(str(x) for x in missing[:20]))

    if not found.empty:
        found = found.sort_values("anomaly_pct", ascending=False)
        st.subheader("Summary of found profiles")
        show_cols = [
            "playerid",
            "normal_pct",
            "anomaly_pct",
            "xgb_proba",
            "xgb_flag",
            "if_flag",
            "is_anomaly",
        ]
        show_cols = [c for c in show_cols if c in found.columns]
        st.dataframe(safe_dataframe(found[show_cols]), width="stretch")

        st.subheader("Detailed Analysis of Each Account")
        for _, row in found.iterrows():
            with st.expander(f"View Details for playerid {int(row['playerid'])}", expanded=(len(found) == 1)):
                render_profile(
                    row,
                    base_size=len(profile_df),
                    behavior_reference=behavior_reference_df,
                    baseline_df=profile_df,
                )

# Section 2: Real-time Steam Crawling (Online Inference)
st.divider()
st.header("2) Real-time Steam Crawl (Online Inference)")
st.write("")

bundle = load_model_bundle(get_model_bundle_cache_key())
if bundle is None:
    st.error("Saved model memory not found. Please run python3 batch_analysis.py first to generate outputs/model_memory.pkl and associated model artifacts.")
    st.stop()

trained_at = bundle["memory"].get("trained_at")
baseline_size = bundle["memory"].get("baseline_size", 0)
st.caption(f"Baseline model: {trained_at} | Number of players in baseline: {baseline_size:,}")

crawl_text = st.text_area(
    "Steam ID to crawl",
    placeholder="76561198405841744\n76561198354838543",
    key="crawl_ids_text",
)
crawl_ids = sorted(set(parse_ids_from_text(crawl_text)))

if "train_proc" not in st.session_state:
    st.session_state["train_proc"] = None
if "train_log_path" not in st.session_state:
    st.session_state["train_log_path"] = ""
if "train_last_rc" not in st.session_state:
    st.session_state["train_last_rc"] = None

train_proc = st.session_state.get("train_proc")
is_running = False
if train_proc is not None:
    try:
        is_running = train_proc.poll() is None
    except Exception:
        is_running = False

if train_proc is not None and not is_running and st.session_state.get("train_last_rc") is None:
    try:
        rc = train_proc.poll()
    except Exception:
        rc = -1
    st.session_state["train_last_rc"] = rc
    st.session_state["train_proc"] = None
    if rc == 0:
        load_model_bundle.clear()
        load_feature_matrix_lookup.clear()

button_label = "Running..." if is_running else "Run Crawl Training"

c1, c2, c3 = st.columns([1, 1, 0.5])
with c1:
    if st.button("Run Steam Crawl", width="stretch"):
        if not crawl_ids:
            st.warning("You need to enter at least 1 Steam ID.")
        else:
            python_exec = get_preferred_python_executable()
            cmd = [
                python_exec,
                "-u",
                "steam_crawling.py",
                "--steam-ids",
                *[str(x) for x in crawl_ids],
                "--data",
                "players,purchased_games,history,reviews",
            ]
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            progress_slot = st.empty()
            progress_slot.progress(35, text="Crawling Steam API...")
            with st.spinner("Crawling Steam API..."):
                proc = subprocess.run(
                    cmd,
                    cwd=os.getcwd(),
                    capture_output=True,
                    text=True,
                    creationflags=creation_flags,
                )

            ok = proc.returncode == 0
            progress_slot.progress(100, text="Crawl finished")
            log_text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()

            if ok:
                st.success("Crawl completed successfully. Click the scoring button to analyze with the trained model.")
            else:
                st.error("Crawl failed. Please check the log section below.")
            with st.expander("Crawl Log (after run)", expanded=False):
                st.text(log_text or "(empty log)")

with c2:

    if st.button("Score Profiles with Trained Model", width="stretch"):
        if not crawl_ids:
            st.warning("You need to enter Steam ID before comparing.")
        else:
            crawled = load_crawled_data()
            missing_files = [name for name, df in crawled.items() if df is None]
            if len(missing_files) == 4:
                st.error("No crawled data found in data/crawled. Please run the crawl first.")
            else:
                if missing_files:
                    st.info("Missing some crawl files: " + ", ".join(missing_files) + ". Analyzing with available data.")

                rows = []

                feature_profiles = []
                valid_ids = []
                id_index = build_crawled_id_index(crawled)
                unknown_ids = []
                low_data_ids = []
                eligible_ids = []

                for pid in crawl_ids:
                    exists_in_crawled, has_behavior_rows, _ = classify_crawled_id_status(pid, id_index)
                    if not exists_in_crawled:
                        unknown_ids.append(pid)
                        continue
                    if not has_behavior_rows:
                        low_data_ids.append(pid)
                        continue
                    eligible_ids.append(pid)

                if unknown_ids:
                    st.warning(
                        "Unknown Steam IDs (not found in any crawled table): "
                        + ", ".join(str(x) for x in unknown_ids[:20])
                    )
                if low_data_ids:
                    st.info(
                        "IDs found in crawled tables but without behavioral rows (history/purchased/reviews), skipped: "
                        + ", ".join(str(x) for x in low_data_ids[:20])
                    )

                data_quality_warnings = {}  # pid -> reason mapping
                for pid in eligible_ids:
                    try:
                        profile = compute_temp_profile(pid, crawled, bundle.get("memory"))
                        is_quality_ok, reason = online_data_quality_gate(profile)
                        if not is_quality_ok:
                            data_quality_warnings[str(pid)] = reason
                        feature_profiles.append(profile)
                        valid_ids.append(pid)
                    except Exception as exc:
                        st.error(f"Failed to compute features for player {pid}: {exc}")

                if data_quality_warnings:
                    warning_text = "⚠️ The IDs below have missing behavior data (scores may be inaccurate):\n"
                    for pid, reason in list(data_quality_warnings.items())[:10]:
                        warning_text += f"  • {pid}: {reason}\n"
                    if len(data_quality_warnings) > 10:
                        warning_text += f"  ... and {len(data_quality_warnings) - 10} more IDs"
                    st.warning(warning_text)

                analyzed_batch = []
                if not feature_profiles:
                    st.warning("No eligible profiles available for inference.")
                else:
                    try:
                        online_metric_path, online_metric_df, online_feature_cols = export_online_metrics_csv(
                            feature_profiles,
                            bundle,
                        )
                        st.caption(
                            f"Saved online metrics export: {online_metric_path} "
                            f"({len(online_metric_df):,} IDs x {len(online_feature_cols)} metrics)"
                        )
                        st.download_button(
                            "Download online 27 metrics (CSV)",
                            data=online_metric_df.to_csv(index=False).encode("utf-8-sig"),
                            file_name=online_metric_path.name,
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.warning(f"Could not export online 27-metric files: {exc}")

                    try:
                        analyzed_batch = infer_online_profiles_batch(feature_profiles, bundle)
                    except Exception as exc:
                        st.error(f"Batch inference failed: {exc}")

                for pid, analyzed in zip(valid_ids, analyzed_batch):
                    proxy_anomaly = analyzed.get("composite_score", np.nan)
                    label = online_assessment_label(
                        proxy_anomaly,
                        analyzed.get("is_anomaly", np.nan),
                    )

                    data_quality_note = data_quality_warnings.get(str(pid), "Sufficient data")
                    confidence = confidence_from_votes(
                        analyzed.get("is_anomaly", np.nan),
                        analyzed.get("xgb_flag", np.nan),
                        analyzed.get("if_flag", np.nan),
                        analyzed.get("composite_score", np.nan),
                    )
                    confidence = adjust_confidence_by_data_quality(confidence, data_quality_note)
                    rows.append(
                        {
                            "playerid": pid,
                            "normal_pct": analyzed.get("normal_pct", np.nan),
                            "anomaly_pct": analyzed.get("composite_score", np.nan),
                            "xgb_pct": analyzed.get("xgb_pct", np.nan),
                            "if_pct": analyzed.get("if_pct", np.nan),
                            "xgb_flag": analyzed.get("xgb_flag", np.nan),
                            "if_flag": analyzed.get("if_flag", np.nan),
                            "is_anomaly": analyzed.get("is_anomaly", np.nan),
                            "assessment": label,
                            "baseline_players": analyzed.get("baseline_size", np.nan),
                            "trained_at": analyzed.get("trained_at", np.nan),
                            "risk_bucket": risk_bucket_from_score(analyzed.get("composite_score", np.nan)),
                            "confidence": confidence,
                            "data_quality": data_quality_note,
                        }
                    )

                if rows:
                    result_df = pd.DataFrame(rows).sort_values(["is_anomaly", "anomaly_pct"], ascending=[False, False])
                    if "playerid" in result_df.columns:
                        result_df["playerid"] = result_df["playerid"].astype(str)
                    st.subheader("Online Inference Results")

                    quick_cols = [
                        "playerid",
                        "assessment",
                        "risk_bucket",
                        "confidence",
                        "anomaly_pct",
                        "normal_pct",
                        "xgb_pct",
                        "if_pct",
                        "is_anomaly",
                    ]
                    quick_cols = [c for c in quick_cols if c in result_df.columns]
                    quick_view = result_df[quick_cols].rename(
                        columns={
                            "assessment": "Assessment",
                            "risk_bucket": "Risk Bucket",
                            "confidence": "Confidence",
                            "anomaly_pct": "Anomaly Score",
                            "normal_pct": "Normal Score",
                            "xgb_pct": "XGB Percentile",
                            "if_pct": "IF Percentile",
                        }
                    )
                    if "playerid" in quick_view.columns:
                        quick_view["playerid"] = quick_view["playerid"].astype(str)
                    st.dataframe(safe_dataframe(quick_view), width="stretch", hide_index=True)

                    detail_df = pd.DataFrame(analyzed_batch)
                    if not detail_df.empty:
                        detail_df = detail_df.copy()
                        detail_df["playerid"] = [str(pid) for pid in valid_ids[: len(detail_df)]]
                        feature_cols = [str(c) for c in bundle["memory"].get("feature_columns", [])]
                        if not feature_cols:
                            feature_cols = [m for m in METRIC_EN_LABELS.keys() if m in detail_df.columns]

                        for metric in feature_cols:
                            if metric not in detail_df.columns:
                                detail_df[metric] = np.nan

                        cohen_map = {}
                        if behavior_reference_df is not None and not behavior_reference_df.empty:
                            cohen_map = behavior_reference_df.set_index("metric")["cohen_d_flagged_vs_normal"].to_dict()

                        st.write("Comparison with Baseline (Showing Top 12 Most Deviant Metrics per Player)")
                        rendered_count = 0
                        for _, prow in detail_df.iterrows():
                            pid = str(prow.get("playerid", "")) or "Unknown"
                            comp_rows = []

                            for metric in feature_cols:
                                if metric not in profile_df.columns:
                                    continue

                                player_value = pd.to_numeric(pd.Series([prow.get(metric, np.nan)]), errors="coerce").iloc[0]
                                baseline_series = pd.to_numeric(profile_df[metric], errors="coerce").dropna()
                                if baseline_series.empty or pd.isna(player_value):
                                    continue

                                cohen_val = cohen_map.get(metric, np.nan)
                                susp_pct = suspicious_percentile_by_cohen(
                                    float(player_value),
                                    baseline_series,
                                    cohen_val,
                                )

                                comp_rows.append(
                                    {
                                        "Metric": metric_display_name(metric),
                                        "Value": float(player_value),
                                        "Baseline Median": float(baseline_series.median()),
                                        "Baseline IQR": float(
                                            baseline_series.quantile(0.75) - baseline_series.quantile(0.25)
                                        ),
                                        "Suspicious Percentile": susp_pct,
                                    }
                                )

                            if not comp_rows:
                                continue

                            rendered_count += 1
                            cmp_df = pd.DataFrame(comp_rows).sort_values("Suspicious Percentile", ascending=False)
                            top_df = cmp_df.head(12)
                            with st.expander(f"Player {pid} - top deviant metrics", expanded=(len(detail_df) == 1)):
                                prow_score = pd.to_numeric(prow.get("composite_score", np.nan), errors="coerce")
                                base_conf = confidence_from_votes(
                                    prow.get("is_anomaly", np.nan),
                                    prow.get("xgb_flag", np.nan),
                                    prow.get("if_flag", np.nan),
                                    prow_score,
                                )
                                conf = adjust_confidence_by_data_quality(
                                    base_conf,
                                    data_quality_warnings.get(str(pid), "Sufficient data"),
                                )
                                st.write(
                                    f"Assessment: {online_assessment_label(prow_score, prow.get('is_anomaly', np.nan))} | "
                                    f"Risk Level: {risk_bucket_from_score(prow_score)} | "
                                    f"Confidence: {conf}"
                                )
                                reason_text = ", ".join(top_df["Metric"].head(3).tolist())
                                if reason_text:
                                    st.caption(f"Main Reasons: {reason_text}")

                                forensic_chart = (
                                    alt.Chart(top_df)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X("Suspicious Percentile:Q", title="Suspicious Percentile"),
                                        y=alt.Y("Metric:N", sort="-x", title="Metric"),
                                        tooltip=[
                                            alt.Tooltip("Metric:N"),
                                            alt.Tooltip("Suspicious Percentile:Q", format=".2f"),
                                            alt.Tooltip("Value:Q", format=".4f"),
                                            alt.Tooltip("Baseline Median:Q", format=".4f"),
                                        ],
                                    )
                                    .properties(height=320)
                                )
                                st.altair_chart(forensic_chart, width="stretch")
                                st.dataframe(safe_dataframe(top_df), width="stretch", hide_index=True)

                        if rendered_count == 0:
                            st.info("No deviant metrics found.")

with c3:
    start_clicked = st.button(button_label, width="content", disabled=is_running, key="train_bg_btn")

if start_clicked:
    python_exec = get_preferred_python_executable()
    train_cmd = [python_exec, "batch_analysis.py", "--force-run"]

    os.makedirs("outputs/logs", exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("outputs", "logs", f"batch_train_{stamp}.log")

    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            train_cmd,
            cwd=os.getcwd(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creation_flags,
        )

    st.session_state["train_proc"] = proc
    st.session_state["train_log_path"] = log_path
    st.session_state["train_last_rc"] = None
    st.rerun()

# Section 3: Data Overview and Insights
st.divider()
st.header("3) Data Overview and Insights")
summary_c1, summary_c2, summary_c3, summary_c4, summary_c5 = st.columns(5)
summary_c1.metric("Total Accounts in Baseline", f"{len(profile_df):,}")
summary_c2.metric("Accounts Flagged", f"{int((profile_df['is_anomaly'] == 1).sum()):,}")
summary_c3.metric("Normal", f"{int((profile_df['is_anomaly'] == 0).sum()):,}")
summary_c4.metric(
    "Flag Rate",
    f"{((profile_df['is_anomaly'] == 1).mean() * 100):.2f}%" if len(profile_df) else "N/A",
)
summary_c5.metric("Average Anomaly Score", f"{profile_df['anomaly_pct'].mean():.2f}")

tab_overview_1, tab_overview_2, tab_overview_3, tab_overview_4, tab_overview_5 = st.tabs(
    ["Score Distribution", "Country", "Model Relationships", "Notable Behaviors", "Other"]
)

with tab_overview_1:
    col_a, col_b = st.columns([6, 4]) # Adjust the ratio so the chart is wider
    
    with col_a:
        # --- 1. Composite Score Histogram ---
        st.subheader("Composite Score Distribution (Relative Percentile)")
        st.caption("The Composite Score is a weighted percentile rank (0-100), not a raw probability.")
        
        score_bins = np.arange(0, 105, 5)
        binned = pd.cut(
            profile_df["anomaly_pct"],
            bins=score_bins,
            include_lowest=True,
            right=True,
        )
        hist = (
            binned.value_counts(sort=False)
            .rename_axis("bin")
            .reset_index(name="count")
        )
        hist["bin"] = hist["bin"].astype(str)
        hist["pct"] = hist["count"] / max(len(profile_df), 1) * 100

        st.bar_chart(hist.set_index("bin")["count"])

        binned_counts = binned.value_counts(sort=False)
        low_0_5 = int(binned_counts.iloc[0]) if len(binned_counts) > 0 else 0
        st.caption(
            f"Note: This reflects a uniform percentile distribution. Bin (0,5] contains {low_0_5} accounts."
        )

        with st.expander("View Detailed Histogram Bin Counts"):
            st.dataframe(
                safe_dataframe(hist.rename(columns={"bin": "Score Range", "count": "Frequency", "pct": "Percentage (%)"})), 
                use_container_width=True
            )

        st.divider()

        # --- 2. XGBoost Probability Histogram ---
        st.subheader("XGBoost Probability Distribution (xgb_proba)")
        if "xgb_proba" in profile_df.columns:
            proba = pd.to_numeric(profile_df["xgb_proba"], errors="coerce").dropna().clip(0, 1)
            proba_bins = np.linspace(0.0, 1.0, 21)
            proba_binned = pd.cut(
                proba,
                bins=proba_bins,
                include_lowest=True,
                right=True,
            )
            proba_hist = (
                proba_binned.value_counts(sort=False)
                .rename_axis("bin")
                .reset_index(name="count")
            )
            proba_hist["bin"] = proba_hist["bin"].astype(str)
            proba_hist["pct"] = proba_hist["count"] / max(len(proba), 1) * 100

            st.bar_chart(proba_hist.set_index("bin")["count"])

            proba_counts = proba_binned.value_counts(sort=False)
            bin_0_05 = int(proba_counts.iloc[0]) if len(proba_counts) > 0 else 0
            st.caption(
                f"Interpretation: XGBoost probabilities often skew heavily towards 0 for healthy datasets. "
                f"The [0.0, 0.05] bin currently holds {bin_0_05} accounts."
            )

            with st.expander("View Detailed XGBoost Probability Table"):
                st.dataframe(
                    safe_dataframe(proba_hist.rename(columns={"bin": "Proba Range", "count": "Frequency", "pct": "Percentage (%)"})),
                    use_container_width=True
                )
        else:
            st.info("XGBoost probability data (xgb_proba) is not available in current session.")

    with col_b:
        # --- 3. Top Suspects Table ---
        st.subheader("Most Suspicious Accounts")
        st.write("Top 30 accounts ranked by Anomaly Score.")
        
        top_cols = ["playerid", "anomaly_pct", "xgb_proba", "is_anomaly", "xgb_flag", "if_flag"]
        top_cols = [c for c in top_cols if c in profile_df.columns]
        
        # Mapping to user-friendly names for the table
        display_map = {
            "playerid": "Steam ID",
            "anomaly_pct": "Risk Score (%)",
            "xgb_proba": "XGB Proba",
            "is_anomaly": "Is Anomaly",
            "xgb_flag": "XGB Flag",
            "if_flag": "IF Flag"
        }
        
        sorted_top = profile_df.sort_values("anomaly_pct", ascending=False)[top_cols].head(30)

        if "playerid" in sorted_top.columns:
            sorted_top["playerid"] = sorted_top["playerid"].astype(str)

        st.dataframe(
            safe_dataframe(sorted_top.rename(columns=display_map)),
            use_container_width=True,
            hide_index=True
        )

with tab_overview_2:
    st.subheader("Geographic Anomaly Distribution")
    country_df = profile_df.copy()
    
    if "country" in country_df.columns:
        country_df["country"] = country_df["country"].fillna("Unknown")
        
        # Aggregate statistics by country
        country_stats = (
            country_df.groupby("country", as_index=False)
            .agg(
                total_accounts=("playerid", "count"),
                flagged_accounts=("is_anomaly", "sum"),
                mean_anomaly_pct=("anomaly_pct", "mean"),
            )
        )
        country_stats["flag_rate_pct"] = (
            country_stats["flagged_accounts"] / country_stats["total_accounts"] * 100
        )

        # Overview Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Unique Countries", f"{country_df['country'].nunique(dropna=True):,}")
        m2.metric("Unknown Location Rate", f"{(country_df['country'].eq('Unknown').mean() * 100):.2f}%")
        
        top_country_name = country_stats.sort_values("flagged_accounts", ascending=False).iloc[0]["country"]
        m3.metric("Highest Anomaly Count", top_country_name)

        # Prepare top 15 charts
        top_flagged_count = country_stats.sort_values("flagged_accounts", ascending=False).head(15)
        # Filter countries with at least 10 accounts to avoid 100% rate bias on single-account countries
        top_flagged_rate = (
            country_stats[country_stats["total_accounts"] >= 10]
            .sort_values("flag_rate_pct", ascending=False)
            .head(15)
        )

        c_left, c_right = st.columns(2)
        
        with c_left:
            st.write("### Top 15 Countries by Anomaly Volume")
            # Rename columns for automatic UI labels
            count_chart = top_flagged_count.rename(columns={"country": "Country", "flagged_accounts": "Anomaly Count"})
            st.bar_chart(count_chart.set_index("Country")["Anomaly Count"])
            
        with c_right:
            st.write("### Top 15 Countries by Anomaly Rate")
            st.caption("Minimum 10 accounts per country for statistical significance.")
            if not top_flagged_rate.empty:
                rate_chart = top_flagged_rate.rename(columns={"country": "Country", "flag_rate_pct": "Anomaly Rate (%)"})
                st.bar_chart(rate_chart.set_index("Country")["Anomaly Rate (%)"])
            else:
                st.info("Insufficient data for rate-based ranking (min. 10 accounts threshold).")

        with st.expander("Detailed Geographic Statistics Table"):
            # Rename for display
            display_stats = country_stats.sort_values("flagged_accounts", ascending=False).rename(columns={
                "country": "Country",
                "total_accounts": "Total Users",
                "flagged_accounts": "Flagged Users",
                "flag_rate_pct": "Flag Rate (%)",
                "mean_anomaly_pct": "Avg Anomaly Pct"
            })
            st.dataframe(
                safe_dataframe(display_stats),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Country data is not available in the current dataset.")

with tab_overview_3:
    st.subheader("Model Correlation & Ensemble Consensus")
    st.caption(
        "This tab analyzes the relationship between XGBoost/Isolation Forest suspicions "
        "and the final Ensemble anomaly decision."
    )
    
    with st.expander("Tooltip: Metric Definitions"):
        st.dataframe(
            safe_dataframe(build_tooltip_df(MODEL_RELATION_TOOLTIPS)),
            use_container_width=True,
            hide_index=True,
        )
        
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("Individual Model Flag Distribution")
        for flag_col, label in [("xgb_flag", "XGBoost"), ("if_flag", "Isolation Forest")]:
            if flag_col in profile_df.columns:
                flag_counts = (
                    profile_df[flag_col]
                    .value_counts()
                    .sort_index()
                    .rename_axis(flag_col)
                    .reset_index(name="count")
                )
                flag_counts[flag_col] = flag_counts[flag_col].astype(str)
                
                # Rename for UI
                chart_data = flag_counts.rename(columns={flag_col: f"{label} Flag", "count": "Account Count"})
                st.write(f"**{label} Raw Flags**")
                st.bar_chart(chart_data.set_index(f"{label} Flag")["Account Count"])
                st.caption(f"{label}: 1 = Suspicious, 0 = Normal.")

    with c_right:
        st.subheader("Ensemble Consensus Rate")
        for flag_col, label in [("xgb_flag", "XGBoost"), ("if_flag", "Isolation Forest")]:
            if flag_col in profile_df.columns:
                flag_by_model = (
                    profile_df.groupby(flag_col, as_index=False)
                    .agg(total=("playerid", "count"), flagged=("is_anomaly", "sum"))
                )
                flag_by_model["flag_rate_pct"] = (
                    flag_by_model["flagged"] / flag_by_model["total"] * 100
                )
                flag_by_model[flag_col] = flag_by_model[flag_col].astype(str)
                
                # Rename for UI
                chart_data = flag_by_model.rename(columns={flag_col: f"{label} Status", "flag_rate_pct": "Ensemble Flag Rate (%)"})
                st.write(f"**Final Flag Rate by {label} Status**")
                st.bar_chart(chart_data.set_index(f"{label} Status")["Ensemble Flag Rate (%)"])
                st.caption(f"Percentage of accounts confirmed as anomaly by Ensemble when {label} flag is 0 vs 1.")

    if {"xgb_flag", "if_flag", "is_anomaly"}.issubset(profile_df.columns):
        st.divider()
        st.subheader("Consensus Matrix: XGBoost vs. Isolation Forest")

        combo_df = profile_df[["xgb_flag", "if_flag", "is_anomaly"]].copy()
        combo_df["combo"] = (
            combo_df["xgb_flag"].astype(int).astype(str)
            + "-"
            + combo_df["if_flag"].astype(int).astype(str)
        )

        combo_stats = (
            combo_df.groupby("combo", as_index=False)
            .agg(
                total_accounts=("is_anomaly", "count"),
                ensemble_flagged=("is_anomaly", "sum"),
            )
        )
        combo_stats["ensemble_flag_rate_pct"] = (
            combo_stats["ensemble_flagged"] / combo_stats["total_accounts"] * 100
        )

        combo_order = ["0-0", "0-1", "1-0", "1-1"]
        combo_stats["combo"] = pd.Categorical(
            combo_stats["combo"], categories=combo_order, ordered=True
        )
        combo_stats = combo_stats.sort_values("combo")

        cm1, cm2 = st.columns(2)
        with cm1:
            st.write("### Account Counts by Combination")
            st.bar_chart(combo_stats.set_index("combo")["total_accounts"])
        with cm2:
            st.write("### Ensemble Flag Rate by Combination")
            st.bar_chart(combo_stats.set_index("combo")["ensemble_flag_rate_pct"])

        st.info(
            "**How to read combinations (XGB-IF):**\n"
            "* **1-1**: Both models agree (Highest confidence).\n"
            "* **1-0**: Only XGBoost suspects.\n"
            "* **0-1**: Only Isolation Forest suspects.\n"
            "* **0-0**: Both models agree the user is Normal."
        )

        with st.expander("View Consensus Matrix Detail Table"):
            st.dataframe(
                safe_dataframe(combo_stats.rename(columns={
                    "combo": "Combination (XGB-IF)",
                    "total_accounts": "Total Accounts",
                    "ensemble_flagged": "Ensemble Flagged",
                    "ensemble_flag_rate_pct": "Flag Rate (%)"
                })), 
                use_container_width=True, 
                hide_index=True
            )

with tab_overview_4:
    st.subheader("Behavioral Insights & Differentiation")
    st.caption("Direct comparison between Flagged accounts (Anomaly) and Normal accounts.")

    flagged_all = profile_df[profile_df["is_anomaly"] == 1].copy()
    normal_true = profile_df[profile_df["is_anomaly"] == 0].copy()

    # Compute interquartile range for robust spread comparison.
    def _iqr(series: pd.Series) -> float:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return np.nan
        q1, q3 = np.nanpercentile(clean, [25, 75])
        return float(q3 - q1)

    # Compute Cohen's d to quantify separation between flagged and normal groups.
    def _cohen_d(group_a: pd.Series, group_b: pd.Series) -> float:
        a = pd.to_numeric(group_a, errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(group_b, errors="coerce").dropna().to_numpy(dtype=float)
        if len(a) < 2 or len(b) < 2:
            return np.nan
        var_a = np.var(a, ddof=1)
        var_b = np.var(b, ddof=1)
        pooled_denom = (len(a) - 1) + (len(b) - 1)
        if pooled_denom <= 0:
            return np.nan
        pooled_std = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / pooled_denom)
        if pooled_std == 0 or np.isnan(pooled_std):
            return np.nan
        return float((np.mean(a) - np.mean(b)) / pooled_std)

    # Metrics to exclude from behavioral analysis
    exclude_metrics = {
        "playerid", "is_anomaly", "xgb_flag", "if_flag", 
        "heuristic_bot", "heuristic_normal", "xgb_proba", 
        "xgb_pct", "if_pct", "composite_score", "anomaly_pct", 
        "normal_pct", "created_year",
    }

    numeric_cols = [
        col for col in profile_df.columns
        if col not in exclude_metrics and pd.api.types.is_numeric_dtype(profile_df[col])
    ]
    focus_metrics = sorted(numeric_cols)
    rows = []
    
    for m in focus_metrics:
        if m in profile_df.columns:
            flagged_col = pd.to_numeric(flagged_all[m], errors="coerce")
            normal_col = pd.to_numeric(normal_true[m], errors="coerce")
            rows.append({
                "metric": m,
                "flagged_mean": flagged_col.mean(),
                "normal_mean": normal_col.mean(),
                "flagged_median": flagged_col.median(),
                "normal_median": normal_col.median(),
                "flagged_iqr": _iqr(flagged_col),
                "normal_iqr": _iqr(normal_col),
                "cohen_d_flagged_vs_normal": _cohen_d(flagged_col, normal_col),
            })
            
    behavior_df = pd.DataFrame(rows)
    
    if not behavior_df.empty:
        # Calculate ratio and sort by absolute effect size (Cohen's d)
        behavior_df["ratio_flagged_vs_normal"] = behavior_df["flagged_mean"] / behavior_df["normal_mean"].replace(0, np.nan)
        
        sorted_df = behavior_df.sort_values(
            "cohen_d_flagged_vs_normal", key=lambda s: s.abs(), ascending=False
        )[[
            "metric",
            "ratio_flagged_vs_normal",
            "cohen_d_flagged_vs_normal",
        ]]
        
        # Map to display names
        sorted_df["metric_display"] = sorted_df["metric"].map(METRIC_EN_LABELS).fillna(sorted_df["metric"])

        behavior_view_mode = st.radio(
            "View Mode",
            ["Top 10 Metrics", "All Metrics"],
            horizontal=True,
        )

        st.caption(f"Ranking based on {len(sorted_df)} valid numerical metrics.")

        if behavior_view_mode == "Top 10 Metrics":
            st.write("### Top 10 Features by Separation Strength (|Cohen's d|)")
            display_df = sorted_df.head(10).drop(columns=["metric"]).rename(columns={
                "metric_display": "Feature",
                "ratio_flagged_vs_normal": "Ratio (Bot/Normal)",
                "cohen_d_flagged_vs_normal": "Effect Size (Cohen's d)"
            })
            st.dataframe(safe_dataframe(display_df), use_container_width=True, hide_index=True)
        else:
            st.write("### All Behavioral Metrics (Sorted by Impact)")
            display_df = sorted_df.drop(columns=["metric"]).rename(columns={
                "metric_display": "Feature",
                "ratio_flagged_vs_normal": "Ratio (Bot/Normal)",
                "cohen_d_flagged_vs_normal": "Effect Size (Cohen's d)"
            })
            st.dataframe(safe_dataframe(display_df), use_container_width=True, hide_index=True)

        # Visualization 1: Ratio Chart
        chart_df = behavior_df[["metric", "ratio_flagged_vs_normal"]].dropna(how="all").copy()
        chart_df["metric"] = chart_df["metric"].map(METRIC_EN_LABELS).fillna(chart_df["metric"])
        
        st.write("### Mean Ratio (Flagged vs. Normal)")
        st.caption("Ratio > 1 means the metric is higher in flagged accounts.")
        st.bar_chart(chart_df.set_index("metric")["ratio_flagged_vs_normal"])

        # Visualization 2: Cohen's d Chart
        d_chart = behavior_df[["metric", "cohen_d_flagged_vs_normal"]].dropna(how="all").copy()
        d_chart["metric"] = d_chart["metric"].map(METRIC_EN_LABELS).fillna(d_chart["metric"])
        
        st.write("### Statistical Effect Size (Cohen's d)")
        st.caption("Measures how many standard deviations separate the two groups.")
        st.bar_chart(d_chart.set_index("metric")["cohen_d_flagged_vs_normal"])

        st.info(
            "**Quick Guide:** |Cohen's d| ≈ 0.2 (Small), ≈ 0.5 (Medium), ≥ 0.8 (Large effect). "
            "High Effect Size indicates a strong predictor for the anomaly model."
        )
    else:
        st.warning("Insufficient behavioral data to display comparison charts.")

with tab_overview_5:
    st.subheader("Demographic & Behavioral Correlation")

    raw_profile = profile_df.copy()
    if "created" in raw_profile.columns:
        raw_profile["created_dt"] = pd.to_datetime(raw_profile["created"], errors="coerce")
        raw_profile["created_year"] = raw_profile["created_dt"].dt.year

    c1, c2 = st.columns(2)
    
    with c1:
        # --- 1. Account Age ---
        if "account_age_days" in raw_profile.columns:
            age_bins = [0, 365, 730, 1460, 2190, 3650, np.inf]
            age_labels = ["< 1yr", "1-2yrs", "2-4yrs", "4-6yrs", "6-10yrs", "> 10yrs"]
            age_stats = build_binned_rate_table(raw_profile, "account_age_days", age_bins, age_labels)
            
            # Rename for professional UI
            age_stats_en = age_stats.rename(columns={"bin": "Age Group", "flag_rate_pct": "Anomaly Rate (%)"})
            
            st.write("### Anomaly Rate by Account Age")
            st.bar_chart(age_stats_en.set_index("Age Group")["Anomaly Rate (%)"])
            
            with st.expander("View Detailed Age Statistics"):
                st.dataframe(safe_dataframe(age_stats_en), use_container_width=True)

        # --- 2. Library Size ---
        if "library_size" in raw_profile.columns:
            lib_bins = [0, 10, 25, 50, 100, 250, 500, np.inf]
            lib_labels = ["0-10", "11-25", "26-50", "51-100", "101-250", "251-500", "> 500"]
            lib_stats = build_binned_rate_table(raw_profile, "library_size", lib_bins, lib_labels)
            
            lib_stats_en = lib_stats.rename(columns={"bin": "Library Size", "flag_rate_pct": "Anomaly Rate (%)"})
            
            st.write("### Anomaly Rate by Library Size")
            st.bar_chart(lib_stats_en.set_index("Library Size")["Anomaly Rate (%)"])
            
            with st.expander("View Detailed Library Size Statistics"):
                st.dataframe(safe_dataframe(lib_stats_en), use_container_width=True)

    with c2:
        # --- 3. Total Reviews ---
        if "total_reviews" in raw_profile.columns:
            review_bins = [-0.1, 0, 1, 3, 5, 10, 20, np.inf]
            review_labels = ["0", "1", "2-3", "4-5", "6-10", "11-20", "> 20"]
            review_stats = build_binned_rate_table(raw_profile, "total_reviews", review_bins, review_labels)
            
            review_stats_en = review_stats.rename(columns={"bin": "Total Reviews", "flag_rate_pct": "Anomaly Rate (%)"})
            
            st.write("### Anomaly Rate by Total Reviews")
            st.bar_chart(review_stats_en.set_index("Total Reviews")["Anomaly Rate (%)"])
            
            with st.expander("View Detailed Review Statistics"):
                st.dataframe(safe_dataframe(review_stats_en), use_container_width=True)

        # --- 4. Playtime per Achievement ---
        if "playtime_per_achievement" in raw_profile.columns:
            play_bins = [0, 1, 5, 10, 25, 50, 100, np.inf]
            play_labels = ["0-1", "1-5", "5-10", "10-25", "25-50", "50-100", "> 100"]
            play_stats = build_binned_rate_table(raw_profile, "playtime_per_achievement", play_bins, play_labels)
            
            play_stats_en = play_stats.rename(columns={"bin": "Playtime/Achiev (min)", "flag_rate_pct": "Anomaly Rate (%)"})
            
            st.write("### Anomaly Rate by Playtime/Achievement")
            st.bar_chart(play_stats_en.set_index("Playtime/Achiev (min)")["Anomaly Rate (%)"])
            
            with st.expander("View Detailed Playtime Statistics"):
                st.dataframe(safe_dataframe(play_stats_en), use_container_width=True)

    # --- 5. Created Year ---
    if "created_year" in raw_profile.columns:
        year_stats = build_categorical_rate_table(raw_profile.dropna(subset=["created_year"]), "created_year")
        year_stats["created_year"] = pd.to_numeric(year_stats["created_year"], errors="coerce").astype("Int64")
        year_stats = year_stats.sort_values("created_year")
        
        year_stats_en = year_stats.rename(columns={"created_year": "Year Created", "flag_rate_pct": "Anomaly Rate (%)"})
        
        st.write("### Anomaly Rate by Account Creation Year")
        st.line_chart(year_stats_en.set_index("Year Created")["Anomaly Rate (%)"])
        
        with st.expander("View Detailed Yearly Statistics"):
            st.dataframe(safe_dataframe(year_stats_en), use_container_width=True)

# Section 4: Model Plots and Evaluation
st.divider()
st.header("4) Model Plots and Evaluation")

plot_tab_1, plot_tab_2, plot_tab_3 = st.tabs(["SHAP Summary", "SHAP Scatter", "Model Evaluation"])

with plot_tab_1:
    c1, c2 = st.columns(2)
    with c1:
        show_plot_if_exists("outputs/plots/shap_summary.png", "SHAP summary")
    with c2:
        show_plot_if_exists("outputs/plots/shap_waterfall.png", "SHAP waterfall")

with plot_tab_2:
    scatter_paths = sorted(glob("outputs/plots/shap_scatter_*.png"))
    if scatter_paths:
        cols = st.columns(2)
        for idx, p in enumerate(scatter_paths):
            with cols[idx % 2]:
                # Show SHAP PC scatter plots at a more readable size.
                st.image(p, caption=os.path.basename(p), width=680)
    else:
        st.info("No found file shap_scatter_*.png in outputs/plots")

with plot_tab_3:
    c1, c2 = st.columns(2)
    with c1:
        show_plot_if_exists("outputs/plots/xgb_pr_curve.png", "XGBoost PR curve")
    with c2:
        fi_candidates = sorted(glob("outputs/plots/xgb_feature_importance*.png"))
        if fi_candidates:
            show_plot_if_exists(fi_candidates[0], "XGBoost feature importance")

# Section 5: Output Explorer
st.divider()
st.header("5) Output Explorer")
st.caption("View quick output CSV and log files from the training/pipeline.")

all_output_files = list_output_files("outputs")
csv_files = [p for p in all_output_files if p.lower().endswith(".csv")]
log_files = [p for p in all_output_files if p.lower().endswith(".log") or "/logs/" in p.lower()]

exp_tab_csv, exp_tab_log = st.tabs(["CSV", "LOG"])

with exp_tab_csv:
    if not csv_files:
        st.info("No CSV files found in outputs.")
    else:
        selected_csv = st.selectbox(
            "Select CSV to view",
            options=csv_files,
            index=csv_files.index("outputs/ensemble_results.csv") if "outputs/ensemble_results.csv" in csv_files else 0,
        )

        meta = get_file_meta(selected_csv)
        c1, c2, c3 = st.columns(3)
        c1.metric("Size", meta["size_human"])
        c2.metric("Last Modified", meta["mtime"])
        c3.metric("Path", selected_csv)

        try:
            csv_df = pd.read_csv(selected_csv)
            st.write(f"Rows: {len(csv_df):,} | Cols: {csv_df.shape[1]}")
            preview_rows = st.slider("Preview Rows", min_value=10, max_value=300, value=80, step=10, key="output_csv_preview")
            st.dataframe(safe_dataframe(csv_df.head(preview_rows)), width="stretch")
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")

        try:
            with open(selected_csv, "rb") as f:
                file_bytes = f.read()
            st.download_button(
                "Download CSV",
                data=file_bytes,
                file_name=os.path.basename(selected_csv),
                mime="text/csv",
                key=f"download_{selected_csv}",
            )
        except Exception as exc:
            st.warning(f"Failed to create CSV download button: {exc}")

with exp_tab_log:
    if not log_files:
        st.info("No log files found in outputs.")
    else:
        latest_log = max(log_files, key=lambda p: os.path.getmtime(p))
        selected_log = st.selectbox(
            "Select log to view",
            options=log_files,
            index=log_files.index(latest_log),
        )

        meta = get_file_meta(selected_log)
        c1, c2, c3 = st.columns(3)
        c1.metric("Size", meta["size_human"])
        c2.metric("Last Modified", meta["mtime"])
        c3.metric("Path", selected_log)

        try:
            with open(selected_log, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            st.code(content[-15000:])
        except Exception as exc:
            st.error(f"Failed to read log file: {exc}")

        try:
            with open(selected_log, "rb") as f:
                file_bytes = f.read()
            st.download_button(
                "Download Log",
                data=file_bytes,
                file_name=os.path.basename(selected_log),
                mime="text/plain",
                key=f"download_{selected_log}",
            )
        except Exception as exc:
            st.warning(f"Failed to create log download button: {exc}")

st.caption("© 2026 - Data 4 Life")