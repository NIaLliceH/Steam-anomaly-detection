"""
Phase 2 — Steps 2 & 3: Heuristic Labels and Feature Engineering.

Heuristic labels are built FIRST (before full feature engineering) to serve
as pseudo ground truth for hyperparameter tuning. Known leakage: several
features used in Step 3 overlap with the rule thresholds in Step 2. This is
an accepted trade-off given the absence of real ground truth — document it.
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timezone offset lookup — approximate UTC offsets for major Steam markets.
# When a player's country is unknown or not in this table, offset defaults to
# 0 (UTC) so the feature degrades gracefully rather than erroring.
# ---------------------------------------------------------------------------
_COUNTRY_UTC_OFFSET: dict[str, int] = {
    # Americas — ISO-2
    "US": -5, "CA": -5, "MX": -6,
    "BR": -3, "AR": -3, "CL": -4, "CO": -5, "PE": -5,
    # Europe — ISO-2
    "GB": 0,  "IE": 0,
    "DE": 1,  "FR": 1,  "PL": 1,  "ES": 1,  "IT": 1,
    "NL": 1,  "BE": 1,  "AT": 1,  "CH": 1,  "CZ": 1,
    "SE": 1,  "NO": 1,  "DK": 1,  "FI": 2,
    "PT": 0,  "RO": 2,  "HU": 1,  "SK": 1,  "HR": 1,
    # Eastern Europe / Middle East — ISO-2
    "RU": 3,  "UA": 2,  "BY": 3,  "TR": 3,
    "IL": 2,  "EG": 2,  "SA": 3,  "AE": 4,
    # Asia-Pacific — ISO-2
    "CN": 8,  "JP": 9,  "KR": 9,
    "TW": 8,  "HK": 8,  "SG": 8,
    "TH": 7,  "VN": 7,  "ID": 7,  "MY": 8,  "PH": 8,
    "IN": 5,
    "AU": 10, "NZ": 12,
    # Africa — ISO-2
    "ZA": 2,  "NG": 1,  "KE": 3,
    # -----------------------------------------------------------------------
    # Full country names — players.csv stores full names, not ISO-2 codes.
    # EDA confirmed: 99.7% of non-null entries are full names (ISO-2 match: 0.3%)
    # -----------------------------------------------------------------------
    # Americas — full names
    "United States": -5, "Canada": -5, "Mexico": -6,
    "Brazil": -3, "Argentina": -3, "Chile": -4, "Colombia": -5, "Peru": -5,
    "Venezuela": -4, "Ecuador": -5, "Bolivia": -4, "Paraguay": -4, "Uruguay": -3,
    # Europe — full names
    "United Kingdom": 0, "Ireland": 0,
    "Germany": 1, "France": 1, "Poland": 1, "Spain": 1, "Italy": 1,
    "Netherlands": 1, "Belgium": 1, "Austria": 1, "Switzerland": 1,
    "Czech Republic": 1, "Czechia": 1,
    "Sweden": 1, "Norway": 1, "Denmark": 1, "Finland": 2,
    "Portugal": 0, "Romania": 2, "Hungary": 1, "Slovakia": 1, "Croatia": 1,
    "Bulgaria": 2, "Serbia": 1, "Greece": 2, "Slovenia": 1,
    "Estonia": 2, "Latvia": 2, "Lithuania": 2,
    "Bosnia and Herzegovina": 1, "North Macedonia": 1, "Albania": 1,
    "Montenegro": 1, "Kosovo": 1,
    "Iceland": 0, "Luxembourg": 1, "Malta": 1, "Cyprus": 2,
    # Eastern Europe / CIS — full names
    "Russia": 3, "Russian Federation": 3,
    "Ukraine": 2, "Belarus": 3, "Kazakhstan": 5,
    "Azerbaijan": 4, "Armenia": 4, "Georgia": 4,
    "Uzbekistan": 5, "Kyrgyzstan": 6, "Tajikistan": 5, "Turkmenistan": 5,
    "Moldova": 2,
    # Middle East — full names
    "Turkey": 3, "Türkiye": 3, "Israel": 2, "Egypt": 2,
    "Saudi Arabia": 3, "United Arab Emirates": 4,
    "Iran": 3, "Iraq": 3, "Jordan": 2, "Lebanon": 2, "Kuwait": 3,
    "Qatar": 3, "Bahrain": 3, "Oman": 4, "Yemen": 3,
    # Asia-Pacific — full names
    "China": 8, "Japan": 9, "South Korea": 9, "Korea, Republic of": 9,
    "Taiwan": 8, "Hong Kong": 8, "Singapore": 8,
    "Thailand": 7, "Vietnam": 7, "Indonesia": 7, "Malaysia": 8, "Philippines": 8,
    "India": 5, "Pakistan": 5, "Bangladesh": 6, "Sri Lanka": 5,
    "Nepal": 6, "Myanmar": 6,
    "Australia": 10, "New Zealand": 12,
    "Mongolia": 8,
    # Africa — full names
    "South Africa": 2, "Nigeria": 1, "Kenya": 3,
    "Egypt": 2, "Morocco": 0, "Algeria": 1, "Tunisia": 1,
    "Ethiopia": 3, "Ghana": 0, "Tanzania": 3, "Uganda": 3,
    "Angola": 1, "Congo": 1, "Congo, The Democratic Republic of the": 1,
    "Gabon": 1, "Niger": 1, "Chad": 1, "Somalia": 3, "Djibouti": 3,
    "Rwanda": 2, "Zambia": 2, "Mozambique": 2, "Seychelles": 4,
    # ISO 3166 official/alternative name variants found in data
    "Iran, Islamic Republic of": 3,
    "Viet Nam": 7, "VN": 7,
    "Taiwan, Province of China": 8,
    "Korea, Democratic People's Republic of": 9,
    "Moldova, Republic of": 2,
    "Venezuela, Bolivarian Republic of": -4,
    "Bolivia, Plurinational State of": -4,
    "Lao People's Democratic Republic": 7,
    "Brunei Darussalam": 8,
    "Macao": 8,
    # Balkans / small Europe
    "Bosnia and Herzegovina": 1, "North Macedonia": 1,
    "Albania": 1, "Montenegro": 1, "San Marino": 1,
    "Kosovo": 1, "Liechtenstein": 1, "Monaco": 1, "Gibraltar": 0,
    "Isle of Man": 0, "Greenland": -3,
    # Middle East / South Asia
    "Lebanon": 2, "Palestine, State of": 2,
    "Sri Lanka": 5, "Afghanistan": 4, "Myanmar": 6, "Tajikistan": 5,
    # Americas — small countries
    "Jamaica": -5, "Cuba": -5, "Dominican Republic": -4,
    "Costa Rica": -6, "El Salvador": -6, "Guatemala": -6,
    "Paraguay": -4, "Suriname": -3, "Cabo Verde": -1,
    "Puerto Rico": -4, "Virgin Islands, U.S.": -4,
    "Barbados": -4, "Martinique": -4, "Guadeloupe": -4,
    # Pacific territories
    "Fiji": 12, "Cocos (Keeling) Islands": 7, "Christmas Island": 7,
}


# ---------------------------------------------------------------------------
# Pre-processing helpers (called after loading parquet in main.py)
# ---------------------------------------------------------------------------

def add_time_components(history: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day_of_week, date_only columns derived from date_acquired."""
    history = history.copy()
    history["hour"]        = history["date_acquired"].dt.hour.astype("Int8")
    history["day_of_week"] = history["date_acquired"].dt.dayofweek.astype("Int8")
    history["date_only"]   = history["date_acquired"].dt.date
    return history


def build_player_library(purchased: pd.DataFrame) -> dict:
    """
    Return {playerid (int): set(gameid (int))} for O(1) library membership checks.
    Now handles the structured list of dicts: [{"appid": 10, "playtime_mins": 0}]
    """
    result = {}
    for pid, lib in zip(purchased["playerid"], purchased["library"]):
        if lib is None or not isinstance(lib, np.ndarray):
            result[int(pid)] = set()
        else:
            # lib is an array of dicts
            result[int(pid)] = {int(item.get('appid')) for item in lib if item and item.get('appid') is not None}
    return result


def build_zero_playtime_library(purchased: pd.DataFrame) -> dict:
    """
    Return {playerid (int): set(gameid (int))} containing only games where
    playtime_mins == 0 (owned but never launched).

    Distinct from build_player_library: a player can own a game (present in
    that dict) yet have actually played it (playtime > 0).  This dict isolates
    the subset of owned-but-unplayed games, which is the correct denominator
    for review_unplayed_ratio.

    Players absent from purchased_games are NOT included in the result.
    Callers should treat a missing key as NaN (no library data available)
    rather than an empty set (library known, but all games were played).
    """
    result = {}
    for pid, lib in zip(purchased["playerid"], purchased["library"]):
        pid_int = int(pid)
        if lib is None or not isinstance(lib, np.ndarray):
            result[pid_int] = set()
        else:
            result[pid_int] = {
                int(item['appid'])
                for item in lib
                if item
                and item.get('appid') is not None
                and item.get('playtime_mins', -1) == 0
            }
    return result


# ---------------------------------------------------------------------------
# Step 2 — Heuristic Labels
# ---------------------------------------------------------------------------

def build_heuristic_labels(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Create pseudo ground truth (heuristic_bot = 0/1) using 3 AND-rule groups.

    Accepts the pre-computed feature_matrix (output of build_feature_matrix) so
    that speed, temporal, review, and playtime signals are not duplicated — the
    previous version recomputed median_interval, max_per_day, night_ratio, etc.
    from raw tables, identical to what build_feature_matrix already computed.

    Bot archetypes
    --------------
    Speed bot  : median_unlock_interval_sec < 10 s AND top1_game_concentration > 0.85
    Volume bot : (max_per_day > 500 AND night > 40% AND total_ach > 1000)
                 OR zero_playtime_achievements_ratio > 0.9  (SAM unlocker catch-all)
    Review bot : total_reviews > 5 AND total_achievements == 0
                 AND review_unplayed_ratio > 0.50 AND review_duplication_rate > 0.50

    Normal definition (for PU Learning)
    ------------------------------------
    total_achievements > 10 AND median_unlock_interval_sec > 600 (10 min).
    Threshold lowered from 30 min → 10 min to better reflect realistic gameplay
    pace and reduce confident-normal class under-representation.
    """
    log.info("Step 2 — Building heuristic labels from feature matrix …")

    fm = feature_matrix

    # Safe defaults: NaN in rule features → conservative non-bot value
    median_ivl  = fm["median_unlock_interval_sec"].fillna(np.inf)
    top1_conc   = fm["top1_game_concentration"].fillna(0.0)
    max_per_day = fm["max_achievements_per_day"].fillna(0.0)
    night       = fm["night_activity_ratio"].fillna(0.0)
    total_ach   = fm["total_achievements"].fillna(0.0)
    total_rev   = fm["total_reviews"].fillna(0.0)
    unplayed    = fm["review_unplayed_ratio"].fillna(0.0)
    dup_rate    = fm["review_duplication_rate"].fillna(0.0)
    zp_ratio    = fm["zero_playtime_achievements_ratio"].fillna(0.0)
    avg_rev_len  = fm["avg_review_length"].fillna(0.0)

    # ── Group 1: SPEED BOT ────────────────────────────────────────────────────
    speed_bot = (median_ivl < 10) & (top1_conc > 0.85)

    # ── Group 2: VOLUME BOT ───────────────────────────────────────────────────
    # Primary: inhuman unlock volume + nocturnal pattern.
    # OR: >90% achievements on unplayed games = deterministic SAM signal.
    volume_bot = (
        (max_per_day > 500) & (night > 0.40) & (total_ach > 1000)
    ) | (zp_ratio > 0.9)

    # ── Group 3: REVIEW BOT ───────────────────────────────────────────────────
    # Both dual signals required to prevent false positives from players who
    # merely review a few unplayed demos or use short identical templates.
    review_bot = (
        (total_rev > 5)
        & (total_ach < 5) # Tài khoản gần như không chơi game
        & (unplayed > 0.50)
        & (dup_rate > 0.50)
        & (avg_rev_len < 50) # Thêm: Review rác thường rất ngắn (< 50 ký tự)
    )

    result = pd.DataFrame(index=fm.index)
    result["heuristic_bot"]    = (speed_bot | volume_bot | review_bot).astype(int)
    result["heuristic_normal"] = (
        (total_ach > 10) & (median_ivl > 600)
    ).astype(int)

    log.info("  Speed bots:       %d", int(speed_bot.sum()))
    log.info("  Volume bots:      %d", int(volume_bot.sum()))
    log.info("  Review bots:      %d", int(review_bot.sum()))
    n_bot    = int(result["heuristic_bot"].sum())
    n_normal = int(result["heuristic_normal"].sum())
    log.info("  Heuristic bots   (total unique): %d (%.2f%%)",
             n_bot,    result["heuristic_bot"].mean()    * 100)
    log.info("  Heuristic normal (total unique): %d (%.2f%%)",
             n_normal, result["heuristic_normal"].mean() * 100)
    return result


# ---------------------------------------------------------------------------
# Step 3 — Feature Engineering (vectorised groupby, per-player aggregation)
# ---------------------------------------------------------------------------

def _speed_features(history: pd.DataFrame) -> pd.DataFrame:
    """Group A: unlock speed/rhythm statistics."""
    h = history.sort_values(["playerid", "date_acquired"]).copy()
    h["_prev_gameid"]  = h.groupby("playerid")["gameid"].shift()
    h["interval_sec"]  = (
        h.groupby("playerid")["date_acquired"]
        .diff()
        .dt.total_seconds()
    )

    # Cross-game transitions only — same-game zero-intervals are Steam chain
    # reactions (one milestone unlocking multiple achievements simultaneously),
    # not bot signals. EDA validated: 99.5% of zero-intervals are same-game chains.
    cross_mask = h["_prev_gameid"].notna() & ~(
        (h["interval_sec"] == 0) & (h["gameid"] == h["_prev_gameid"])
    )
    h_cross = h[cross_mask]

    speed = h_cross.groupby("playerid")["interval_sec"].agg(
        median_unlock_interval_sec="median",
        std_unlock_interval_sec="std",
    )
    mean_interval = h_cross.groupby("playerid")["interval_sec"].mean()
    speed["cv_unlock_interval"] = (
        speed["std_unlock_interval_sec"]
        / mean_interval.where(mean_interval > 0)
    )

    max_per_min = (
        history.assign(minute=history["date_acquired"].dt.floor("min"))
        .groupby(["playerid", "minute"])
        .size()
        .groupby(level=0).max()
        .rename("max_achievements_per_minute")
    )
    max_per_day = (
        history.groupby(["playerid", "date_only"])
        .size()
        .groupby(level=0).max()
        .rename("max_achievements_per_day")
    )
    return pd.concat([speed, max_per_min, max_per_day], axis=1)


def _temporal_features(history: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Group B: time-of-day and activity-pattern features (timezone-aware).

    UTC timestamps are shifted to each player's approximate local time using
    the country code in `players`.  Bots that run scripts during the server's
    night window (UTC 00:00–05:59) would appear human if we naively compared
    against local night hours — applying the offset corrects for this.  For
    players whose country is absent from _COUNTRY_UTC_OFFSET, offset defaults
    to 0 (UTC) so the feature degrades gracefully.
    """
    # ── Build per-player UTC offset vector (vectorised, O(N players)) ─────────
    offset_map = (
        players.set_index("playerid")["country"]
        .astype(str)
        .map(_COUNTRY_UTC_OFFSET)
        .fillna(0)
        .astype(int)
    )

    h = history.copy()
    h["utc_offset"] = h["playerid"].map(offset_map).fillna(0).astype(int)
    # 24-hour wrap-around: (UTC_hour + offset) mod 24 gives the local hour.
    h["local_hour"] = (h["hour"] + h["utc_offset"]) % 24

    # Night activity ratio (00:00–05:59 **local time**)
    night_counts = (
        h[h["local_hour"] < 6].groupby("playerid").size().rename("_night")
    )
    total_ach = history.groupby("playerid").size().rename("_total")
    night_ratio = (night_counts / total_ach).rename("night_activity_ratio").fillna(0)

    # Shannon entropy over 24-hour distribution (local time)
    hour_counts = (
        h.groupby(["playerid", "local_hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    hour_entropy = (
        hour_counts.apply(lambda row: scipy_entropy(row), axis=1)
        .rename("hour_entropy")
    )

    # Activity density = active days / calendar span
    span = history.groupby("playerid")["date_acquired"].agg(
        _first="min", _last="max"
    )
    span["span_days"] = (span["_last"] - span["_first"]).dt.days + 1
    active_days = (
        history.groupby("playerid")["date_only"].nunique().rename("_active")
    )
    activity_density = (active_days / span["span_days"]).rename("activity_density")

    return pd.concat([night_ratio, hour_entropy, activity_density], axis=1)


def _diversity_features(history: pd.DataFrame,
                        purchased: pd.DataFrame) -> pd.DataFrame:
    """Group C: game diversity and library coverage."""
    total_achievements  = history.groupby("playerid").size().rename("total_achievements")
    games_with_ach      = history.groupby("playerid")["gameid"].nunique().rename("games_with_achievements")
    lib_size            = purchased.set_index("playerid")["library_size"].rename("library_size")

    # achievement_game_ratio: treat missing library as 0 owned games → denom = 1
    lib_filled = lib_size.reindex(total_achievements.index).fillna(0)
    ach_game_ratio = (games_with_ach / lib_filled.clip(lower=1)).rename("achievement_game_ratio")

    # Per-game concentration metric
    game_counts = history.groupby(["playerid", "gameid"]).size()
    game_totals = game_counts.groupby(level=0).sum()
    game_props  = game_counts / game_totals

    top1_conc = game_props.groupby(level=0).max().rename("top1_game_concentration")
    avg_ach_per_game = (total_achievements / games_with_ach).rename("avg_achievements_per_game")

    return pd.concat(
        [total_achievements, lib_size,
         ach_game_ratio, top1_conc, avg_ach_per_game],
        axis=1,
    )
    
    
def _playtime_features(history: pd.DataFrame, purchased: pd.DataFrame) -> pd.DataFrame:
    """Group F: Playtime plausibility with async-safe achievement cross-referencing.

    **Condition A — API Lag (excluded from all metrics)**
      Game appears in `history` but is NOT present in the player's `library`.
      Cause: the game was purchased *after* the library snapshot was taken, or
      the library API call timed out.  These rows are silently dropped so that
      async artefacts never inflate or deflate the bot score.

    **Condition B — Normal gameplay**
      Game is in `library` with `playtime_mins > 0`.
      The player demonstrably launched the game before unlocking achievements.
      Counts toward both the playtime sum and the valid-achievement denominator.

    **Condition C — SAM / Achievement Unlocker (the anomaly signal)**
      Game is in `library` but `playtime_mins == 0`.
      The achievement was written to the account while the game was never
      launched — the deterministic footprint of Steam Achievement Manager (SAM)
      and similar tools.  Counts toward the denominator but contributes 0 mins.
    """
    # ── Step 1: Explode purchased library → flat (playerid, gameid, playtime_mins)
    # .apply() here is on purchased (~N_players rows), NOT on history (9 M rows).
    lib_df = purchased[["playerid", "library"]].copy()
    lib_df["library"] = lib_df["library"].apply(
        lambda x: list(x) if isinstance(x, np.ndarray) else
                  (x if isinstance(x, list) else [])
    )
    lib_df = lib_df.explode("library").dropna(subset=["library"])

    if lib_df.empty:
        return pd.DataFrame(
            index=pd.Index(history["playerid"].unique(), name="playerid"),
            columns=["zero_playtime_achievements_ratio",
                     "playtime_per_achievement", "total_playtime_mins"],
            dtype=float,
        )

    # pd.json_normalize converts the list of dicts to a tidy DataFrame without
    # any per-row Python loops.
    lib_norm = pd.json_normalize(lib_df["library"].tolist())
    appid_col    = lib_norm["appid"]        if "appid"        in lib_norm.columns else pd.Series(np.nan, index=lib_norm.index)
    playtime_col = lib_norm["playtime_mins"] if "playtime_mins" in lib_norm.columns else pd.Series(-1,    index=lib_norm.index)

    lib_flat = pd.DataFrame({
        "playerid":     lib_df["playerid"].values,
        "gameid":       pd.to_numeric(appid_col,    errors="coerce"),
        "playtime_mins": pd.to_numeric(playtime_col, errors="coerce").fillna(-1),
    }).dropna(subset=["gameid"])
    lib_flat["gameid"]    = lib_flat["gameid"].astype("int64")
    lib_flat["playerid"]  = lib_flat["playerid"].astype("int64")
    # Guard against duplicate (playerid, gameid) entries: keep highest playtime.
    lib_flat = (
        lib_flat.groupby(["playerid", "gameid"], as_index=False)["playtime_mins"].max()
    )

    # ── Step 2: Left-join history with library ─────────────────────────────────
    # Drop rows where gameid is null (malformed achievementids).
    hist_core = (
        history[["playerid", "gameid"]]
        .dropna(subset=["gameid"])
        .assign(gameid=lambda df: df["gameid"].astype("int64"))
    )
    hist_classified = hist_core.merge(lib_flat, on=["playerid", "gameid"], how="left")

    # ── Step 3: Vectorised condition masks ────────────────────────────────────
    # Condition A: playtime_mins is NaN  → game not in library (async lag)
    # Condition B: playtime_mins  > 0   → normal gameplay
    # Condition C: playtime_mins == 0   → owned, zero playtime (SAM bot)
    cond_b = hist_classified["playtime_mins"] > 0
    cond_c = hist_classified["playtime_mins"] == 0

    # ── Step 4: Per-player counts (fully vectorised) ──────────────────────────
    count_b = hist_classified.loc[cond_b, "playerid"].value_counts().rename("_b")
    count_c = hist_classified.loc[cond_c, "playerid"].value_counts().rename("_c")
    counts  = pd.concat([count_b, count_c], axis=1).fillna(0).astype(int)
    denom   = (counts["_b"] + counts["_c"]).replace(0, np.nan)

    zero_pt_ratio = (counts["_c"] / denom).rename("zero_playtime_achievements_ratio")

    # ── Step 5: Total playtime — sum over DISTINCT Condition B games ──────────
    # A game's playtime is a fixed account-level value; summing it once per
    # distinct game (not once per achievement) avoids artificial inflation for
    # games with many achievements.
    b_game_pt = (
        hist_classified.loc[cond_b, ["playerid", "gameid", "playtime_mins"]]
        .drop_duplicates(subset=["playerid", "gameid"])
        .groupby("playerid")["playtime_mins"]
        .sum()
        .rename("total_playtime_mins")
    )
    playtime_per_ach = (b_game_pt / denom).rename("playtime_per_achievement")

    return pd.concat([zero_pt_ratio, playtime_per_ach, b_game_pt], axis=1)


def _review_features(reviews: pd.DataFrame,
                     zero_playtime_library: dict) -> pd.DataFrame:
    """Group D: review behaviour signals."""
    total_reviews = reviews.groupby("playerid").size().rename("total_reviews")

    # Fraction of reviews for games with playtime_mins == 0.
    def _unplayed_ratio(group: pd.DataFrame) -> float:
        pid = int(group.name)
        if pid not in zero_playtime_library:
            return np.nan  # no library data — don't impute a fake ratio
        zero_played = zero_playtime_library[pid]
        return group["gameid"].apply(int).isin(zero_played).mean()

    review_unplayed = (
        reviews.groupby("playerid")
        .apply(_unplayed_ratio, include_groups=False)
        .rename("review_unplayed_ratio")
    )

    # Duplicate review rate: 1 − (unique_texts / total_reviews).
    def _dup_rate(s: pd.Series) -> float:
        cleaned = s.fillna("").str.lower().str.strip()
        if len(cleaned) <= 1:
            return 0.0
        return 1.0 - (cleaned.nunique() / len(cleaned))

    review_dup = (
        reviews.groupby("playerid")["review"]
        .apply(_dup_rate)
        .rename("review_duplication_rate")
    )

    rev_lens = reviews.assign(rlen=reviews["review"].fillna("").str.len())
    avg_rev_len = rev_lens.groupby("playerid")["rlen"].mean().rename("avg_review_length")

    return pd.concat(
        [total_reviews, review_unplayed, review_dup, avg_rev_len],
        axis=1,
    )


def _account_age_features(
    history: pd.DataFrame,
    players: pd.DataFrame,
    reference_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Bonus: account age and time-to-first-achievement."""
    player_created = players.set_index("playerid")["created"]
    first_ach      = history.groupby("playerid")["date_acquired"].min()

    days_before_first = (
        (first_ach - player_created)
        .dt.days
        .clip(lower=0)
        .rename("days_before_first_achievement")
    )
    ref_ts = pd.Timestamp.now() if reference_time is None else pd.Timestamp(reference_time)
    account_age = (
        (ref_ts - player_created)
        .dt.days
        .rename("account_age_days")
    )
    return pd.concat([days_before_first, account_age], axis=1)


def build_feature_matrix(history: pd.DataFrame,
                          reviews: pd.DataFrame,
                          players: pd.DataFrame,
                          purchased: pd.DataFrame,
                          reference_time: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Step 3: Compute all per-player features across 6 groups (A–F).

    Trimming
    --------------
    The raw history table contains ~196 K distinct players, but ~193 K of them
    have fewer than 10 achievements and would be discarded by the downstream
    PU-learning filter anyway.  Computing Shannon entropy, inter-arrival
    standard deviation, concentration metrics, and playtime cross-joins for those players wastes
    significant CPU and RAM.  Trimming *before* any heavy groupby/merge reduces
    the working set by ~98% and cuts wall-clock feature-engineering time
    proportionally.

    Criteria: (≥10 achievements OR ≥3 reviews) AND library_size ≥ 1.
    - OR logic: keeps Review Bot candidates who have 0 achievements but spam reviews.
    - library_size ≥ 1: excludes ghost accounts with no owned games (they would
      produce NaN-only playtime feature rows with no bot signal).
    """
    log.info("Step 3 — Building feature matrix …")

    # ── Trimming ─────────────────────────────────────────────────────────────
    # Keep players with ≥10 achievements (bot/normal signal) OR ≥3 reviews
    # (Review Bot candidates who have 0 achievements but spam reviews).
    # Pure review bots are dropped by the old ≥10 achievements-only filter,
    # making the Review Bot heuristic rule impossible to fire.
    # Also require library_size ≥ 1 to exclude ghost accounts with no owned games
    # (they have no playtime signal and would produce NaN-only feature rows).
    ach_counts_all    = history.groupby("playerid").size()
    review_counts_all = reviews.groupby("playerid").size()
    lib_sizes = purchased.set_index("playerid")["library_size"].fillna(0)
    has_library = lib_sizes[lib_sizes >= 1].index
    core_ids = (
        ach_counts_all[ach_counts_all >= 10].index
        .union(review_counts_all[review_counts_all >= 3].index)
    ).intersection(has_library)
    total_unique = len(
        set(history["playerid"].unique()) | set(reviews["playerid"].unique())
    )
    log.info(
        "  Trimming: %d total players → %d with (≥10 achievements OR ≥3 reviews) AND library_size ≥ 1 (%d dropped)",
        total_unique, len(core_ids), total_unique - len(core_ids),
    )
    history   = history[history["playerid"].isin(core_ids)].copy()
    reviews   = reviews[reviews["playerid"].isin(core_ids)].copy()
    purchased = purchased[purchased["playerid"].isin(core_ids)].copy()

    all_players = pd.Index(
        sorted(set(history["playerid"].unique()) | set(reviews["playerid"].unique())),
        name="playerid",
    )

    log.info("  Group A: speed features …")
    grp_a = _speed_features(history)

    log.info("  Group B: temporal features …")
    grp_b = _temporal_features(history, players)

    log.info("  Group C: diversity features …")
    grp_c = _diversity_features(history, purchased)

    log.info("  Group D: review features …")
    zero_playtime_lib = build_zero_playtime_library(purchased)
    grp_d = _review_features(reviews, zero_playtime_lib)
    grp_d = grp_d.reindex(all_players).fillna(
        {"total_reviews": 0,
         "review_unplayed_ratio": 0.0,
         "review_duplication_rate": 0.0,
            "avg_review_length": 0.0}
    )

    log.info("  Group E: account age features …") # Sửa lại log info
    grp_bonus = _account_age_features(history, players, reference_time=reference_time)

    log.info("  Group F: playtime features …")
    grp_playtime = _playtime_features(history, purchased)

    feature_matrix = pd.concat(
        [grp_a, grp_b, grp_c, grp_d, grp_bonus, grp_playtime], axis=1
    ).reindex(all_players)

    log.info("  Feature matrix: %d players × %d features",
             *feature_matrix.shape)
    return feature_matrix