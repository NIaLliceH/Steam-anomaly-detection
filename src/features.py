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
    # Americas
    "US": -5, "CA": -5, "MX": -6,
    "BR": -3, "AR": -3, "CL": -4, "CO": -5, "PE": -5,
    # Europe (CET/CEST approximated as +1)
    "GB": 0,  "IE": 0,
    "DE": 1,  "FR": 1,  "PL": 1,  "ES": 1,  "IT": 1,
    "NL": 1,  "BE": 1,  "AT": 1,  "CH": 1,  "CZ": 1,
    "SE": 1,  "NO": 1,  "DK": 1,  "FI": 2,
    "PT": 0,  "RO": 2,  "HU": 1,  "SK": 1,  "HR": 1,
    # Eastern Europe / Middle East
    "RU": 3,  "UA": 2,  "BY": 3,  "TR": 3,
    "IL": 2,  "EG": 2,  "SA": 3,  "AE": 4,
    # Asia-Pacific
    "CN": 8,  "JP": 9,  "KR": 9,
    "TW": 8,  "HK": 8,  "SG": 8,
    "TH": 7,  "VN": 7,  "ID": 7,  "MY": 8,  "PH": 8,
    "IN": 5,                        # IST = +5:30, approximated
    "AU": 10, "NZ": 12,
    # Africa
    "ZA": 2,  "NG": 1,  "KE": 3,
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

def build_heuristic_labels(history: pd.DataFrame,
                            reviews: pd.DataFrame,
                            zero_playtime_library: dict | None = None,
                            players: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Create pseudo ground truth (heuristic_bot = 0/1) using 3 AND-rule groups.

    V2 changes vs V1:
    - OR logic → AND logic per group to reduce false positives.
      V1's OR rules flagged 98%+ of players; AND rules target only clear bots.
    - 3 distinct bot archetypes: Speed bot, Volume bot, Review bot.
    - Review bot rule uses review_unplayed_ratio when zero_playtime_library is
      provided (preferred), or falls back to review_dup_ratio otherwise.
    - night_ratio is now computed in local time when `players` (with country
      column) is supplied.  Falls back to UTC if players=None.
    """
    log.info("Step 2 — Building heuristic labels …")

    # ── Speed stats ──────────────────────────────────────────────────────────
    diffs = (
        history.sort_values(["playerid", "date_acquired"])
        .groupby("playerid")["date_acquired"]
        .apply(lambda x: x.diff().dt.total_seconds())
    )
    median_interval = diffs.groupby(level=0).median().rename("median_interval")
    min_interval    = diffs.groupby(level=0).min().rename("min_interval")

    # ── Game concentration ────────────────────────────────────────────────────
    top1_conc = (
        history.groupby(["playerid", "gameid"]).size()
        .groupby(level=0).apply(lambda x: x.max() / x.sum())
        .rename("top1_concentration")
    )

    # ── Max achievements per day ──────────────────────────────────────────────
    max_per_day = (
        history.groupby(["playerid", "date_only"]).size()
        .groupby(level=0).max()
        .rename("max_per_day")
    )

    # ── Night activity ratio (00:00–05:59 local time) ────────────────────────
    # Apply per-player UTC offset so the window is relative to the player's
    # local timezone, not the server clock.  Same logic as _temporal_features.
    if players is not None and "country" in players.columns:
        offset_map = (
            players.set_index("playerid")["country"]
            .astype(str)
            .map(_COUNTRY_UTC_OFFSET)
            .fillna(0)
            .astype(int)
        )
        _h = history.copy()
        _h["local_hour"] = (_h["hour"] + _h["playerid"].map(offset_map).fillna(0).astype(int)) % 24
        night_counts = _h[_h["local_hour"] < 6].groupby("playerid").size().rename("_night")
        total_counts = history.groupby("playerid").size().rename("_total")
        night_ratio  = (night_counts / total_counts).rename("night_ratio").fillna(0)
        log.info("  night_ratio computed in local time (country-aware UTC offset).")
    else:
        night_ratio = (
            history.assign(is_night=(history["hour"] < 6))
            .groupby("playerid")["is_night"].mean()
            .rename("night_ratio")
        )
        log.info("  night_ratio computed in UTC (no players data supplied).")

    # ── Achievement and review counts ─────────────────────────────────────────
    ach_counts    = history.groupby("playerid").size().rename("ach_count")
    review_counts = reviews.groupby("playerid").size().rename("review_count")

    # ── Review signal for Group 3 ─────────────────────────────────────────────
    # Preferred: review_unplayed_ratio — fraction of reviews for games with 0
    # playtime.  Requires zero_playtime_library to be passed by the caller.
    # Fallback: review_dup_ratio — copy-paste rate, always computable from text.
    if zero_playtime_library is not None:
        def _unplayed(group: pd.DataFrame) -> float:
            pid = int(group.name)
            if pid not in zero_playtime_library:
                return np.nan
            return group["gameid"].apply(int).isin(zero_playtime_library[pid]).mean()

        review_signal = (
            reviews.groupby("playerid")
            .apply(_unplayed, include_groups=False)
            .rename("_review_signal")
        )
        review_signal_threshold = 0.50
        log.info("  Review bot signal: review_unplayed_ratio (threshold %.2f)",
                 review_signal_threshold)
    else:
        def _dup_rate(s: pd.Series) -> float:
            cleaned = s.fillna("").str.lower().str.strip()
            return 0.0 if len(cleaned) <= 1 else 1.0 - (cleaned.nunique() / len(cleaned))

        review_signal = (
            reviews.groupby("playerid")["review"]
            .apply(_dup_rate)
            .rename("_review_signal")
        )
        review_signal_threshold = 0.50
        log.info("  Review bot signal: review_dup_ratio (fallback, threshold %.2f)",
                 review_signal_threshold)

    df = pd.concat(
        [median_interval, min_interval, top1_conc, max_per_day,
         night_ratio, ach_counts, review_counts, review_signal],
        axis=1,
    ).fillna({"review_count": 0, "night_ratio": 0.0, "min_interval": 0.0,
              "_review_signal": 0.0})

    # ── Group 1: SPEED BOT — fast unlock AND concentrated in one game ─────────
    # Real players need ≥ 30s between achievements even in easy games.
    speed_bot = (
        (df["median_interval"] < 10)        # median < 10 s
        & (df["min_interval"]  < 1)         # at least one unlock < 1 s
        & (df["top1_concentration"] > 0.85) # 85%+ achievements from a single game
    )

    # ── Group 2: VOLUME BOT — inhuman volume AND nocturnal activity ───────────
    # 500 achievements/day = 1 every 3 minutes, 24/7 — impossible for humans.
    volume_bot = (
        (df["max_per_day"]  > 500)
        & (df["night_ratio"] > 0.40)        # > 40% activity between 00:00–05:59
        & (df["ach_count"]   > 1000)        # total ≥ 1 000 achievements
    )

    # ── Group 3: REVIEW BOT — zero gameplay + suspicious review pattern ─────────
    review_bot = (
        (df["review_count"]    > 5)                          # enough sample
        & (df["ach_count"]    == 0)                          # never actually played
        & (df["_review_signal"] > review_signal_threshold)   # unplayed or dup reviews
    )

    df["heuristic_bot"] = (speed_bot | volume_bot | review_bot).astype(int)

    # ── Strict normal definition for PU Learning ──────────────────────────────
    # These players are confidently NOT bots: they have meaningful achievement
    # history (> 10) AND unlock achievements at a human pace (median > 30 min).
    # Used to filter the XGBoost training set to "confident bot vs confident
    # normal", dropping the ambiguous grey area that would add noise to labels.
    heuristic_normal = (
        (df["ach_count"] > 10)
        & (df["median_interval"] > 1800)  # > 30 minutes between achievements
    ).astype(int)
    df["heuristic_normal"] = heuristic_normal

    log.info("  Speed bots:       %d", speed_bot.sum())
    log.info("  Volume bots:      %d", volume_bot.sum())
    log.info("  Review bots:      %d", review_bot.sum())
    n_bot = df["heuristic_bot"].sum()
    n_normal = df["heuristic_normal"].sum()
    log.info("  Heuristic bots   (total unique): %d (%.2f%%)",
             n_bot, df["heuristic_bot"].mean() * 100)
    log.info("  Heuristic normal (total unique): %d (%.2f%%)",
             n_normal, df["heuristic_normal"].mean() * 100)
    return df[["heuristic_bot", "heuristic_normal"]]


# ---------------------------------------------------------------------------
# Step 3 — Feature Engineering (vectorised groupby, per-player aggregation)
# ---------------------------------------------------------------------------

def _speed_features(history: pd.DataFrame) -> pd.DataFrame:
    """Group A: unlock speed/rhythm statistics."""
    h = history.sort_values(["playerid", "date_acquired"])
    h = h.copy()
    h["interval_sec"] = (
        h.groupby("playerid")["date_acquired"]
        .diff()
        .dt.total_seconds()
    )

    speed = h.groupby("playerid")["interval_sec"].agg(
        median_unlock_interval_sec="median",
        min_unlock_interval_sec="min",
        std_unlock_interval_sec="std",
    )
    mean_interval = h.groupby("playerid")["interval_sec"].mean()
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

    weekend_ratio = (
        history.groupby("playerid")["day_of_week"]
        .apply(lambda x: x.isin([5, 6]).mean())
        .rename("weekend_ratio")
    )
    return pd.concat([night_ratio, hour_entropy, activity_density, weekend_ratio], axis=1)


def _diversity_features(history: pd.DataFrame,
                        purchased: pd.DataFrame) -> pd.DataFrame:
    """Group C: game diversity and library coverage."""
    total_achievements  = history.groupby("playerid").size().rename("total_achievements")
    games_with_ach      = history.groupby("playerid")["gameid"].nunique().rename("games_with_achievements")
    lib_size            = purchased.set_index("playerid")["library_size"].rename("library_size")

    # achievement_game_ratio: treat missing library as 0 owned games → denom = 1
    lib_filled = lib_size.reindex(total_achievements.index).fillna(0)
    ach_game_ratio = (games_with_ach / lib_filled.clip(lower=1)).rename("achievement_game_ratio")

    # Per-game concentration metrics
    game_counts = history.groupby(["playerid", "gameid"]).size()
    game_totals = game_counts.groupby(level=0).sum()
    game_props  = game_counts / game_totals

    top1_conc = game_props.groupby(level=0).max().rename("top1_game_concentration")
    top3_conc = (
        game_props.groupby(level=0)
        .apply(lambda x: x.nlargest(3).sum())
        .rename("top3_game_concentration")
    )
    game_hhi = (
        (game_props ** 2).groupby(level=0).sum().rename("game_hhi")
    )
    avg_ach_per_game = (total_achievements / games_with_ach).rename("avg_achievements_per_game")

    return pd.concat(
        [total_achievements, games_with_ach, lib_size,
         ach_game_ratio, top1_conc, top3_conc, game_hhi, avg_ach_per_game],
        axis=1,
    )
    
    
def _playtime_features(history: pd.DataFrame, purchased: pd.DataFrame) -> pd.DataFrame:
    """Group F: Playtime plausibility with async-safe achievement cross-referencing.

    Due to asynchronous Steam API crawling, the `history` and `purchased` tables
    can be captured at different points in time.  Naively treating every game in
    `history` as "owned" produces false positives.  This function classifies each
    achievement row into one of three mutually exclusive conditions:

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

    Features returned
    -----------------
    zero_playtime_achievements_ratio : |C| / (|B| + |C|)
        Fraction of in-library achievements that were unlocked with 0 playtime.
        Condition A rows are excluded from the denominator so async lag does not
        dilute the signal.
    playtime_per_achievement : Σplaytime_B_games / (|B| + |C|)
        Average minutes of playtime backing each in-library achievement.
        Low values combined with a high ratio above are a compound bot signal.
    total_playtime_mins : Σplaytime_B_games (per player)
        Sum of playtime over all Condition B games (distinct per player).

    Implementation note
    -------------------
    All row-level operations use `.merge()`, boolean indexing, and `.groupby()`
    on the purchased-library table (~100 K rows after explode) and on the
    history × library join.  No `.apply()` with custom functions iterates over
    the 9-million-row history DataFrame.
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
    """Group D: review behaviour signals.

    `review_unowned_ratio` was removed — Steam blocks reviews for unowned games
    so it mostly flagged private-profile players, not real bots.

    Replaced by `review_unplayed_ratio`: fraction of a player's reviews that
    are for games they own but have 0 playtime.  Bots typically bulk-review
    games they've never launched; legitimate reviewers almost always have some
    playtime.  Players not present in purchased_games receive NaN (the median
    imputer handles them downstream).
    """
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
    min_rev_len = rev_lens.groupby("playerid")["rlen"].min().rename("min_review_length")

    return pd.concat(
        [total_reviews, review_unplayed, review_dup, avg_rev_len, min_rev_len],
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
                          player_library: dict,
                          reference_time: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Step 3: Compute all per-player features across 6 groups (A–F).

    Early Trimming
    --------------
    The raw history table contains ~196 K distinct players, but ~193 K of them
    have fewer than 10 achievements and would be discarded by the downstream
    PU-learning filter anyway.  Computing Shannon entropy, inter-arrival
    standard deviation, HHI, and playtime cross-joins for those players wastes
    significant CPU and RAM.  Trimming to ≥10-achievement players *before* any
    heavy groupby/merge reduces the working set by ~98% and cuts wall-clock
    feature-engineering time proportionally.

    The threshold (10) matches the `heuristic_normal` definition in
    `build_heuristic_labels`, so trimming here is consistent with the label
    generation logic.
    """
    log.info("Step 3 — Building feature matrix …")

    # ── Early Trimming ────────────────────────────────────────────────────────
    ach_counts_all = history.groupby("playerid").size()
    core_ids = ach_counts_all[ach_counts_all >= 10].index
    log.info(
        "  Early trim: %d total players → %d with ≥10 achievements (%d dropped)",
        ach_counts_all.shape[0], len(core_ids),
        ach_counts_all.shape[0] - len(core_ids),
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
         "avg_review_length": 0.0,
         "min_review_length": 0.0}
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