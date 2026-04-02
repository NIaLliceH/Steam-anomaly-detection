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
# Pre-processing helpers (called after loading parquet in main.py)
# ---------------------------------------------------------------------------

def add_time_components(history: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day_of_week, date_only columns derived from date_acquired."""
    history = history.copy()
    history["hour"]        = history["date_acquired"].dt.hour.astype("int8")
    history["day_of_week"] = history["date_acquired"].dt.dayofweek.astype("int8")
    history["date_only"]   = history["date_acquired"].dt.date
    return history


def build_player_library(purchased: pd.DataFrame) -> dict:
    """Return {playerid: set(gameid)} for O(1) library membership checks."""
    return {
        pid: set(lib) if lib is not None else set()
        for pid, lib in zip(purchased["playerid"], purchased["library"])
    }


# ---------------------------------------------------------------------------
# Step 2 — Heuristic Labels
# ---------------------------------------------------------------------------

def build_heuristic_labels(history: pd.DataFrame,
                            reviews: pd.DataFrame,
                            player_library: dict) -> pd.DataFrame:
    """
    Create pseudo ground truth (heuristic_bot = 0/1) using 5 rules.
    Must be called before build_feature_matrix to keep the pipeline order
    explicit (avoids accidentally reusing fully-engineered features).
    """
    log.info("Step 2 — Building heuristic labels …")

    # Rule 1: Median unlock interval < 2 seconds
    speed_stats = (
        history.sort_values(["playerid", "date_acquired"])
        .groupby("playerid")["date_acquired"]
        .apply(lambda x: x.diff().dt.total_seconds().median())
        .rename("median_interval")
    )

    # Rule 2: Top-1 game concentration > 0.90
    top1_conc = (
        history.groupby(["playerid", "gameid"])
        .size()
        .groupby(level=0)
        .apply(lambda x: x.max() / x.sum())
        .rename("top1_concentration")
    )

    # Rule 3: Max achievements / day > 500
    max_per_day = (
        history.groupby(["playerid", "date_only"])
        .size()
        .groupby(level=0).max()
        .rename("max_per_day")
    )

    # Rules 4 & 5: Review-behaviour signals
    review_counts = reviews.groupby("playerid").size().rename("review_count")
    ach_counts    = history.groupby("playerid").size().rename("ach_count")

    # Unowned ratio: fraction of reviews for games not in the player's library
    players_with_reviews = reviews["playerid"].unique()
    log.info("  Computing unowned_ratio for %d players with reviews …",
             len(players_with_reviews))

    def calc_unowned_ratio(pid):
        pr = reviews[reviews["playerid"] == pid]
        if len(pr) == 0:
            return 0.0
        lib = player_library.get(pid, set())
        return (~pr["gameid"].isin(lib)).sum() / len(pr)

    unowned_ratio = pd.Series(
        {pid: calc_unowned_ratio(pid) for pid in players_with_reviews},
        name="unowned_ratio",
    )

    heuristic_df = pd.concat(
        [speed_stats, top1_conc, max_per_day, review_counts, ach_counts, unowned_ratio],
        axis=1,
    ).fillna(0)

    heuristic_df["heuristic_bot"] = (
        (heuristic_df["median_interval"] < 2)
        | (heuristic_df["top1_concentration"] > 0.90)
        | (heuristic_df["max_per_day"] > 500)
        | ((heuristic_df["review_count"] > 0) & (heuristic_df["ach_count"] == 0))
        | (heuristic_df["unowned_ratio"] > 0.80)
    ).astype(int)

    n = heuristic_df["heuristic_bot"].sum()
    log.info("  Heuristic bots: %d (%.2f%%)", n, heuristic_df["heuristic_bot"].mean() * 100)
    return heuristic_df[["heuristic_bot"]]


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


def _temporal_features(history: pd.DataFrame) -> pd.DataFrame:
    """Group B: time-of-day and activity-pattern features."""
    # Night activity ratio (00:00–05:59)
    night_counts = (
        history[history["hour"] < 6].groupby("playerid").size().rename("_night")
    )
    total_ach = history.groupby("playerid").size().rename("_total")
    night_ratio = (night_counts / total_ach).rename("night_activity_ratio").fillna(0)

    # Shannon entropy over 24-hour distribution
    hour_counts = (
        history.groupby(["playerid", "hour"])
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


def _review_features(reviews: pd.DataFrame, player_library: dict) -> pd.DataFrame:
    """Group D: review behaviour signals."""
    total_reviews = reviews.groupby("playerid").size().rename("total_reviews")

    # Fraction of reviews for unowned games
    def unowned_ratio_fn(group):
        lib = player_library.get(group.name, set())
        return (~group["gameid"].isin(lib)).mean()

    review_unowned = (
        reviews.groupby("playerid")
        .apply(unowned_ratio_fn)
        .rename("review_unowned_ratio")
    )

    # Duplicate review rate (case-insensitive, whitespace-stripped)
    review_dup = (
        reviews.groupby("playerid")["review"]
        .apply(lambda s: s.fillna("").str.lower().str.strip().duplicated().sum() / len(s))
        .rename("review_duplication_rate")
    )

    rev_lens = reviews.assign(rlen=reviews["review"].fillna("").str.len())
    avg_rev_len = rev_lens.groupby("playerid")["rlen"].mean().rename("avg_review_length")
    min_rev_len = rev_lens.groupby("playerid")["rlen"].min().rename("min_review_length")

    return pd.concat(
        [total_reviews, review_unowned, review_dup, avg_rev_len, min_rev_len],
        axis=1,
    )


def _account_age_features(history: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Bonus: account age and time-to-first-achievement."""
    player_created = players.set_index("playerid")["created"]
    first_ach      = history.groupby("playerid")["date_acquired"].min()

    days_before_first = (
        (first_ach - player_created)
        .dt.days
        .clip(lower=0)
        .rename("days_before_first_achievement")
    )
    account_age = (
        (pd.Timestamp.now() - player_created)
        .dt.days
        .rename("account_age_days")
    )
    return pd.concat([days_before_first, account_age], axis=1)


def build_feature_matrix(history: pd.DataFrame,
                          reviews: pd.DataFrame,
                          players: pd.DataFrame,
                          purchased: pd.DataFrame,
                          player_library: dict) -> pd.DataFrame:
    """
    Step 3: Compute all 23 per-player features across 5 groups.
    Players appearing only in reviews (0 achievements) get NaN for
    Groups A–C, which the median imputer handles downstream.
    """
    log.info("Step 3 — Building feature matrix …")

    all_players = pd.Index(
        sorted(set(history["playerid"].unique()) | set(reviews["playerid"].unique())),
        name="playerid",
    )

    log.info("  Group A: speed features …")
    grp_a = _speed_features(history)

    log.info("  Group B: temporal features …")
    grp_b = _temporal_features(history)

    log.info("  Group C: diversity features …")
    grp_c = _diversity_features(history, purchased)

    log.info("  Group D: review features …")
    grp_d = _review_features(reviews, player_library)
    # Players with no reviews default to 0 (not NaN) for review features
    grp_d = grp_d.reindex(all_players).fillna(
        {"total_reviews": 0, "review_unowned_ratio": 0.0,
         "review_duplication_rate": 0.0, "avg_review_length": 0.0,
         "min_review_length": 0.0}
    )

    log.info("  Bonus: account age features …")
    grp_bonus = _account_age_features(history, players)

    feature_matrix = pd.concat(
        [grp_a, grp_b, grp_c, grp_d, grp_bonus], axis=1
    ).reindex(all_players)

    log.info("  Feature matrix: %d players × %d features",
             *feature_matrix.shape)
    return feature_matrix
