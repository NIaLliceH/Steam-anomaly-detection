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
    """
    Return {playerid (int): set(gameid (int))} for O(1) library membership checks.

    Explicit int conversion prevents type mismatch between library items (may be
    stored as str in parquet) and reviews.gameid (int) — without this, isin()
    always returns False, causing review_unowned_ratio = 1.0 for every player.
    """
    result = {}
    for pid, lib in zip(purchased["playerid"], purchased["library"]):
        if lib is None:
            result[int(pid)] = set()
        else:
            result[int(pid)] = {int(g) for g in lib if g is not None}
    return result


# ---------------------------------------------------------------------------
# Step 2 — Heuristic Labels
# ---------------------------------------------------------------------------

def build_heuristic_labels(history: pd.DataFrame,
                            reviews: pd.DataFrame,
                            player_library: dict) -> pd.DataFrame:
    """
    Create pseudo ground truth (heuristic_bot = 0/1) using 3 AND-rule groups.

    V2 changes vs V1:
    - OR logic → AND logic per group to reduce false positives.
      V1's OR rules flagged 98%+ of players; AND rules target only clear bots.
    - 3 distinct bot archetypes: Speed bot, Volume bot, Review bot.
    - unowned_ratio uses explicit int() casts (Bug #2 fix).
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

    # ── Night activity ratio (00:00–05:59) ────────────────────────────────────
    night_ratio = (
        history.assign(is_night=(history["hour"] < 6))
        .groupby("playerid")["is_night"].mean()
        .rename("night_ratio")
    )

    # ── Achievement and review counts ─────────────────────────────────────────
    ach_counts    = history.groupby("playerid").size().rename("ach_count")
    review_counts = reviews.groupby("playerid").size().rename("review_count")

    # ── Unowned ratio (vectorised groupby — faster than row-level loop) ───────
    players_with_reviews = reviews["playerid"].unique()
    log.info("  Computing unowned_ratio for %d players with reviews …",
             len(players_with_reviews))

    reviews_sub = reviews[reviews["playerid"].isin(players_with_reviews)].copy()
    reviews_sub["gameid_int"] = reviews_sub["gameid"].apply(int)

    def _unowned(group):
        pid = int(group.name)
        if pid not in player_library:
            return np.nan  # no library data → don't flag as review_bot
        lib = player_library[pid]
        return (~group["gameid_int"].isin(lib)).mean()

    unowned_ratio = (
        reviews_sub.groupby("playerid")
        .apply(_unowned, include_groups=False)
        .rename("unowned_ratio")
    )

    df = pd.concat(
        [median_interval, min_interval, top1_conc, max_per_day,
         night_ratio, ach_counts, review_counts, unowned_ratio],
        axis=1,
    ).fillna({"review_count": 0, "night_ratio": 0.0, "min_interval": 0.0})
    # unowned_ratio NaN → review_bot condition (> 0.70) evaluates False → safe

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

    # ── Group 3: REVIEW BOT — reviews games they never played ─────────────────
    review_bot = (
        (df["review_count"]   > 5)          # enough sample to rule out coincidence
        & (df["ach_count"]   == 0)          # zero achievements (never actually played)
        & (df["unowned_ratio"] > 0.70)      # 70%+ reviews for games not in library
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

    # Fraction of reviews for unowned games.
    # Players absent from purchased_games have NO library data (private profile
    # or not crawled) — return NaN so the downstream SimpleImputer fills them
    # with the median of players who DO have library data.
    # Using set() as default would force ratio = 1.0 for ~75% of players,
    # artificially inflating the feature and dominating SHAP values.
    def unowned_ratio_fn(group):
        pid = int(group.name)
        if pid not in player_library:
            return np.nan
        lib = player_library[pid]
        return (~group["gameid"].apply(int).isin(lib)).mean()

    review_unowned = (
        reviews.groupby("playerid")
        .apply(unowned_ratio_fn, include_groups=False)
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
    # Players with no reviews get 0 for count/rate features.
    # review_unowned_ratio is intentionally left as NaN for:
    #   (a) players with no reviews (unknown — not 0, since 0 would mean "all owned")
    #   (b) players with reviews but absent from purchased_games (no library data)
    # The SimpleImputer downstream will fill both cases with the median of
    # players for whom we have valid library data.
    grp_d = grp_d.reindex(all_players).fillna(
        {"total_reviews": 0,
         "review_duplication_rate": 0.0,
         "avg_review_length": 0.0,
         "min_review_length": 0.0}
        # review_unowned_ratio: leave NaN — handled by imputer
    )

    log.info("  Bonus: account age features …")
    grp_bonus = _account_age_features(history, players)

    feature_matrix = pd.concat(
        [grp_a, grp_b, grp_c, grp_d, grp_bonus], axis=1
    ).reindex(all_players)

    log.info("  Feature matrix: %d players × %d features",
             *feature_matrix.shape)
    return feature_matrix
