# Data pipeline

## Phase 1: Data preparation - `src/data_prep.py`
### Load and clean
- load_private_ids(): read_csv
- load_history():
  - read_csv
  - merge_with_crawled
  - extract gameid from achievementid
  - datetime typecast
  - remove private player
  - drop duplicate
- load_players():
  - read_csv
  - merge_with_crawled
  - datetime typecast
  - remove private player
  - drop duplicate
- load_reviews():
  - read_csv
  - merge_with_crawled
  - datetime typecast
  - remove private player
  - drop duplicate
- load_purchased():
  - parse library field -> [{"appid":10, "playetime_mins":0},...]
  - remove private player
  - drop duplicate

### Export parquet
Output: data/processed/*.parquet


## Phase 2: ML Model - `main.py`
### Load and Pro-process
- load_parquets()
- add_time_componets() -> add cols hour, day_of_week, date_only for table history
- feature_reference_time = max(date_acquired) -> start point to calc account_age_days
- build_player_library -> store purchased as dict for O(1) lookup

### Heuristic labels - `features.py`
- build_heuristic_labels
  - calc median_interval, min_interval from history
  - top1_concentration -> max achievement per game / sum achievement
  - max_per_day -> max achievement per day
  - night_ratio -> 0-6 AM local time
  - ach_counts
  - review_counts
  - review_signal = review_unplayed_ratio