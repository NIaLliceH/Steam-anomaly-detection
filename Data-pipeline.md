# Data pipeline

## Phase 1: Data preparation - `src/data_prep.py`
### Helpers
- `_parse_list_fast()`: parse library field — hỗ trợ cả 2 format:
  - Cũ: `[10, 20, 30]` → `[{"appid": 10, "playtime_mins": -1}, ...]`
  - Mới: `[{"appid": 10, "playtime_mins": 0}, ...]`
- `_read_purchased_robust()`: đọc purchased CSV có malformed rows (unescaped inner quotes) bằng cách repair thủ công từng dòng lỗi
- `_merge_with_crawled()`: nếu tồn tại `data/crawled/<filename>`, concat vào df_raw trong RAM (không sửa file gốc)

### Load and clean
- `load_private_ids()`: đọc `private_steamids.csv` → set playerid cần lọc
- `load_history()`:
  - read_csv
  - merge_with_crawled
  - extract gameid từ achievementid bằng regex `^(\d+)_`
  - datetime typecast
  - remove private player
  - drop duplicate (subset: playerid, achievementid, date_acquired)
- `load_players()`:
  - read_csv
  - merge_with_crawled
  - datetime typecast (cột `created`)
  - remove private player
  - drop duplicate (subset: playerid)
- `load_reviews()`:
  - read_csv
  - merge_with_crawled
  - datetime typecast (cột `posted`)
  - remove private player
  - drop duplicate (subset: reviewid)
- `load_purchased()`:
  - `_read_purchased_robust()` thay vì read_csv thông thường
  - merge crawled/purchased_games.csv nếu có
  - parse library field bằng `_parse_list_fast()` → list of dicts
  - thêm cột `library_size` = len(library)
  - remove private player
  - drop duplicate (subset: playerid)

### Export parquet
Output: `data/processed/*.parquet` (history, players, reviews, purchased)


## Phase 2: ML Model - `main.py`
### Load and Pre-process
- `load_parquets()` → history, players, reviews, purchased
- `add_time_components()` → thêm cols `hour`, `day_of_week`, `date_only` cho history
- `feature_reference_time` = max(date_acquired) → mốc thời gian cố định để tính `account_age_days`

### Feature engineering - `features.py`
#### Trimming (trong `build_feature_matrix`, trước khi tính bất kỳ feature nào)
- Điều kiện: **(≥10 achievements OR ≥3 reviews) AND library_size ≥ 1**
  - OR logic: giữ lại Review Bot candidates có 0 achievement nhưng spam review
  - `library_size ≥ 1`: loại ghost accounts không có game nào (sẽ sinh NaN-only feature rows)
- Giảm ~98% working set (~196K → ~3K players), tăng tốc toàn bộ groupby/merge
- Thực hiện một lần duy nhất; không còn bước trimming thứ hai trong `main.py`

#### Group A: `_speed_features()`
- `median_unlock_interval_sec`, `std_unlock_interval_sec`
- `cv_unlock_interval` = std / mean (coefficient of variation)
- `max_achievements_per_minute`
- `max_achievements_per_day`

#### Group B: `_temporal_features()`
- `night_activity_ratio` → local_hour < 6 (timezone-aware, dùng `_COUNTRY_UTC_OFFSET`)
- `hour_entropy` → Shannon entropy phân phối achievement theo 24 giờ (local time)
- `activity_density` → active_days / calendar_span (ngày đầu đến ngày cuối)

#### Group C: `_diversity_features()`
- `total_achievements`
- `library_size`
- `achievement_game_ratio` → games_with_achievements / library_size *(games_with_achievements tính nội bộ, không export)*
- `top1_game_concentration` → % achievement từ game có nhiều nhất
- `top3_game_concentration` → % achievement từ 3 game nhiều nhất
- `game_hhi` → Herfindahl index = Σ(game_proportion²)
- `avg_achievements_per_game` → total_achievements / games_with_achievements

#### Group D: `_review_features()`
- `total_reviews`
- `review_unplayed_ratio` → fraction reviews cho game có playtime == 0
- `review_duplication_rate` → 1 − (unique_texts / total_reviews)
- `avg_review_length`, `min_review_length`

#### Group E: `_account_age_features()`
- `days_before_first_achievement` → first_achievement_date − account_created_date
- `account_age_days` → feature_reference_time − account_created_date

#### Group F: `_playtime_features()`
- Left-join history với purchased library:
  - **Condition A**: game có trong history nhưng không có trong library → API lag → bỏ qua
  - **Condition B**: game trong library với playtime > 0 → gameplay bình thường
  - **Condition C**: game trong library với playtime == 0 → bot signal (SAM/unlocker)
- `zero_playtime_achievements_ratio` = C / (B + C)
- `total_playtime_mins` = tổng playtime của các game thuộc B (distinct game, tránh inflate)
- `playtime_per_achievement` = total_playtime_mins / (B + C)

**Tổng cộng: 25 features**


### Heuristic labels - `features.py`
`build_heuristic_labels(feature_matrix)` — đọc trực tiếp từ các cột của feature_matrix (không tính lại từ raw tables).

- **Speed bot**: `median_unlock_interval_sec < 10s` AND `top1_game_concentration > 0.85`
- **Volume bot**: (`max_per_day > 500` AND `night_activity_ratio > 0.40` AND `total_achievements > 1000`) OR `zero_playtime_achievements_ratio > 0.9` (SAM catch-all)
- **Review bot**: `total_reviews > 5` AND `total_achievements == 0` AND `review_unplayed_ratio > 0.50` AND `review_duplication_rate > 0.50` AND `avg_rev_len` < 50
- `heuristic_bot` = speed_bot | volume_bot | review_bot
- `heuristic_normal` → `total_achievements > 10` AND `median_unlock_interval_sec > 600s` (10 min, dùng cho PU Learning)

`integrate_human_labels()` → override heuristic_bot/normal bằng nhãn từ `data/reviewed.csv` (HITL loop)


## Preprocessing - `models.py`
Hai paths riêng biệt cho IF và XGBoost:

**Path A — IsolationForest**: `apply_log_transform()` → `preprocess()` (SimpleImputer + StandardScaler)
- `apply_log_transform()`: `log1p(clip(x, 0))` cho 9 heavy-tailed features
  - Lý do: **IsolationForest path-length stability** — feature spanning 6+ bậc độ lớn làm uniform random split không đại diện cho mật độ dữ liệu thực; log1p cân bằng lại
- `preprocess()`: Pipeline `SimpleImputer(median)` → `StandardScaler` → trả về `X_if` dạng DataFrame; lưu `preprocessor.pkl`

**Path B — XGBoost**: `X_raw` trực tiếp (không qua impute, không qua scale)
- XGBoost xử lý NaN natively qua sparsity-aware split finding
- NaN trong review features (player không có review) là **tín hiệu thực sự**, không phải gap cần fill
- Invariant với monotonic transform → log1p không có tác dụng


## Tuning & Training
- `tune_models()` → grid search trên IsolationForest, tối đa hoá ROC-AUC so với y_heuristic → `best_if_params`
- `train_best_models()` → train IF với `best_if_params`, trả về `(models, scores)`, lưu `best_if.pkl`
- `train_xgboost_semisupervised(X_raw, y_heuristic, y_normal, ...)`
  - Chỉ train trên **confident subset**: heuristic_bot == 1 OR heuristic_normal == 1 (bỏ grey area)
  - `scale_pos_weight` = neg_count / pos_count (xử lý class imbalance)
  - `RandomizedSearchCV` (n_iter=50, cv=5, metric=PR-AUC) → `best_xgb`
  - Predict_proba chạy trên **toàn bộ** X_raw (không chỉ training subset)
  - Lưu `best_xgb.pkl`, `xgb_tuning_results.csv`


## Build Ensemble - `models.py`
- `build_ensemble()`
  - `if_pct` → percentile rank IF scores (0–100)
  - `xgb_pct` → percentile rank XGBoost probabilities (0–100)
  - `composite = 0.70 * xgb_pct + 0.30 * if_pct`
  - `is_anomaly` = composite >= 85
  - Log IF ROC-AUC để theo dõi chất lượng (không còn auto-flip)
- `tune_ensemble_weights()` *(bước phân tích thêm)*
  - Sweep `xgb_weight` từ 0.0 → 1.0 (step=0.05), tính 3 metrics tại mỗi điểm:
    - `Precision@100`, `PR-AUC`, `High_Conflict_Cases`
  - Xác định "optimal sweet spot": argmax Precision@100 trong tập HCC ≥ median(HCC)
  - Lưu `ensemble_weight_metrics.csv`, `plots/ensemble_weight_tuning.png`


## Active Learning - `active_learning.py`
- `generate_review_sample()`
  - conflict = model says bot (composite >= 85) AND heuristic says normal (heuristic_bot == 0)
  - sort by composite_score desc → top 50
  - merge với feature_matrix → reviewer có đủ context để gán nhãn
  - lưu → `outputs/to_review.csv`
  - Human fill `human_label` (1=bot, 0=normal) → save as `data/reviewed.csv` → re-run pipeline
- `integrate_human_labels()` *(gọi đầu pipeline, trước modelling)*
  - Override heuristic_bot/normal bằng nhãn đã review
  - Chỉ áp dụng cho rows có human_label hợp lệ (0 hoặc 1), bỏ qua blank


## Evaluation - `evaluate.py`
- `evaluate()`
  - `_model_comparison()`:
    - Tính ROC-AUC, PR-AUC, Flagged Rate%, Precision@100/500/1000 cho 3 models (XGBoost, IsolationForest, Ensemble)
    - Lưu `model_comparison.csv`
  - `_evaluate_xgboost()`:
    - PR-AUC, optimal threshold (argmax F1)
    - `classification_report` tại threshold đó
    - Lưu `xgb_pr_curve.png`, `xgb_feature_importance.png`
  - So sánh mean features: top-50 flagged vs normal → lưu `top50_flagged_profiles.csv`
- `_shap_plots()`:
  - Sample 5000 rows ngẫu nhiên từ `X_raw` (unscaled — giữ axis values trên thang đo gốc, dễ đọc)
  - Tính SHAP values via native XGBoost `booster.predict(pred_contribs=True)` (bypass TreeExplainer parser)
  - Lưu `shap_summary.png` → global feature importance
  - Lưu `shap_waterfall.png` → player đáng ngờ nhất trong sample
  - Lưu `shap_scatter_<feat>.png` cho top-3 features quan trọng nhất
