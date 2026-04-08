# Steam Account Anomaly Detection System

> **Hệ thống phát hiện bất thường sử dụng Dynamic Duo — Semi-supervised XGBoost (primary) + IsolationForest (secondary) — để nhận diện bot accounts và hoạt động gian lận trên nền tảng Steam.**

---

## Bài toán

Steam là nền tảng game lớn nhất thế giới với hàng triệu tài khoản người dùng. Trong đó tồn tại các tài khoản có hành vi bất thường:

- **Speed Bot**: Tài khoản dùng script unlock thành tích trong <1 giây, tập trung vào 1 game
- **Volume Bot**: Tài khoản unlock >500 thành tích/ngày và hoạt động ban đêm bất thường
- **Review Bot**: Tài khoản đăng review cho game chưa từng mua/sở hữu

Dự án sử dụng **Semi-supervised Ensemble (Dynamic Duo)** gồm XGBoost (primary, weight 0.80) + IsolationForest (secondary, weight 0.20) để phân tích behavioral patterns từ dữ liệu lịch sử thành tích, thư viện game, review, và thời gian chơi của từng người chơi.

### Thách thức chính

| Thách thức | Cách xử lý |
|---|---|
| Không có ground truth | Heuristic labels V2 với AND-logic 3 nhóm bot rõ ràng |
| Class imbalance cực độ | PU Learning — loại grey area khỏi tập train XGBoost |
| Ghost accounts (~75%) | Data trimming: chỉ giữ players có ≥ 10 thành tích và ≥ 1 game |
| PCA collapse do heavy tails | log1p transform 10 features trước preprocessing |
| IQR ≈ 0 của ghost accounts | Thay RobustScaler bằng StandardScaler sau log1p |
| review_unowned_ratio bị inflate | Trả NaN thay vì set() rỗng cho player không có library data |
| Playtime coverage thấp | playtime_per_achievement = NaN nếu thiếu purchased data |

---

## Kiến trúc Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1 — DATA ETL  (src/data_prep.py)                     │
│                                                             │
│  Raw CSV (~1.5 GB, ~196k players)                           │
│    history.csv, players.csv, reviews.csv,                   │
│    purchased_games.csv, private_steamids.csv                │
│                                ↓                            │
│  • Lọc private accounts (227k IDs)                          │
│  • Parse datetime với explicit format                       │
│  • Extract gameid từ achievementid string (regex)           │
│  • Parse library column → list of dicts                     │
│    [{"appid": int, "playtime_mins": int}]                   │
│  • Dedup + type-optimize → export Parquet                   │
│                                                             │
│  Output: data/processed/{history,players,reviews,           │
│          purchased}.parquet                                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2 — ML PIPELINE  (main.py)                           │
│                                                             │
│  Step 2: Heuristic Labels V2  (features.py)                 │
│    • 3 nhóm bot — AND logic (không OR):                     │
│      - Speed Bot: median_interval<10s AND min<1s            │
│                   AND top1>85%                              │
│      - Volume Bot: max_per_day>500 AND night>40%            │
│                    AND ach>1k                               │
│      - Review Bot: reviews>5 AND 0 ach AND unowned>70%     │
│    • Strict Normal: ach_count>10 AND median_interval>1800s  │
│    Output: heuristic_labels.csv                             │
│                                                             │
│  Step 3: Feature Engineering  (features.py)                 │
│    • 27 features / 6 nhóm:                                  │
│      A. Speed (6): median/min/std interval, cv,             │
│                    max_per_min, max_per_day                 │
│      B. Temporal (4): night ratio, hour entropy,            │
│                       density, weekend ratio                │
│      C. Diversity (8): total ach, games w/ ach, library     │
│                        size, ach/game ratio, top1/top3/HHI, │
│                        avg_ach_per_game                     │
│      D. Review (5): count, unowned_ratio*, dup rate,        │
│                     avg/min length                          │
│      E. Account Age (2): days_before_first_ach, account_age │
│      F. Playtime (2): total_playtime_mins,                  │
│                       playtime_per_achievement*             │
│    • *NaN nếu không có purchased data                       │
│    Output: feature_matrix.csv                               │
│                                                             │
│  DATA TRIMMING  (main.py)                                   │
│    • Loại ghost/sparse accounts:                            │
│      total_achievements >= 10 AND library_size >= 1         │
│    • ~196k → ~40-60k players (active accounts only)         │
│                                                             │
│  Step 4: Preprocessing  (models.py)                         │
│    1. log1p(clip(x, 0)) cho 10 heavy-tailed features        │
│    2. SimpleImputer(strategy="median")                       │
│    3. StandardScaler  (thay RobustScaler — IQR≈0 bug)       │
│    4. PCA(n_components=0.90)  giữ 90% variance              │
│    → Log per-component variance để verify không collapse    │
│    Output: preprocessor.pkl                                 │
│                                                             │
│  Step 5a: IF Tuning  (models.py)                            │
│    • Grid search IsolationForest (36 combos):               │
│      - n_estimators: [100, 200, 300]                        │
│      - max_samples: ["auto", 0.8, 0.6]                      │
│      - contamination: [0.02, 0.05, 0.10]                    │
│      - max_features: [0.8, 1.0]                             │
│    • Scoring: ROC-AUC vs heuristic labels                   │
│    Output: tuning_results.csv                               │
│                                                             │
│  Step 5b: XGBoost — PRIMARY MODEL  (models.py)              │
│    • PU Learning filter:                                    │
│      train chỉ trên: heuristic_bot==1 OR heuristic_normal==1│
│      loại grey area (không rõ bot hay người thật)           │
│    • scale_pos_weight = confirmed_normal / bot              │
│    • RandomizedSearchCV(n_iter=50, cv=5, scoring=PR-AUC)    │
│    • predict_proba() chạy trên TOÀN BỘ trimmed dataset     │
│    Output: best_xgb.pkl, xgb_tuning_results.csv            │
│                                                             │
│  Step 6: Train Final IsolationForest  (models.py)           │
│    • Retrain IF với best params trên toàn bộ PCA data       │
│                                                             │
│  Step 7: Dynamic Duo Ensemble  (models.py)                  │
│    • Percentile rank mỗi model (0–100)                      │
│    • IF auto-flip: nếu ROC-AUC < 0.4 → đảo score           │
│    • Weighted composite:                                    │
│        composite = 0.80 × XGB_pct + 0.20 × IF_pct          │
│    • is_anomaly = 1 nếu composite_score ≥ 85               │
│    Output: ensemble_results.csv                             │
│                                                             │
│  Step 8: Evaluation  (evaluate.py)                          │
│    • ROC-AUC và PR-AUC cho XGBoost, IF, Ensemble            │
│    • Precision@100/500/1000                                 │
│    • Top-50 flagged vs normal feature profile               │
│    • XGBoost: PR curve, feature importance                  │
│    Output: model_comparison.csv, top50_flagged_profiles.csv │
│                                                             │
│  Step 9: SHAP Explanations  (evaluate.py)                   │
│    • Native XGBoost SHAP (pred_contribs=True)               │
│    • Summary bar plot, waterfall top-1, scatter top-3       │
│    Output: plots/shap_*.png                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Cấu trúc thư mục

```
btl/
├── batch_analysis.py        # Entry point: inject data + chạy pipeline + report
├── main.py                  # ML pipeline orchestrator (Steps 2-9)
├── steam_crawling.py        # Script crawl dữ liệu Steam
├── targeted_crawler.py      # Crawl cho danh sách player cụ thể
├── merge_data.py            # Gộp file dữ liệu đã crawl
├── README.md
├── dataset_structure.md     # Mô tả format dữ liệu thô
│
├── src/
│   ├── data_prep.py         # Phase 1: ETL, clean, export Parquet
│   ├── features.py          # Step 2-3: Heuristic labels + feature engineering
│   ├── models.py            # Step 4-7: Preprocessing, training, ensemble
│   └── evaluate.py          # Step 8-9: Metrics, plots, SHAP
│
├── data/
│   ├── raw/                 # CSV gốc
│   ├── processed/           # Parquet (output của data_prep.py)
│   ├── crawled/             # Data mới để inject
│   └── archive/             # Backup theo timestamp
│
└── outputs/
    ├── feature_matrix.csv
    ├── heuristic_labels.csv
    ├── tuning_results.csv
    ├── xgb_tuning_results.csv
    ├── ensemble_results.csv
    ├── model_comparison.csv
    ├── top50_flagged_profiles.csv
    ├── best_xgb.pkl
    ├── preprocessor.pkl
    └── plots/
        ├── shap_summary.png
        ├── shap_waterfall.png
        ├── shap_scatter_*.png
        ├── xgb_pr_curve.png
        └── xgb_feature_importance.png
```

---

## Input / Output

### Input — Raw CSV

| File | Kích thước | Nội dung |
|------|-----------|----------|
| `history.csv` | ~647 MB | playerid, achievementid, date_acquired |
| `reviews.csv` | ~551 MB | playerid, gameid, review text, posted date |
| `players.csv` | ~18 MB | playerid, country, created date |
| `purchased_games.csv` | ~92 MB | playerid, library (list of dicts với appid + playtime_mins) |
| `private_steamids.csv` | ~4 MB | playerid của tài khoản private |

**Vị trí**: `data/raw/`

### Output chính — `ensemble_results.csv`

| Cột | Mô tả |
|-----|-------|
| `playerid` | Steam player ID |
| `composite_score` | Điểm nghi vấn tổng hợp 0–100 (cao = đáng ngờ hơn) |
| `is_anomaly` | 1 = bị flag (composite ≥ 85), 0 = bình thường |
| `xgb_proba` | Xác suất bot theo XGBoost (0–1) |
| `xgb_pct` | Percentile rank XGBoost (0–100) |
| `if_pct` | Percentile rank IsolationForest (0–100, sau auto-flip nếu cần) |
| `heuristic_bot` | Nhãn heuristic (pseudo ground truth) |

---

## Cách chạy

### Chạy toàn bộ pipeline

```bash
python3 batch_analysis.py
```

Script sẽ:
1. Inject data mới từ `data/crawled/` vào `data/raw/`
2. Chạy Phase 1 (data_prep.py)
3. Chạy Phase 2 (main.py)
4. Tạo report cho `TARGET_STEAM_IDS`

Chỉnh danh sách account cần kiểm tra trong `batch_analysis.py`:
```python
TARGET_STEAM_IDS = [76561198287996067, ...]
```

### Tùy chọn CLI

```bash
python3 batch_analysis.py --query-only                  # Chỉ tạo report, bỏ qua pipeline
python3 batch_analysis.py --steam-ids ID1 ID2 ...       # Query player ID cụ thể
python3 batch_analysis.py --format markdown             # Report format: markdown | html | all
python3 batch_analysis.py --force-run                   # Chạy pipeline dù không có data mới
```

### Chạy từng bước

```bash
# Phase 1: ETL
python3 src/data_prep.py

# Phase 2: ML Pipeline
python3 main.py
```

### Tạo report

```python
from batch_analysis import generate_report

generate_report(TARGET_STEAM_IDS, output_format='console')   # In ra terminal
generate_report(TARGET_STEAM_IDS, output_format='markdown')  # outputs/detection_report.md
generate_report(TARGET_STEAM_IDS, output_format='html')      # outputs/detection_report.html
generate_report(TARGET_STEAM_IDS, output_format='all')       # Cả 3 format
```

---

## Feature Engineering — 27 features

### Group A — Tốc độ unlock (6 features)
| Feature | Ý nghĩa |
|---------|---------|
| `median_unlock_interval_sec` | Thời gian trung vị giữa 2 lần unlock liên tiếp |
| `min_unlock_interval_sec` | Khoảng cách nhỏ nhất — phát hiện unlock tức thì |
| `std_unlock_interval_sec` | Độ lệch chuẩn — bots thường đều đặn bất thường |
| `cv_unlock_interval` | Hệ số biến thiên (std/mean) |
| `max_achievements_per_minute` | Tốc độ đỉnh theo phút |
| `max_achievements_per_day` | Tốc độ đỉnh theo ngày |

### Group B — Temporal (4 features)
| Feature | Ý nghĩa |
|---------|---------|
| `night_activity_ratio` | Tỷ lệ hoạt động 00:00–05:59 |
| `hour_entropy` | Shannon entropy phân phối 24h (thấp = pattern cứng nhắc) |
| `activity_density` | Số ngày active / tổng span ngày |
| `weekend_ratio` | Tỷ lệ hoạt động cuối tuần |

### Group C — Game diversity (8 features)
| Feature | Ý nghĩa |
|---------|---------|
| `total_achievements` | Tổng thành tích |
| `games_with_achievements` | Số game khác nhau có thành tích |
| `library_size` | Số game trong thư viện |
| `achievement_game_ratio` | Tỷ lệ games có ach / library |
| `top1_game_concentration` | % thành tích đến từ 1 game |
| `top3_game_concentration` | % thành tích đến từ 3 game |
| `game_hhi` | Herfindahl index — đo mức độ tập trung |
| `avg_achievements_per_game` | Trung bình ach mỗi game |

### Group D — Review behavior (5 features)
| Feature | Ý nghĩa |
|---------|---------|
| `total_reviews` | Số review đã đăng |
| `review_unowned_ratio` | % review cho game không có trong library (NaN nếu không có library data) |
| `review_duplication_rate` | Tỷ lệ review trùng lặp (sau lowercase + strip) |
| `avg_review_length` | Độ dài trung bình review (ký tự) |
| `min_review_length` | Độ dài review ngắn nhất |

### Group E — Account age (2 features)
| Feature | Ý nghĩa |
|---------|---------|
| `days_before_first_achievement` | Số ngày từ tạo account đến lần unlock đầu tiên |
| `account_age_days` | Tuổi account tính đến ngày chạy |

### Group F — Playtime (2 features)
| Feature | Ý nghĩa |
|---------|---------|
| `total_playtime_mins` | Tổng thời gian chơi (phút) trong thư viện (NaN nếu không có data) |
| `playtime_per_achievement` | Thời gian chơi / tổng thành tích (NaN nếu thiếu playtime hoặc ach = 0) |

---

## Heuristic Labels V2

Nhãn heuristic dùng làm pseudo ground truth để tuning và đánh giá (không phải ground truth thực sự).

### Nhóm bot (heuristic_bot = 1)

Dùng AND logic để giảm false positive:

**Speed Bot** — Bot unlock nhanh bất thường:
```
median_interval < 10s AND min_interval < 1s AND top1_concentration > 85%
```

**Volume Bot** — Bot khối lượng lớn + hoạt động ban đêm:
```
max_per_day > 500 AND night_ratio > 40% AND ach_count > 1000
```

**Review Bot** — Bot review game chưa chơi:
```
review_count > 5 AND ach_count == 0 AND unowned_ratio > 70%
```

### Người dùng bình thường (heuristic_normal = 1)

Dùng cho PU Learning — chỉ bao gồm người rõ ràng không phải bot:
```
ach_count > 10 AND median_interval > 1800s (30 phút)
```

---

## Preprocessing Pipeline

```
X_raw (27 features)
    ↓
log1p(clip(x, 0)) cho 10 heavy-tailed features:
  total_achievements, max_achievements_per_day,
  max_achievements_per_minute, median_unlock_interval_sec,
  min_unlock_interval_sec, std_unlock_interval_sec,
  avg_achievements_per_game, library_size,
  total_reviews, avg_review_length
    ↓
SimpleImputer(strategy="median")   ← fill NaN bằng median
    ↓
StandardScaler()   ← z-score (tốt hơn RobustScaler sau log1p)
    ↓
PCA(n_components=0.90)   ← giữ 90% variance → ~8-12 components
    ↓
X_scaled (PCA components)
```

**Lý do log1p**: `std_unlock_interval_sec` có thể lên đến hàng triệu giây. Không có transform → PCA collapse: PC1 giải thích >92% variance → IF/XGBoost chạy trên dữ liệu gần 1D.

**Lý do StandardScaler thay RobustScaler**: Ghost accounts (0 achievements) chiếm ~75% dataset — IQR của hầu hết features ≈ 0 → RobustScaler scale active players lên giá trị cực lớn.

---

## Models — Dynamic Duo

### XGBoost — Primary (weight 0.80)

- **Training set**: chỉ `heuristic_bot==1` OR `heuristic_normal==1` (PU Learning — loại grey area)
- **scale_pos_weight** = confirmed_normal / bot (xử lý imbalance)
- **Tuning**: RandomizedSearchCV(n_iter=50, cv=5, scoring=average_precision)
- **Inference**: predict_proba() trên toàn bộ trimmed dataset

### IsolationForest — Secondary (weight 0.20)

- Phát hiện statistical outliers trong không gian PCA (unsupervised)
- Grid search: 36 combos (n_estimators × max_samples × contamination × max_features)
- Auto-flip: nếu ROC-AUC < 0.4 → đảo score (inversion detection)

### Ensemble

```
xgb_pct = percentile_rank(xgb_proba)         # 0–100
if_pct  = percentile_rank(if_scores)          # 0–100, sau auto-flip nếu cần

composite_score = 0.80 × xgb_pct + 0.20 × if_pct

is_anomaly = 1  nếu composite_score ≥ 85
```

LOF và OneClassSVM đã bị loại bỏ: LOF có độ phức tạp O(N²) không khả thi với dataset lớn; OCSVM có weight quá thấp để đóng góp có ý nghĩa.

---

## Dependencies

```
pandas
numpy
scikit-learn
scipy
xgboost
shap
matplotlib
joblib
pyarrow          # đọc/ghi Parquet
```

Cài đặt: `pip install -r requirements.txt`

---

## Giới hạn đã biết

1. **Target leakage**: Heuristic labels được tính từ các features cũng dùng để train — metrics đo "model có bắt chước heuristic rules không", không phải "model phát hiện bot thực sự tốt đến đâu"
2. **Không có ground truth**: Không có nhãn thực xác nhận tài khoản nào là bot
3. **purchased_games coverage thấp**: ~75% người chơi không có dữ liệu purchased_games → `review_unowned_ratio` và `playtime_per_achievement` bị impute bằng median
4. **Ghost accounts**: ~75% trong 196k players có <10 thành tích và bị loại ở bước data trimming — chỉ ~40–60k accounts được đưa vào model

---

## Crawl thêm dữ liệu purchased_games

Dữ liệu gốc chỉ có purchased_games cho ~25% player — phần còn lại là tài khoản private hoặc chưa được crawl. Điều này làm cho `review_unowned_ratio` và `playtime_per_achievement` bị impute thay vì dùng giá trị thực, giảm độ tin cậy của 2 features này.

Có 2 script để bổ sung dữ liệu:

### 1. `steam_crawling.py` — Crawl đầy đủ cho vài account cụ thể

Crawl toàn bộ 4 loại dữ liệu (players, purchased_games, history, reviews) cho danh sách account nhỏ qua Steam Web API + HTML scraping.

```bash
# Yêu cầu: API_KEY trong file .env
echo "API_KEY=your_key_here" > .env

# Crawl toàn bộ data (mặc định dùng STEAM_IDS trong file)
python3 steam_crawling.py

# Crawl account cụ thể
python3 steam_crawling.py 76561198405841744

# Crawl nhiều account
python3 steam_crawling.py --steam-ids 76561198405841744 76561198354838543

# Chỉ crawl purchased_games và history
python3 steam_crawling.py --data purchased_games,history
```

**Loại dữ liệu có thể crawl:** `players`, `purchased_games`, `history`, `reviews`

Dữ liệu được lưu vào `data/crawled/` và tự động nối vào file gốc khi chạy `batch_analysis.py`.

> **Lưu ý**: Steam bị chặn ở một số vùng mạng — nên dùng VPN. Script tự động rate-limit (0.2s/request cho achievements, 0.5s/trang cho reviews) để tránh HTTP 429.

---

### 2. `targeted_crawler.py` — Crawl có ưu tiên cho ~10,000 player

Script chuyên dụng để bổ sung `purchased_games` (có `playtime_mins`) cho một tập con được chọn lọc từ kết quả model hiện tại. Dùng khi muốn cải thiện coverage cho `review_unowned_ratio` và `playtime_per_achievement` ở quy mô lớn hơn.

**Thứ tự ưu tiên chọn player:**

| Ưu tiên | Tiêu chí | Lý do |
|---------|---------|-------|
| 1 | `heuristic_bot==1` hoặc `heuristic_normal==1` | PU training set — data chất lượng cao nhất |
| 2 | `is_anomaly==1` hoặc `composite_score>=80` | Grey area cần xác nhận |
| 3 | Còn lại (random baseline) | Đủ quota để đánh giá |

**Quota mặc định**: 10,000 player. Thay đổi trong file:
```python
TARGET_QUOTA = 10_000
```

**Chạy crawler:**

```bash
# Yêu cầu: đã chạy main.py trước (cần heuristic_labels.csv + ensemble_results.csv)
python3 targeted_crawler.py
```

Script tự động:
- Đọc `outputs/heuristic_labels.csv` và `outputs/ensemble_results.csv` để chọn danh sách player
- Crawl `GetOwnedGames` (Steam API) lấy `appid` + `playtime_forever` cho từng player
- Ghi kết quả vào `data/crawled/targeted_purchased_games.csv` theo JSON format: `[{"appid": int, "playtime_mins": int}, ...]`
- Lưu checkpoint vào `data/crawled/processed_ids.txt` — có thể dừng giữa chừng và chạy lại, sẽ tiếp tục từ chỗ dừng

**Rate limiting tích hợp:**
- 1.1s giữa mỗi request
- Tự động sleep 60s khi gặp HTTP 429
- Flush dữ liệu ra disk sau mỗi 100 records

**Merge vào dataset gốc:**

```bash
python3 merge_data.py
```

Script đọc `data/crawled/targeted_purchased_games.csv`, nối với `data/raw/purchased_games.csv`, dedup theo `playerid` (ưu tiên data mới), và ghi đè lại file raw.

**Workflow hoàn chỉnh:**

```
1. python3 main.py                    # Chạy model lần đầu (có thể thiếu purchased data)
2. python3 targeted_crawler.py        # Crawl purchased_games + playtime cho player ưu tiên
3. python3 merge_data.py              # Merge data mới vào data/raw/purchased_games.csv
4. python3 src/data_prep.py           # Re-process Parquet
5. python3 main.py                    # Chạy lại model với coverage được cải thiện
```

**Định dạng output** của `targeted_crawler.py` (JSON array) khác với format gốc của `steam_crawling.py` (Python list string). `src/data_prep.py` hỗ trợ cả 2 format khi parse cột `library`.

---

**Last Updated**: April 8, 2026
