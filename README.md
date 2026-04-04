# Steam Account Anomaly Detection System

> **Hệ thống phát hiện bất thường sử dụng Semi-supervised XGBoost kết hợp Ensemble để nhận diện bot accounts và hoạt động gian lận trên nền tảng Steam.**

---

## Bài toán

Steam là nền tảng game lớn nhất thế giới với hàng triệu tài khoản người dùng. Trong đó tồn tại các tài khoản có hành vi bất thường:

- **Bot tự động**: Tài khoản dùng script để unlock thành tích mà không chơi thật
- **Review Fraud**: Đăng đánh giá cho game chưa từng mua/chơi
- **Account Farm**: Tài khoản hoạt động bất thường về thời gian hoặc khối lượng

Dự án sử dụng **Semi-supervised Ensemble ML** gồm XGBoost (primary) + IsolationForest / LOF / OneClassSVM (secondary) để phân tích behavioral patterns từ dữ liệu lịch sử thành tích, thư viện game, và review của từng người chơi.

### Thách thức chính

| Thách thức | Cách xử lý |
|---|---|
| Không có ground truth | Heuristic labels V2 với AND-logic 3 nhóm bot rõ ràng |
| Class imbalance cực độ | PU Learning — loại grey area khỏi tập train XGBoost |
| Ghost accounts (~75%) | Data trimming: chỉ giữ players có ≥ 10 thành tích và ≥ 1 game |
| PCA collapse do heavy tails | log1p transform 10 features trước preprocessing |
| IQR ≈ 0 của ghost accounts | Thay RobustScaler bằng StandardScaler sau log1p |
| review_unowned_ratio bị inflate | Trả NaN thay vì set() rỗng cho player không có library data |

---

## Kiến trúc Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1 — DATA ETL  (src/data_prep.py)                     │
│                                                             │
│  Raw CSV (~1.5 GB)                                          │
│    history.csv, players.csv, reviews.csv,                   │
│    purchased_games.csv, private_steamids.csv                │
│                                ↓                            │
│  • Lọc private accounts (227k IDs)                          │
│  • Parse datetime với explicit format                       │
│  • Extract gameid từ achievementid string                   │
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
│      - Speed Bot: median_interval<10s AND min<1s AND top1>85%│
│      - Volume Bot: max_per_day>500 AND night>40% AND ach>1k  │
│      - Review Bot: reviews>5 AND 0 ach AND unowned>70%      │
│    • Strict Normal: ach_count>10 AND median_interval>1800s  │
│    Output: heuristic_labels.csv (bot + normal columns)      │
│                                                             │
│  Step 3: Feature Engineering  (features.py)                 │
│    • 25 features / 5 nhóm:                                  │
│      A. Speed (6): median/min/std interval, cv, max/min/day │
│      B. Temporal (4): night ratio, hour entropy, density,   │
│                       weekend ratio                         │
│      C. Diversity (8): total ach, games w/ ach, library     │
│                        size, ach/game ratio, top1/3/HHI     │
│      D. Review (5): count, unowned_ratio*, dup rate,        │
│                     avg/min length                          │
│      E. Account Age (2): days_before_first_ach, account_age │
│    • *unowned_ratio = NaN nếu không có library data         │
│      (tránh inflate giả tạo)                                │
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
│    Output: preprocessor.pkl, PCA component names           │
│                                                             │
│  Step 5a: Unsupervised Tuning  (models.py)                  │
│    • Grid search trên heuristic labels:                     │
│      - IsolationForest: 54 combos                           │
│      - LOF (novelty=True, sample=20k): 24 combos            │
│      - OneClassSVM (sample=20k): 12 combos                  │
│    Output: tuning_results.csv                               │
│                                                             │
│  Step 5b: XGBoost — PRIMARY MODEL  (models.py)              │
│    • PU Learning filter:                                    │
│      train chỉ trên: heuristic_bot==1 OR heuristic_normal==1│
│      loại grey area (không rõ bot hay người thật)           │
│    • scale_pos_weight = neg/pos (xử lý class imbalance)     │
│    • RandomizedSearchCV(n_iter=50, cv=5, scoring=PR-AUC)    │
│    • predict_proba() chạy trên TOÀN BỘ trimmed dataset     │
│    Output: best_xgb.pkl, xgb_tuning_results.csv            │
│                                                             │
│  Step 6: Train Final Unsupervised  (models.py)              │
│    • Retrain IF/LOF/OCSVM với best params trên toàn bộ data │
│                                                             │
│  Step 7: Ensemble  (models.py)                              │
│    • Percentile rank mỗi model (0-100)                      │
│    • IF auto-flip: nếu AUC < 0.4 → đảo score (inversion)   │
│    • Weighted composite:                                    │
│        XGB×0.50 + LOF×0.30 + IF×0.15 + OCSVM×0.05         │
│    • Voting: flag nếu ≥ 2 models đồng ý (top 5%)           │
│    Output: ensemble_results.csv                             │
│                                                             │
│  Step 8: Evaluation  (evaluate.py)                          │
│    • ROC-AUC và PR-AUC cho tất cả models                    │
│    • Precision@100/500/1000                                 │
│    • Top-50 flagged vs normal feature profile               │
│    • XGBoost: PR curve, optimal F1 threshold,               │
│               feature importance                            │
│    Output: model_comparison_v2.csv, plots/                  │
│                                                             │
│  Step 9: SHAP Explanations  (evaluate.py)                   │
│    • Native XGBoost SHAP (pred_contribs=True)               │
│      bypass lỗi TreeExplainer "[5E-1]"                      │
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
    ├── model_comparison_v2.csv
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
| `purchased_games.csv` | ~92 MB | playerid, library (list of gameids) |
| `private_steamids.csv` | ~4 MB | playerid của tài khoản private |

**Vị trí**: `data/raw/`

### Output chính — `ensemble_results.csv`

| Cột | Mô tả |
|-----|-------|
| `playerid` | Steam player ID |
| `composite_score` | Điểm nghi vấn tổng hợp 0–100 (cao = đáng ngờ hơn) |
| `is_anomaly` | 1 = bị flag, 0 = bình thường |
| `xgb_proba` | Xác suất bot theo XGBoost (0–1) |
| `xgb_pct` / `if_pct` / `lof_pct` / `svm_pct` | Percentile rank từng model |
| `vote_count` | Số model đồng ý flag (0–4) |
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

## Feature Engineering — 25 features

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

---

## Heuristic Labels V2

Nhãn heuristic được dùng làm pseudo ground truth để tuning và đánh giá (không phải ground truth thực sự).

### Nhóm bot (heuristic_bot = 1)

Dùng AND logic để giảm false positive (khác V1 dùng OR):

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
X_raw (25 features)
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
PCA(n_components=0.90)   ← giữ 90% variance → 8-12 components
    ↓
X_scaled (PCA components)
```

**Lý do log1p**: `std_unlock_interval_sec` có thể lên đến hàng triệu giây, làm PCA collapse: PC1 giải thích >92% variance chỉ bởi 1 chiều → LOF/OCSVM chạy trên dữ liệu 1D.

**Lý do StandardScaler thay RobustScaler**: Ghost accounts (0 achievements) chiếm ~75% dataset — IQR của hầu hết features ≈ 0 → RobustScaler scale active players lên giá trị cực lớn.

---

## Models

### XGBoost — Primary (Semi-supervised PU Learning)

- **Training set**: chỉ `heuristic_bot==1` OR `heuristic_normal==1`
  - Loại grey area để tránh noisy labels
- **scale_pos_weight** = confirmed_normal / bot (xử lý imbalance)
- **Tuning**: RandomizedSearchCV(n_iter=50, cv=5, scoring=PR-AUC)
- **Scoring**: predict_proba() trên toàn bộ trimmed dataset
- **Weight trong ensemble**: 0.50

### IsolationForest — Secondary

- Phát hiện statistical outliers trong không gian PCA
- Grid search: 54 combos (n_estimators, max_samples, contamination, max_features)
- Auto-flip: nếu ROC-AUC < 0.4 → đảo score (inversion detection)
- **Weight**: 0.15

### Local Outlier Factor — Secondary

- Phát hiện local density anomalies
- Train trên 20k sample (O(N²) complexity)
- **Weight**: 0.30

### OneClassSVM — Secondary

- RBF kernel, train trên 20k sample
- **Weight**: 0.05

### Ensemble

```
composite_score = 0.50 × XGB_pct
               + 0.30 × LOF_pct
               + 0.15 × IF_pct (sau khi flip nếu cần)
               + 0.05 × SVM_pct

is_anomaly = 1  nếu ≥ 2 trong 4 models flag (top 5% của model đó)
```

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

1. **Target leakage**: Heuristic labels được tính từ các features cũng dùng để train — metrics ROC-AUC đo "model có bắt chước heuristic rules không", không phải "model phát hiện bot thực sự tốt đến đâu"
2. **Không có ground truth**: Không có nhãn thực xác nhận tài khoản nào là bot
3. **purchased_games coverage thấp**: ~75% người có review không có trong bảng purchased_games → `review_unowned_ratio` bị impute bằng median

---

**Last Updated**: April 4, 2026
