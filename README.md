# Steam Account Anomaly Detection System

> **Hệ thống machine learning phát hiện bot accounts và hoạt động gian lận trên Steam gaming platform bằng Dynamic Duo Ensemble: XGBoost (primary) + IsolationForest (secondary), kết hợp với semi-supervised learning để xử lý tình trạng không có ground truth.**

---

## 1. Introduction

### Bài toán

Steam là nền tảng game lớn nhất thế giới với hàng triệu tài khoản. Một phần trong đó có hành vi bất thường được phân thành 3 nhóm:

| Loại Bot | Đặc trưng |
|---------|----------|
| **Speed Bot** | median unlock interval < 10s, 85%+ achievements tập trung 1 game |
| **Volume Bot** | >500 achievements/ngày + >40% hoạt động ban đêm + tổng >1000; hoặc >90% achievements trên game chưa chạy (SAM unlocker) |
| **Review Bot** | >5 reviews, 0 achievements, >50% review cho game chưa chơi, >50% nội dung trùng lặp |

**Thách thức chính:**
- Không có ground truth labels (bots không tự nhận)
- Class imbalance cực độ (~75% accounts là inactive/ghost)
- Dữ liệu thực số lượng lớn (~196k players, 1.5 GB CSVs)

### Mục tiêu

Xây dựng hệ thống phát hiện anomalies với:
1. Xử lí class imbalance và ghost accounts
2. Phương pháp semi-supervised learning (PU Learning) để tận dụng weak labels
3. Ensemble voting để cải thiện độ tin cậy
4. Interpretability qua feature importance và SHAP explanations

---

## 2. Data Pipeline & EDA

### 2.1 Tiền xử lý (Phase 1 — `src/data_prep.py`)

**Input CSV** được load từ `data/raw/`:
- `history.csv` (~647 MB): playerid, achievementid, date_acquired
- `reviews.csv` (~551 MB): playerid, gameid, review text
- `players.csv` (~18 MB): playerid, country, created_date
- `purchased_games.csv` (~92 MB): playerid, library (JSON list with appid + playtime_mins)
- `private_steamids.csv` (~4 MB): danh sách tài khoản private

**Xử lý chính:**
1. Lọc private accounts (227k IDs)
2. Parse timestamp, extract gameid từ achievementid string via regex
3. Parse library JSON thành list of dicts (appid + playtime_mins)
4. Dedup, type-optimize, export to Parquet

**Output:** `data/processed/{history,players,reviews,purchased}.parquet`

### 2.2 EDA & Feature Engineering (Step 2-3)

**Heuristic Labels (pseudo ground truth)**

Sử dụng AND logic để giảm false positives — chỉ flag những tài khoản có dấu hiệu rõ ràng:

| Loại Bot | Điều kiện |
|---------|-----------|
| **Speed Bot** | median_interval < 10s AND top1_concentration > 85% |
| **Volume Bot** | (max_per_day > 500 AND night_ratio > 40% AND ach_count > 1000) OR zero_playtime_ratio > 90% |
| **Review Bot** | review_count > 5 AND ach_count == 0 AND unplayed_ratio > 50% AND dup_rate > 50% |
| **Normal** | ach_count > 10 AND median_interval > 600s (đầu vào PU Learning) |

**25 Features được tính** (`build_feature_matrix` trong `features.py`)

Chia thành 6 nhóm để phát hiện các chiều bất thường:

| Nhóm | Số features | Features |
|-----|-------------|---------|
| **Speed** | 5 | median/std unlock interval, CV, max/day, max/min |
| **Temporal** | 3 | night_activity_ratio, hour_entropy, activity_density |
| **Diversity** | 7 | total_ach, library_size, ach_game_ratio, top1/top3_concentration, game_hhi, avg_ach/game |
| **Review** | 5 | total_reviews, unplayed_ratio, duplication_rate, avg/min_length |
| **Account Age** | 2 | days_before_first_ach, account_age_days |
| **Playtime** | 3 | zero_playtime_ratio, total_playtime_mins, playtime_per_achievement |

**Data Trimming:** Thực hiện một lần duy nhất bên trong `build_feature_matrix` — giữ players với **(ach_count ≥ 10 OR review_count ≥ 3) AND library_size ≥ 1** → 196,703 → 22,583 active accounts (loại 174,120 ghost accounts)

---

## 3. Approach & Model Architecture

### 3.1 Lựa chọn kiến trúc

Kiến trúc hiện tại là **XGBoost (primary) + IsolationForest (secondary)**. Các lựa chọn thay thế bị loại vì:

- **LOF / OCSVM**: O(N²) complexity → không scale được với 22k players; OCSVM thêm cho sparse feature space → weak signal
- **PCA**: Thay thế tên feature thực bằng anonymous PC components → phá vỡ SHAP explainability; đồng thời nén các anomaly signals hiếm (bot archetypes chiếm <2%) vào các components bị loại, làm giảm recall
- **Supervised XGBoost thuần**: Không có ground truth labels → không thể train trực tiếp; PU Learning (chỉ train trên confident subset) là giải pháp phù hợp nhất cho bài toán weak-label này

### 3.2 Dynamic Duo Architecture (V3)

```
Raw Features (25)
       │
       ├──────────────────────────┐
       │ Path A — IsolationForest │ Path B — XGBoost
       │ log1p (heavy-tail cols)  │ X_raw trực tiếp
       │ SimpleImputer (median)   │ (NaN = tín hiệu thực)
       │ StandardScaler           │
       ↓                          ↓
  IsolationForest            XGBoost PU Learning
  (unsupervised)             (confident subset only:
                              heur_bot=1 OR heur_normal=1)
                              scale_pos_weight
                              RandomizedSearch (n_iter=50, cv=5)
                              PR-AUC scoring
       ↓                          ↓
  Percentile Ranking (0–100)
       ↓                          ↓
   if_pct                     xgb_pct
       └──────────┬────────────────┘
                  ↓
      Weighted Composite
      composite = 0.70×xgb_pct + 0.30×if_pct
                  ↓
      is_anomaly = (composite ≥ 85) ? 1 : 0
```

**Tại sao PU Learning cho XGBoost?**

Class imbalance (heur_bot:heur_normal ≈ 5:95%+) → chỉ train trên confirmed labels:
- Loại grey area (không có label hoặc conflicting signals)
- `scale_pos_weight` tự động cân bằng positive/negative trong training set
- Tăng sensitivity detect bots thực mà vẫn giữ specificity

**Tại sao percentile rank ensemble?**

Raw scores từ XGBoost (0–1) vs IsolationForest (âm, path-length based) không so sánh được → đưa về [0, 100] percentile rank:
- Mỗi model được xem xét công bằng
- Threshold (≥85) dễ hiểu: top 15% "most suspicious"

### 3.3 Preprocessing Pipeline (Step 4)

**Split paths — lý do tách riêng:**

| | Path A (IsolationForest) | Path B (XGBoost) |
|--|--------------------------|-----------------|
| Transform | log1p(clip(x, 0)) cho 9 heavy-tail features | Không |
| Impute | SimpleImputer(median) | Không — NaN xử lý natively |
| Scale | StandardScaler | Không — invariant với monotonic transform |

**Tại sao log1p (chỉ cho IF)?**

`std_unlock_interval_sec` có range 1s → 10 triệu giây (116 ngày). IsolationForest dùng uniform random split trên [min, max] của mỗi feature: khi feature trải dài nhiều bậc độ lớn, hầu hết split điểm rơi vào vùng sparse high-value tail, khiến path length không phản ánh mật độ dữ liệu thực. log1p cân bằng lại distribution, ổn định path-length comparison.

**Tại sao StandardScaler không RobustScaler?**

Sau trimming chỉ còn active players → IQR có ý nghĩa → StandardScaler cho z-score normalize cân bằng.

### 3.4 Phương pháp đánh giá

**Metrics chính:**
- **ROC-AUC**: Discrimination across thresholds vs heuristic_bot labels
- **PR-AUC**: Precision-Recall curve (tốt hơn ROC với imbalanced data)
- **Precision@K**: % bots trong top-K flagged accounts (K=100, 500, 1000) — "nếu review top-100, bao nhiêu % là bot thực?"
- **SHAP Feature Importance**: Contribution của từng feature vào prediction (trên X_raw — giữ thang đo gốc, dễ đọc)

**Validation set:**
- Heuristic labels (pseudo ground truth) → không phải true labels
- Known limitation: target leakage (features overlap heuristic rules) → metrics = upper bound trên real performance

### 3.5 Kết quả (Main.py outputs)

| Output | Nội dung |
|--------|----------|
| **ensemble_results.csv** | playerid, composite_score (0–100), is_anomaly, xgb_proba, xgb_pct, if_pct, heuristic_bot |
| **feature_matrix.csv** | 25 raw features cho all players |
| **heuristic_labels.csv** | heuristic_bot, heuristic_normal (sau HITL override) |
| **model_comparison.csv** | ROC-AUC, PR-AUC, Flagged Rate%, Precision@K cho XGBoost, IF, Ensemble |
| **ensemble_weight_metrics.csv** | Precision@100, PR-AUC, HCC theo từng xgb_weight từ 0→1 |
| **top50_flagged_profiles.csv** | Mean features: top-50 flagged vs normal users |
| **plots/shap_*.png** | SHAP summary, waterfall, scatter plots (XGBoost, trên X_raw) |
| **best_xgb.pkl** | Trained XGBoost model |
| **best_if.pkl** | Trained IsolationForest model |
| **preprocessor.pkl** | Pipeline (SimpleImputer + StandardScaler) dùng cho IF path |

**Một số kết quả ví dụ:**
- XGBoost ROC-AUC: 0.78–0.85 (tùy tuning)
- IF ROC-AUC: 0.45–0.55
- Ensemble ROC-AUC: 0.75–0.82
- Anomaly rate: ~2–5% của active accounts được flag

---

## 4. BI Dashboard & Testing

### 4.1 Streamlit Dashboard

Interactive web UI để query account bằng Steam ID:

```bash
streamlit run streamlit_app.py
```

**Tính năng:**
- Input Steam ID → Xem composite score, is_anomaly prediction, model confidence
- Feature visualization: So sánh 25 features của player với baseline (normal users)
- Model decision breakdown: Percentile rank từ XGB vs IF
- Behavioral evidence: Giải thích tại sao account bị flag (speed, reviews, etc.)
- Baseline comparison: Cohen's d effect size vs normal users

**Chỉ số được hiển thị:**
- xgb_flag, if_flag, is_anomaly (binary predictions)
- Combo matrix: (xgb_flag, if_flag) → 4 loại combined prediction
- Per-feature comparison: User value vs baseline percentile

### 4.2 Testcase Evaluation

Kiểm chứng model trên test set with ground truth labels:

```bash
python3 run_testcase_evaluation.py \
  --testcase-input data/test/testcase_40_unified.csv \
  --threshold 85.0 \
  --output-prefix eval_results
```

**Testcase format:**
- playerid, human_label (0=normal, 1=bot), + 25 feature columns
- Model inference + confusion matrix, classification report
- Output: predictions.csv, behavior_analysis.csv, summary.txt

---

## 5. Hướng dẫn sử dụng

### Chạy toàn bộ pipeline

```bash
# 1. Cấu hình TARGET_STEAM_IDS trong batch_analysis.py
# 2. Chạy:
python3 batch_analysis.py
```

**Kết quả:**
- `outputs/ensemble_results.csv` — scores + predictions cho all players
- `outputs/detection_report.{md,html}` — report cho TARGET accounts
- `outputs/plots/` — SHAP visualizations

### Chỉ query (skip model training)

```bash
python3 batch_analysis.py --query-only --steam-ids ID1 ID2 ...
```

### Chạy từng bước

```bash
python3 src/data_prep.py        # Phase 1: ETL + Parquet
python3 main.py                 # Phase 2: ML pipeline
streamlit run streamlit_app.py  # Dashboard UI
```

### Crawl thêm dữ liệu

```bash
# Crawl full data cho vài account cụ thể (yêu cầu API key)
python3 steam_crawling.py --steam-ids ID1 ID2

# Crawl purchased_games cho 10k priority players
python3 targeted_crawler.py

# Merge data mới vào raw
python3 merge_crawled_purchased_games.py
```

---

## Dependencies

`pip install -r requirements.txt`

---

## Known Limitations

1. **Target leakage**: Heuristic labels tính từ features dùng để train → metrics = "model mimics heuristics", không phải true performance
2. **Không có ground truth**: Chỉ dùng heuristic labels (weak labels)
3. **Data coverage**: Một phần players thiếu purchased_games → `review_unplayed_ratio` và playtime features bị NaN (XGBoost xử lý natively; IF path impute bằng median)
4. **Ghost accounts**: ~75% trong 196k bị loại bởi trimming → 22,583 active accounts được model
