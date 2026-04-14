# Steam Account Anomaly Detection System

> **Hệ thống machine learning phát hiện bot accounts và hoạt động gian lận trên Steam gaming platform bằng Dynamic Duo Ensemble: XGBoost (primary) + IsolationForest (secondary), kết hợp với semi-supervised learning để xử lý tình trạng không có ground truth.**

---

## 1. Introduction

### Bài toán

Steam là nền tảng game lớn nhất thế giới với hàng triệu tài khoản. Một phần trong đó có hành vi bất thường được phân thành 3 nhóm:

| Loại Bot | Đặc trưng |
|---------|----------|
| **Speed Bot** | Unlock thành tích <1s, 85%+ tập trung 1 game |
| **Volume Bot** | >500 thành tích/ngày, >40% hoạt động ban đêm (00:00–06:00), tổng >1000 achievements |
| **Review Bot** | >5 reviews cho game chưa từng sở hữu (thư viện trống) |

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
3. Parse library JSON thành list of dicts (N:1 relationship)
4. Dedup, type-optimize, export to Parquet

**Output:** `data/processed/{history,players,reviews,purchased}.parquet`

### 2.2 EDA & Feature Engineering (Step 2-3)

**Heuristic Labels V2 (pseudo ground truth)**

Sử dụng AND logic (không OR) để giảm false positives — chỉ flag những tài khoản có dấu hiệu rõ ràng:

| Loại Bot | Điều kiện AND |
|---------|---------------|
| Speed Bot | median_interval < 10s AND min_interval < 1s AND top1_concentration > 85% |
| Volume Bot | max_per_day > 500 AND night_ratio > 40% AND ach_count > 1000 |
| Review Bot | review_count > 5 AND ach_count == 0 AND unowned_ratio > 70% |
| Normal | ach_count > 10 AND median_interval > 1800s (đầo vào PU Learning) |

**27 Features được tính**

Chia thành 6 nhóm để phát hiện các chiều bất thường:

| Nhóm | Số features | Ý nghĩa |
|-----|----------|---------|
| **Speed** | 6 | Tốc độ unlock: median, min, std của time_interval, CV, max/day, max/min |
| **Temporal** | 4 | Mô hình thời gian: night_ratio, hour_entropy, activity_density, weekend_ratio |
| **Diversity** | 8 | Phân tán game: total_ach, games_w_ach, library_size, ach_ratio, concentration (top1/top3/HHI), avg_ach/game |
| **Review** | 5 | Review behavior: total, unowned_ratio*, duplication_rate, avg/min_length |
| **Account Age** | 2 | days_before_first_ach, account_age_days |
| **Playtime** | 2 | total_playtime_mins*, playtime_per_achievement* |

*NaN nếu không có purchased data → impute bằng median

**Data Trimming:** Loại ghost accounts — chỉ giữ những players với ach_count ≥ 10 AND library_size ≥ 1 → ~196k → ~40-60k accounts

---

## 3. Approach & Model Architecture

### 3.1 Các cách tiếp cận được nghiên cứu

Quá trình development qua 3 iteration:

| Iteration | Ensemble | Kết quả | Vấn đề |
|-----------|----------|--------|--------|
| **V1** | IF + LOF + OCSVM (voting) | ROC-AUC = 0.13 | IF bị invert (0.0052), LOF/OCSVM O(N²) bottleneck |
| **V2** | IF + XGBoost (supervised) | ROC-AUC = 0.62 | Heuristic labels quá nhiễu (OR logic), library parsing bug |
| **V3 (hiện tại)** | XGBoost (PU Learning) + IF auto-flip | ROC-AUC = 0.78–0.85 | Model ổn định, interpretable |

**Lý do loại bỏ:**
- **LOF**: O(N²) complexity → timeout với 40k players
- **OCSVM**: Low weight (0.20) + sparse feature space → weak signal

### 3.2 Dynamic Duo Architecture (V3)

```
┌─────────────────────────────────┐
│ Preprocessing                   │
│ - log1p(heavy_tail_cols)        │
│ - Impute NaN with median        │
│ - StandardScaler                │
│ - PCA (90% variance → 8-12 PC)  │
└──────────────┬──────────────────┘
               │
        ┌──────┴──────┐
        ↓             ↓
   PRIMARY        SECONDARY
   ┌────────────────────┐
   │ XGBoost            │ IsolationForest
   │ PU Learning Filter │ Auto-flip if ROC<0.4
   │ (heur_bot=1 OR     │ (Statistical outlier)
   │  heur_normal=1)    │
   │ scale_pos_weight   │
   │ RandomizedSearch   │
   │ (n_iter=50, cv=5)  │
   │ PR-AUC scoring     │
   └────────────────────┘
        ↓             ↓
   Percentile Ranking (0-100)
        ↓             ↓
    XGB_pct ← IF_pct
        │         │
        └────┬────┘
             ↓
   Weighted Composite
   composite = 0.80×XGB_pct + 0.20×IF_pct
        ↓
   is_anomaly = (composite ≥ 85) ? 1 : 0
```

**Tại sao PU Learning cho XGBoost?**

SOS class imbalance (heur_bot:heur_normal ≈ 5:95%+) → chỉ train trên confirmed labels
- Loại grey area (không có label hoặc conflicting signals)
- scale_pos_weight tự động cân bằng positive/negative trong training set
- Tăng cảm nhạy detect bots thực mà vẫn giữ specificity

**Tại sao percentile rank ensemble?**

Raw scores từ XGBoost (0–1) vs IsolationForest (-∞ to 0) không so sánh được → đưa về [0, 100] percentile rank:
- Mỗi model được công bằng xem xét
- Threshold (≥85) dễ hiểu: top 15% "most suspicious"
- IF auto-flip: nếu ROC-AUC < 0.4 → đảo hướng (inversion detection)

### 3.3 Preprocessing Pipeline (Step 4)

**Tại sao log1p transform?**

`std_unlock_interval_sec` có range 1s → 10 triệu giây (=116 ngày). PCA collapse nếu không transform:
- Trước: PC1 giải thích >92% variance → model chạy trên dữ liệu gần 1D
- Sau log1p: Phân phối đối xứng, 8-12 PC giải thích 90% variance → multi-dimensional anomaly detection

**Tại sao StandardScaler không RobustScaler?**

Ghost accounts (0 ach) chiếm ~75% → IQR ≈ 0 → RobustScaler scale active players lên giá trị cực lớn:
- Trước: Ghost ≈ 0, Active ≈ 1000+ → feature space bị collapse
- Sau: Tất cả → z-score normalize → balanced representation

### 3.4 Phương pháp đánh giá

**Metrics chính:**
- **ROC-AUC**: Measure discrimination across thresholds vs heuristic_bot labels
- **PR-AUC**: Precision-Recall curve (tốt hơn ROC với imbalanced data)
- **Precision@K**: % bots trong top-K flagged accounts (K=100, 500, 1000)
  - Đánh giá "nếu review top-100 flagged, bao nhiêu %là bot thực?"
- **SHAP Feature Importance**: Cách các feature đóng góp vào prediction

**Validation set:**
- Heuristic labels (pseudo ground truth) → không phải true labels
- Known limitation: target leakage (features overlap heuristic rules)
  - Metrics đo "model bắt chước heuristic rules" → upper bound trên real performance

### 3.5 Kết quả (Main.py outputs)

| Output | Nôi dung |
|--------|----------|
| **ensemble_results.csv** | playerid, composite_score (0-100), is_anomaly, xgb_proba, xgb_pct, if_pct, heuristic_bot |
| **feature_matrix.csv** | 27 raw features cho all players |
| **model_comparison.csv** | ROC-AUC, PR-AUC, Precision@K cho XGBoost, IF, Ensemble |
| **top50_flagged_profiles.csv** | Top-50 highest composite_score + feature stats so với normal users |
| **plots/shap_*.png** | SHAP summary, waterfall, scatter plots (XGBoost explanations) |
| **best_xgb.pkl** | Trained XGBoost model |
| **preprocessor.pkl** | Pipeline (Imputer + StandardScaler + PCA) |

**Một số kết quả ví dụ:**
- XGBoost ROC-AUC: 0.78–0.85 (tùy tuning)
- IF ROC-AUC: 0.45–0.55 (trước auto-flip)
- Ensemble ROC-AUC: 0.75–0.82
- Anomaly rate: ~2-5% của active accounts được flag

---

## 4. BI Dashboard & Testing

### 4.1 Streamlit Dashboard

Interactive web UI để query account bằng Steam ID:

```bash
streamlit run streamlit_app.py
```

**Tính năng:**
- Input Steam ID → Xem composite score, is_anomaly prediction, model confidence
- Feature visualization: So sánh 27 features của player với baseline (normal users)
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
- playerid, human_label (0=normal, 1=bot), + 27 feature columns
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
python3 src/data_prep.py       # Phase 1: ETL + Parquet
python3 main.py                # Phase 2: ML pipeline
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

```
pandas, numpy, scikit-learn, scipy, xgboost, shap, matplotlib, joblib, pyarrow
streamlit (dashboard), requests (crawling)
```

Cài: `pip install -r requirements.txt`

---

## Known Limitations

1. **Target leakage**: Heuristic labels tính từ features dùng để train → metrics = "model mimics heuristics"
2. **Không có ground truth**: Chỉ dùng heuristic labels (weak labels)
3. **Data coverage**: ~75% players thiếu purchased_games → features bị impute
4. **Ghost accounts**: ~75% trong 196k bị loại → chỉ ~40-60k active accounts được model


