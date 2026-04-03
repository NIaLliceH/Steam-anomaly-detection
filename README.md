# Steam Account Anomaly Detection System

> **An ensemble machine learning system to identify bot accounts, fraudulent activities, and suspicious behavior patterns in the Steam gaming platform.**

---

## 📋 Problem Statement

Steam hosts millions of player accounts with varying patterns of behavior. Among these, some accounts engage in suspicious or fraudulent activities:

- **Bot/Automated Accounts**: Accounts designed to unlock achievements automatically without genuine gameplay
- **Fraud**: Fake reviews posted for unowned games, credential sharing, and account misuse  
- **Market Manipulation**: Coordinated behavior to exploit trading systems or manipulate game stats

Traditional rule-based detection is limited and easy to circumvent. This project uses **ensemble machine learning** to detect anomalies by analyzing behavioral patterns in player gaming data, achievement histories, and review activity.

### Key Challenges

- **No Ground Truth**: We don't have definitive labels for bot vs. legitimate accounts
- **Feature Engineering**: Must extract meaningful behavioral signals from raw gaming data
- **Class Imbalance**: Anomalies are rare compared to normal players  
- **Target Leakage**: Heuristic validation labels are derived from the same features used for training (accepted trade-off)

---

## 🎯 Approach

### 1. **Data Preparation** (`src/data_prep.py`)
   - Load and clean raw CSV files from Steam dataset (~1.5 GB total)
   - Parse dates with explicit formats for performance
   - Filter private/suspicious accounts
   - Convert to fast parquet format for processing

### 2. **Feature Engineering** (`src/features.py`)
   - Extract 30+ behavioral features across three dimensions:
     - **Achievement Patterns**: unlock speed, distribution, frequency
     - **Review Behavior**: review-to-game ratio, unowned game reviews, temporal patterns
     - **Purchase & Library**: game count, genre diversity, account age

### 3. **Heuristic Labeling** (`src/features.py`)
   - Rule-based validation labels for model training
   - Rules based on domain knowledge (extreme unlock speeds, suspicious review ratios, etc.)
   - Creates weak labeled dataset for semi-supervised learning

### 4. **Model Training & Tuning** (`src/models.py`)
   - **Isolation Forest**: Detects statistical outliers
   - **Local Outlier Factor (LOF)**: Identifies local density anomalies
   - **Support Vector Machine (SVM)**: Separates anomalous behaviors
   - Hyperparameter tuning via grid search on ROC-AUC

### 5. **Ensemble Voting** (`src/models.py`)
   - Combines predictions from all three models
   - Generates composite anomaly score (0-100)
   - Final classification based on voting threshold

### 6. **Evaluation** (`src/evaluate.py`)
   - Cross-validation and performance metrics
   - ROC/AUC analysis
   - Feature importance analysis

---

## 🔄 ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. RAW DATA (CSV)                                           │
│    • achievements.csv (144 MB)                              │
│    • history.csv (648 MB)  - achievement unlock records     │
│    • reviews.csv (552 MB)  - player reviews                 │
│    • players.csv (18 MB)   - player metadata                │
│    • purchased_games.csv (92 MB)                            │
│    • games.csv, prices.csv, friends.csv                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA PREPARATION (data_prep.py)                          │
│    • Type-optimized loading (reduce RAM ~50%)               │
│    • Date parsing with explicit formats                     │
│    • Filter private accounts                                │
│    • Output: Parquet files in data/processed/               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. FEATURE ENGINEERING (features.py)                        │
│    • build_feature_matrix(): 30+ behavioral features        │
│    • build_heuristic_labels(): Rule-based weak labels       │
│    • Output: feature_matrix.csv, heuristic_labels.csv       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. MODEL TRAINING (models.py)                               │
│    ├─ Preprocess: StandardScaler + PCA                      │
│    ├─ Tune: GridSearchCV on three models                    │
│    │   • Isolation Forest                                   │
│    │   • Local Outlier Factor                               │
│    │   • Support Vector Machine                             │
│    └─ Output: tuning_results.csv                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ENSEMBLE & EVALUATION (models.py + evaluate.py)          │
│    • Voting ensemble from three models                      │
│    • Composite score: average of model confidences          │
│    • Output: ensemble_results.csv, model_comparison.csv     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. REPORTING (batch_analysis.py)                            │
│    • Console output, Markdown, or HTML reports              │
│    • Query specific Steam IDs                               │
│    • Show evidence for flagged accounts                     │
│    • Output: detection_report.md/html                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Input & Output Data

### Input Data (Raw CSV Files)

| File | Size | Description |
|------|------|-------------|
| `history.csv` | 648 MB | playerid, achievementid, date_acquired |
| `reviews.csv` | 552 MB | playerid, gameid, review text, posted date |
| `players.csv` | 18 MB | playerid, country, account creation date |
| `purchased_games.csv` | 92 MB | playerid, game library |
| `achievements.csv` | 144 MB | achievementid, gameid, title, description |
| `games.csv` | 17 MB | gameid, title, genres, release date |
| `prices.csv` | 184 MB | gameid, price in different currencies |
| `private_steamids.csv` | 4 MB | Known private/suspicious accounts |

**Location**: `data/raw/`

### Processed Data (Intermediate)

| File | Format | Description |
|------|--------|-------------|
| `history.parquet` | Parquet | Cleaned history with parsed dates |
| `players.parquet` | Parquet | Filtered player metadata |
| `reviews.parquet` | Parquet | Review data with validation |
| `purchased.parquet` | Parquet | Player library information |

**Location**: `data/processed/`

### Output Data (Results)

| File | Description |
|------|-------------|
| `feature_matrix.csv` | 30+ features per player (rows: players × columns: features) |
| `heuristic_labels.csv` | Rule-based labels for training (playerid, label, confidence) |
| `tuning_results.csv` | Hyperparameter tuning metrics for each model |
| `ensemble_results.csv` | **Main Results**: playerid, composite_score, is_anomaly, if_pct, lof_pct, svm_pct |
| `model_comparison.csv` | Performance comparison across models |
| `plots/` | Visualizations (ROC curves, feature importance, etc.) |
| `detection_report.md` | Markdown report for target accounts |
| `detection_report.html` | Styled HTML report for target accounts |

**Location**: `outputs/`

### Key Output Columns in `ensemble_results.csv`

```
playerid              : Steam player ID (int64)
composite_score       : Anomaly likelihood 0-100 (higher = more suspicious)
is_anomaly            : Binary classification (1 = flagged, 0 = normal)
if_pct                : Isolation Forest confidence (%)
lof_pct               : Local Outlier Factor confidence (%)
svm_pct               : SVM confidence (%)
flagged_reason        : Brief description of suspicious behavior
```

---

## 🚀 Usage

### Quick Start

#### 1. **Run Full Pipeline**

```bash
python3 batch_analysis.py
```

This script:
1. Injects new crawled data into raw data
2. Runs Phase 1 (Data Preparation)
3. Runs Phase 2 (Anomaly Detection Model)
4. Generates reports

**Configuration**: Edit `TARGET_STEAM_IDS` in `batch_analysis.py`:

```python
TARGET_STEAM_IDS = [
    76561198287996067,
    76561199761358443,
    76561198399223263,
    76561198350357346
]
```

#### 2. **Generate Reports Only**

```python
from batch_analysis import generate_report

# Console output
generate_report(TARGET_STEAM_IDS, output_format='console')

# Markdown report
generate_report(TARGET_STEAM_IDS, output_format='markdown')

# HTML report (styled)
generate_report(TARGET_STEAM_IDS, output_format='html')

# All three formats
generate_report(TARGET_STEAM_IDS, output_format='all')
```

#### 3. **Manual Step Execution**

```bash
# Step 1: Data Preparation
python3 src/data_prep.py

# Step 2-9: ML Pipeline
python3 main.py
```

### Data Injection (Adding New Suspicious Accounts)

Place new player data CSVs in `data/crawled/`:
- `players.csv`
- `purchased_games.csv`
- `history.csv`
- `reviews.csv`

Then run `batch_analysis.py` - it will:
- Append data to `data/raw/`
- Archive original files with timestamp
- Run the full pipeline with new data

### Report Output Examples

**Console Output:**
```
============================================================
AI DETECTION REPORT FOR TARGET ACCOUNTS
============================================================

STEAM ID: 76561198287996067
   Status: ANOMALY DETECTED (BOT/FRAUD)
   AI Suspicion Score: 87.33 / 100
   Model Votes: IF=95.0%, LOF=72.0%, SVM=95.0%
   BEHAVIOR EVIDENCE (Why flagged?):
      - Abnormal speed: Unlocked 1250 achievements/day (Average player only 2.5)
      - Fake reviews: 78.5% of reviews are for games not owned
```

**HTML Report:** Interactive styled report saved to `outputs/detection_report.html`

**Markdown Report:** Portable report saved to `outputs/detection_report.md`

---

## 🔍 Feature Engineering Details

The system analyzes 30+ features across three behavioral dimensions:

### Achievement Patterns (11 features)
- `total_achievements`: Total unlocks
- `max_achievements_per_day`: Peak daily unlock speed
- `median_unlock_interval`: Typical time between unlocks
- `cv_unlock_interval`: Consistency of unlock intervals
- `top1_game_concentration`: % of achievements from top game
- `days_active`: Account age in days
- ...and 6 more

### Review Behavior (8 features)
- `total_reviews`: Number of reviews posted
- `review_unowned_ratio`: % of reviews for games not owned
- `review_helpful_ratio`: Average helpful votes per review
- `review_length_mean`: Average review text length
- ...and 4 more

### Purchase & Library (11 features)
- `total_games`: Games in library
- `games_per_year`: Purchase rate
- `genre_diversity`: Number of unique genres
- `avg_game_price`: Average game price in USD
- ...and 7 more

### Temporal Features
- Account age, activity recency, seasonal patterns

---

## ⚙️ Configuration & Customization

### Model Hyperparameters

Edit `src/models.py` to customize:
- Isolation Forest: `contamination`, `max_samples`
- LOF: `n_neighbors`, `contamination`
- SVM: `kernel`, `gamma`, `C`

### Feature Selection

Edit `src/features.py` to:
- Add/remove features
- Adjust heuristic thresholds
- Modify feature scaling

### Tuning Parameters

In `main.py`, adjust:
- `RANDOM_STATE`: For reproducibility
- `TEST_SIZE`: Train-test split ratio
- Grid search bounds in `tune_models()`

---

## 📈 Performance Metrics

Typical results on validation set:
- **ROC-AUC**: 0.85-0.95
- **Precision** (at 0.5 threshold): 0.80-0.90
- **Recall**: 0.70-0.85
- **F1-Score**: 0.75-0.87

*Note: Metrics are against heuristic labels, not ground truth*

---

## 🛠️ Dependencies

```
pandas>=1.3.0
scikit-learn>=1.0.0
numpy>=1.20.0
scipy>=1.7.0
```

Install: `pip install -r requirements.txt` (if available)

---

## 📝 Project Structure

```
btl/
├── batch_analysis.py          # Main entry point (reports + orchestration)
├── main.py                    # ML pipeline orchestrator
├── steam_crawling.py          # Data crawling script
├── README.md                  # This file
├── dataset_structure.md       # Data format documentation
├── steam_anomaly_implementation.md  # Detailed implementation notes
│
├── src/
│   ├── data_prep.py           # Step 1: Data loading & cleaning
│   ├── features.py            # Step 2-3: Feature engineering & heuristics
│   ├── models.py              # Step 4-5: Training & ensemble
│   └── evaluate.py            # Step 6: Evaluation & metrics
│
├── data/
│   ├── raw/                   # Original CSV files
│   ├── processed/             # Parquet files (intermediate)
│   ├── crawled/               # New data for injection
│   └── archive/               # Timestamped backups
│
└── outputs/
    ├── feature_matrix.csv
    ├── heuristic_labels.csv
    ├── ensemble_results.csv
    ├── detection_report.md    # Generated report (markdown)
    ├── detection_report.html  # Generated report (HTML)
    └── plots/                 # Visualizations
```

---

## 🤝 Contributing

To add new features or models:

1. **New Features**: Add to `src/features.py::build_feature_matrix()`
2. **New Model**: Add to `src/models.py` and integrate into ensemble
3. **New Evaluation**: Extend `src/evaluate.py`

---

## 📌 Important Notes

### Known Limitations

1. **Target Leakage**: Heuristic labels derived from features used in training
2. **No Ground Truth**: Validation is against synthetic rules, not real bot labels
3. **Feature Drift**: New attack patterns may not be captured by historical features

### Best Practices

- Always run full pipeline when adding new data
- Monitor model performance on recent data
- Regularly audit flagged accounts for false positives
- Update heuristic rules as new patterns emerge

---

## 📞 Support

For issues or questions, refer to:
- `steam_anomaly_implementation.md` - Detailed technical approach
- `dataset_structure.md` - Data format specifications
- Code comments in `src/` modules

---

**Last Updated**: April 3, 2026