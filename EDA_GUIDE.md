# EDA Guide — Steam Anomaly Detection

**Goal:** Understand your data through exploratory analysis before building the anomaly detection model.

> **Prerequisites:** Run Phase 1 first: `python3 src/data_prep.py` (generates processed parquets)

---

## 1. Quick Start (5 minutes)

### Option A: Run Standalone Python Script
Best for **quick stats** without Jupyter:

```bash
python3 eda_standalone.py --output-dir outputs/eda_results
```

Output:
- Console statistics (5 sections)
- 3 PNG visualizations (temporal, distributions, labels)

### Option B: Full Interactive Jupyter Notebook
Best for **detailed exploration** and visualization:

```bash
jupyter notebook eda_comprehensive.ipynb
```

Or in VS Code: Open `eda_comprehensive.ipynb` → Run all cells

---

## 2. What You'll Learn

### EDA Notebook Sections

| Section | Focus | Key Questions |
|---------|-------|----------------|
| **1** | Data Overview | How many records? Ghost accounts? Memory usage? |
| **2** | Feature Distributions | Are features normally distributed? Heavy tails? |
| **3** | Correlations | Feature independence? Multicollinearity issues? |
| **4** | Temporal Patterns | When do achievements unlock? Night activity? |
| **5** | Behavioral Segments | How many bots vs normal? Bot archetypes? |
| **6** | Feature Validation | Are calculations correct? How much NaN data? |
| **7** | Trimming Impact | What do ghost accounts look like? Why trim? |
| **8** | Class Imbalance | Extreme imbalance? Justifies PU Learning? |

### Expected Insights

**Data Quality:**
- ✓ Ghost accounts: 88% of 196K removed → 22.5K active players
- ✓ Missing data: ~50% NaN in playtime (API lag), ~30% NaN in reviews (no reviews)
- ✓ Data integrity: All core columns valid (playerid, date_acquired, library)

**Feature Properties:**
- ✓ Heavy-tailed: `median_unlock_interval_sec` (1s → 10M s) → justify log1p transform
- ✓ Independent: Most correlations |r| < 0.7 → ensemble voting valid
- ✓ Sparse: 25% → 50% NaN in review features → XGBoost handles natively

**Behavioral Patterns:**
- ✓ Night activity: Normal ~20-30%, bots ~25%+ (24/7)
- ✓ Speed signature: <10s median interval clusters distinctly
- ✓ Bot prevalence: ~2-5% heuristic_bot, ~50% normal, ~35% grey area

**Modeling Implications:**
- ✓ Extreme imbalance (1:50 bot:normal) → PU Learning justified
- ✓ Weak labels + no ground truth → semi-supervised confidence subset approach
- ✓ Dual preprocessing: Log1p for IsolationForest (path stability), raw for XGBoost (NaN handling)

---

## 3. Execution Workflow

### Phase 0: Raw Data Profiling
```bash
# Original EDA (raw data statistics)
jupyter notebook eda.ipynb
```

### Phase 1: Data Preparation
```bash
python3 src/data_prep.py
# Output: data/processed/{history,players,reviews,purchased}.parquet
```

### Phase 2: Comprehensive EDA (THIS GUIDE)
```bash
# Option A: Quick stats
python3 eda_standalone.py

# Option B: Full interactive
jupyter notebook eda_comprehensive.ipynb
```

Generates:
- Console output (5 sections)
- Feature distributions (histograms)
- Correlation heatmap
- Temporal patterns (hour/day)
- Bot profiles by heuristic type
- Trimming impact analysis
- Class imbalance visualization

### Phase 3: Full ML Pipeline
```bash
python3 main.py
# Trains XGBoost + IsolationForest ensemble
# Generates ensemble_results.csv, model comparison metrics
```

---

## 4. Interpreting Outputs

### Console Output

**Section 1: Data Overview**
```
HISTORY:
  Rows: 9,234,567
  Unique players: 196,703
  Memory: 847.23 MB
  Date range: 2009-01-01 to 2025-05-05

Ghost Account Trimming (Expected):
  Before: 196,703
  After: 22,583
  Removed (ghost): 174,120 (88.5%)
```

→ **Interpretation:** 88% inactive accounts removed; 22.5K active remain

**Section 2: Feature Analysis**
```
Heavy-tailed features (|skew| > 2): 5
  - median_unlock_interval_sec: 6.34
  - max_achievements_per_day: 4.21
  - total_playtime_mins: 3.15
  - std_unlock_interval_sec: 2.89
```

→ **Interpretation:** Log1p transform needed for IsolationForest stability

**Section 4: Temporal Patterns**
```
Achievement unlocks by hour (UTC):
  Total achievements: 9,234,567
  Night activity (UTC 00:00-05:59): 22.3%
  Peak hour: 14:00 (483,201 achievements)
```

→ **Interpretation:** Fairly even distribution (humans vary by timezone)

**Section 5: Heuristic Labels**
```
Label Distribution:
  Total: 22,583
  Bot: 934 (4.14%)
  Normal: 11,247 (49.80%)
  Grey area: 10,402 (46.06%)
```

→ **Interpretation:** ~5% confirmed bots, ~50% confirmed normal, ~46% ambiguous

### Visualizations

**01_temporal_patterns.png**
- Bar chart: achievements per hour (UTC)
- Bar chart: achievements per day-of-week
- Expected: Roughly even (humans distributed globally)
- Anomaly: All midnight would indicate bots

**02_feature_distributions.png**
- 6 histograms: representative features
- Look for: Skewness, outliers, bimodality
- Expected: Some right-skew (time-based metrics)
- Anomaly: Heavy left-tail would suggest truncation

**03_heuristic_labels.png**
- Pie chart: label proportions
- Bar chart: label counts
- Expected: ~5% bot, ~50% normal
- Imbalance ratio: Justifies PU Learning

---

## 5. Troubleshooting

### "feature_matrix.csv not found"
```
⚠ feature_matrix not available. Skipping Section 2.
```
→ **Solution:** Run `python3 main.py` first (generates feature_matrix.csv)

### "heuristic_labels.csv not found"
```
⚠ heuristic_labels.csv or feature_matrix not found.
```
→ **Solution:** Run `python3 main.py` first

### High NaN rates (>70%) in a feature
```
review_unplayed_ratio: 68.5%
```
→ **Interpretation:** Many players have no reviews (expected). XGBoost handles natively; IsolationForest imputes with median.

### Unexpected anomaly rate (>20%)
```
Bot: 4,500 (19.9%)
```
→ **Check:** Did trimming fail? Are heuristic thresholds too loose?

---

## 6. Next Steps After EDA

1. **Understand the findings:** Review plots and statistics
2. **Validate assumptions:** Confirm bot archetypes match domain knowledge
3. **Run full pipeline:** `python3 main.py`
4. **Check model performance:** Look at outputs/model_comparison.csv
5. **Deploy dashboard:** `streamlit run streamlit_app.py`

---

## 7. File Reference

| File | Purpose | Output |
|------|---------|--------|
| `eda_standalone.py` | Quick EDA script | Console stats + 3 PNGs |
| `eda_comprehensive.ipynb` | Full interactive EDA | 8 sections, 9 PNGs |
| `eda.ipynb` | Original raw data profiling | Basic statistics |
| `outputs/plots/eda_*.png` | Generated visualizations | 9 PNG files |

---

## 8. Key Metrics to Remember

After running EDA, you should know:

- **Active players:** 22,583 (after trimming 174K ghost accounts)
- **Ghost account rate:** 88.5% of raw population
- **Anomaly prevalence:** 2-5% (heuristic_bot)
- **Class imbalance:** ~1:50 (bot:normal)
- **Heavy-tailed features:** 5+ (justify log1p)
- **Night activity:** 20-30% (normal), 25%+ (bots)
- **Speed bot signature:** <10s median interval
- **Missing data:** 30-50% in review/playtime features

These metrics feed directly into model design decisions (PU Learning, log1p transform, dual preprocessing paths).

---

## Questions?

Refer to:
- `Data-pipeline.md` — Detailed feature definitions
- `README.md` — Full project architecture
- Notebook comments — Code-level explanations
