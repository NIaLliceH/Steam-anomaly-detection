# Steam Anomaly Detection — Implementation Plan

## ⚠️ Cảnh báo quan trọng trước khi bắt đầu

> **Target Leakage (nhẹ) đã biết trước:**
> Heuristic labels ở Step 2 được xây từ các rules dựa trên `median_unlock_interval`,
> `top1_game_concentration`, `max_achievements_per_day` — chính là features được dùng
> để train model ở Step 5. Do đó ROC-AUC khi tuning **chỉ đo mức độ model bắt chước
> rules**, không phải khả năng phát hiện anomaly thực sự. Đây là trade-off chấp nhận được
> vì không có ground truth thực. Ghi chú rõ điều này trong notebook/báo cáo.

---

## Project Structure
```
steam_anomaly/
├── data/
│   ├── history.csv          # 647 MB
│   ├── reviews.csv          # 551 MB
│   ├── players.csv          # 18 MB
│   ├── purchased_games.csv  # 91 MB
│   ├── achievements.csv     # 144 MB
│   ├── private_steamids.csv # 4 MB
│   └── games.csv            # 17 MB
├── outputs/
│   ├── feature_matrix.csv
│   ├── heuristic_labels.csv
│   ├── tuning_results.csv
│   ├── ensemble_results.csv
│   ├── model_comparison.csv
│   └── plots/
├── src/
│   ├── load.py
│   ├── features.py
│   ├── models.py
│   └── evaluate.py
└── main.py
```

---

## Step 1 — Data Loading & Cleaning

### 1.1 Load với dtype tường minh (tiết kiệm RAM ~50%)

```python
import pandas as pd
import numpy as np
import json
import re

# --- history.csv (647 MB) ---
history = pd.read_csv(
    'data/history.csv',
    usecols=['playerid', 'achievementid', 'date_acquired'],
    dtype={'playerid': 'int32', 'achievementid': 'str'},
    parse_dates=False   # parse thủ công bên dưới để kiểm soát format
)

# --- players.csv ---
players = pd.read_csv(
    'data/players.csv',
    usecols=['playerid', 'country', 'created'],
    dtype={'playerid': 'int32', 'country': 'category'},
    parse_dates=False
)

# --- purchased_games.csv ---
purchased = pd.read_csv(
    'data/purchased_games.csv',
    usecols=['playerid', 'library'],
    dtype={'playerid': 'int32'}
)

# --- reviews.csv (551 MB) ---
reviews = pd.read_csv(
    'data/reviews.csv',
    usecols=['playerid', 'gameid', 'review', 'posted'],
    dtype={'playerid': 'int32', 'gameid': 'int32'}
)

# --- private_steamids ---
private_ids = pd.read_csv(
    'data/private_steamids.csv',
    dtype={'playerid': 'int32'}
)
private_set = set(private_ids['playerid'])
```

### 1.2 Parse datetime với format tường minh (nhanh hơn 5–10x)

```python
# BẮT BUỘC dùng format tường minh, không để pandas tự đoán
history['date_acquired'] = pd.to_datetime(
    history['date_acquired'],
    format='%Y-%m-%d %H:%M:%S',
    errors='coerce'     # NaT thay vì crash nếu có dòng lỗi
)
players['created'] = pd.to_datetime(
    players['created'],
    format='%Y-%m-%d %H:%M:%S',
    errors='coerce'
)

# Extract time components
history['hour']        = history['date_acquired'].dt.hour.astype('int8')
history['day_of_week'] = history['date_acquired'].dt.dayofweek.astype('int8')
history['date_only']   = history['date_acquired'].dt.date
```

### 1.3 Extract gameid an toàn (regex, không dùng split)

```python
# achievementid format: "12345_some_achievement_name"
# str.split('_')[0] SẼ LỖI nếu tên có dấu _
history['gameid'] = (
    history['achievementid']
    .str.extract(r'^(\d+)_')[0]
    .astype('int32')
)
```

### 1.4 Parse library nhanh (json thay vì ast.literal_eval)

```python
def parse_library_fast(s):
    """
    Nhanh hơn ast.literal_eval ~3x.
    Input: "[10, 20, 30]" hoặc "[]" hoặc NaN
    Output: list of int
    """
    if pd.isna(s):
        return []
    try:
        # Thay dấu nháy đơn (nếu có) thành nháy kép cho json
        return json.loads(s.replace("'", '"'))
    except Exception:
        return []

purchased['library_list'] = purchased['library'].apply(parse_library_fast)
purchased['library_size'] = purchased['library_list'].apply(len).astype('int32')

# Tạo dict để lookup nhanh: {playerid: set(gameids)}
player_library = {
    row['playerid']: set(row['library_list'])
    for _, row in purchased.iterrows()
}
```

### 1.5 Filter private accounts

```python
# Xóa private accounts khỏi tất cả dataframes
history  = history[~history['playerid'].isin(private_set)]
players  = players[~players['playerid'].isin(private_set)]
reviews  = reviews[~reviews['playerid'].isin(private_set)]
purchased = purchased[~purchased['playerid'].isin(private_set)]

# ĐỂ Ý: KHÔNG filter < 5 achievements ở đây
# Review bots có thể có 0 achievements — lọc sớm sẽ bỏ sót chúng
```

---

## Step 2 — Heuristic Labels (tạo TRƯỚC feature engineering)

> **Mục đích:** Tạo pseudo ground truth để dùng làm `y_true` cho hyperparameter tuning.
> Không phải ground truth thực — ROC-AUC từ đây chỉ mang nghĩa tương đối.

```python
# Tính các aggregates tối thiểu cần cho rules
# (sẽ tính lại đầy đủ ở Step 3, đây chỉ để tạo labels nhanh)

# Rule 1: Median unlock interval < 2 giây
speed_stats = (
    history.sort_values(['playerid', 'date_acquired'])
    .groupby('playerid')['date_acquired']
    .apply(lambda x: x.diff().dt.total_seconds().median())
    .rename('median_interval')
)

# Rule 2: Top-1 game concentration > 0.90
top1_conc = (
    history.groupby(['playerid', 'gameid'])
    .size()
    .groupby(level=0)
    .apply(lambda x: x.max() / x.sum())
    .rename('top1_concentration')
)

# Rule 3: Max achievements/day > 500
max_per_day = (
    history.groupby(['playerid', 'date_only'])
    .size()
    .groupby(level=0).max()
    .rename('max_per_day')
)

# Rule 4 & 5: Review behavior
review_counts   = reviews.groupby('playerid').size().rename('review_count')
ach_counts      = history.groupby('playerid').size().rename('ach_count')

def calc_unowned_ratio(pid):
    player_reviews = reviews[reviews['playerid'] == pid]
    if len(player_reviews) == 0:
        return 0.0
    lib = player_library.get(pid, set())
    unowned = (~player_reviews['gameid'].isin(lib)).sum()
    return unowned / len(player_reviews)

# Chỉ tính unowned_ratio cho players có reviews (tiết kiệm thời gian)
players_with_reviews = reviews['playerid'].unique()
unowned_ratio = pd.Series(
    {pid: calc_unowned_ratio(pid) for pid in players_with_reviews},
    name='unowned_ratio'
)

# Ghép tất cả vào bảng heuristic
heuristic_df = pd.concat([
    speed_stats, top1_conc, max_per_day,
    review_counts, ach_counts, unowned_ratio
], axis=1).fillna(0)

heuristic_df['heuristic_bot'] = (
    (heuristic_df['median_interval'] < 2) |
    (heuristic_df['top1_concentration'] > 0.90) |
    (heuristic_df['max_per_day'] > 500) |
    ((heuristic_df['review_count'] > 0) & (heuristic_df['ach_count'] == 0)) |
    (heuristic_df['unowned_ratio'] > 0.80)
).astype(int)

heuristic_df[['heuristic_bot']].to_csv('outputs/heuristic_labels.csv')
print(f"Heuristic bots: {heuristic_df['heuristic_bot'].sum()} "
      f"({heuristic_df['heuristic_bot'].mean()*100:.2f}%)")
```

---

## Step 3 — Feature Engineering (per-player aggregation)

```python
from scipy.stats import entropy as scipy_entropy

all_players = set(history['playerid'].unique()) | set(reviews['playerid'].unique())
feature_rows = []

for pid in all_players:
    ph = history[history['playerid'] == pid].copy()
    pr = reviews[reviews['playerid'] == pid]
    lib_size = purchased.loc[purchased['playerid'] == pid, 'library_size']
    lib_size = lib_size.values[0] if len(lib_size) > 0 else 0
    row = {'playerid': pid}

    # === GROUP A: Speed Features ===
    if len(ph) >= 2:
        ph = ph.sort_values('date_acquired')
        diffs = ph['date_acquired'].diff().dt.total_seconds().dropna()
        row['median_unlock_interval_sec'] = diffs.median()
        row['min_unlock_interval_sec']    = diffs.min()
        row['std_unlock_interval_sec']    = diffs.std()
        mean_diff = diffs.mean()
        row['cv_unlock_interval'] = (
            diffs.std() / mean_diff if mean_diff > 0 else 0
        )
        row['max_achievements_per_minute'] = (
            ph.groupby(ph['date_acquired'].dt.floor('min')).size().max()
        )
        row['max_achievements_per_day'] = (
            ph.groupby('date_only').size().max()
        )
    else:
        for col in ['median_unlock_interval_sec', 'min_unlock_interval_sec',
                    'std_unlock_interval_sec', 'cv_unlock_interval',
                    'max_achievements_per_minute', 'max_achievements_per_day']:
            row[col] = np.nan

    # === GROUP B: Temporal Features ===
    if len(ph) > 0:
        hour_dist = ph['hour'].value_counts(normalize=True)
        row['night_activity_ratio'] = hour_dist[
            hour_dist.index.isin(range(0, 6))
        ].sum()
        full_hour_dist = hour_dist.reindex(range(24), fill_value=0)
        row['hour_entropy'] = scipy_entropy(full_hour_dist)

        active_days  = ph['date_only'].nunique()
        span_days    = max(
            (ph['date_acquired'].max() - ph['date_acquired'].min()).days + 1,
            1
        )
        row['activity_density'] = active_days / span_days
        row['weekend_ratio']    = ph['day_of_week'].isin([5, 6]).mean()
    else:
        row.update({
            'night_activity_ratio': np.nan,
            'hour_entropy': np.nan,
            'activity_density': np.nan,
            'weekend_ratio': np.nan
        })

    # === GROUP C: Diversity Features ===
    total_ach   = len(ph)
    games_played = ph['gameid'].nunique() if len(ph) > 0 else 0
    row['total_achievements']    = total_ach
    row['games_with_achievements'] = games_played
    row['library_size']          = lib_size
    row['achievement_game_ratio'] = games_played / max(lib_size, 1)

    if len(ph) > 0 and games_played > 0:
        game_props = ph['gameid'].value_counts(normalize=True)
        row['top1_game_concentration'] = game_props.iloc[0]
        row['top3_game_concentration'] = game_props.iloc[:3].sum()
        row['game_hhi']                = (game_props ** 2).sum()
        row['avg_achievements_per_game'] = total_ach / games_played
    else:
        row.update({
            'top1_game_concentration': np.nan,
            'top3_game_concentration': np.nan,
            'game_hhi': np.nan,
            'avg_achievements_per_game': np.nan
        })

    # === GROUP D: Review Features ===
    row['total_reviews'] = len(pr)
    if len(pr) > 0:
        lib = player_library.get(pid, set())
        row['review_unowned_ratio'] = (
            (~pr['gameid'].isin(lib)).sum() / len(pr)
        )
        pr_clean = pr['review'].fillna('').str.lower().str.strip()
        row['review_duplication_rate'] = (
            pr_clean.duplicated().sum() / len(pr)
        )
        review_lens = pr['review'].fillna('').str.len()
        row['avg_review_length'] = review_lens.mean()
        row['min_review_length'] = review_lens.min()
    else:
        row.update({
            'review_unowned_ratio': 0.0,
            'review_duplication_rate': 0.0,
            'avg_review_length': 0.0,
            'min_review_length': 0.0
        })

    # === BONUS: Account Age ===
    player_info = players[players['playerid'] == pid]
    if len(player_info) > 0 and len(ph) > 0:
        created = player_info['created'].values[0]
        row['days_before_first_achievement'] = max(
            (ph['date_acquired'].min() - pd.Timestamp(created)).days, 0
        )
        row['account_age_days'] = (
            pd.Timestamp.now() - pd.Timestamp(created)
        ).days
    else:
        row['days_before_first_achievement'] = np.nan
        row['account_age_days'] = np.nan

    feature_rows.append(row)

feature_matrix = pd.DataFrame(feature_rows).set_index('playerid')
feature_matrix.to_csv('outputs/feature_matrix.csv')
```

---

## Step 4 — Preprocessing

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import joblib

# Align với heuristic labels (chỉ giữ players có trong cả hai)
common_ids = feature_matrix.index.intersection(heuristic_df.index)
X_raw = feature_matrix.loc[common_ids]
y_heuristic = heuristic_df.loc[common_ids, 'heuristic_bot']

feature_names = X_raw.columns.tolist()

preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())      # Dùng RobustScaler, KHÔNG StandardScaler
])

X_scaled = preprocessor.fit_transform(X_raw)
joblib.dump(preprocessor, 'outputs/preprocessor.pkl')
```

---

## Step 5 — Hyperparameter Tuning

> Metric: `roc_auc_score(y_heuristic, anomaly_scores)`
> Nhắc lại: ROC-AUC ở đây đo "model bắt chước rule tốt thế nào", **không phải** accuracy thực.

### 5.1 Isolation Forest (train toàn bộ data)

```python
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
import time

param_grid_if = {
    'n_estimators':  [100, 200, 300],
    'max_samples':   ['auto', 0.8, 0.6],
    'contamination': [0.02, 0.05, 0.10],
    'max_features':  [0.8, 1.0],
    'random_state':  [42]
}

results_if = []
for params in ParameterGrid(param_grid_if):
    t0 = time.time()
    model = IsolationForest(**params)
    model.fit(X_scaled)
    scores = -model.score_samples(X_scaled)   # higher = more anomalous
    auc = roc_auc_score(y_heuristic, scores)
    results_if.append({**params, 'roc_auc': auc, 'runtime': time.time()-t0})

results_if_df = pd.DataFrame(results_if).sort_values('roc_auc', ascending=False)
best_if_params = results_if_df.iloc[0].drop(['roc_auc', 'runtime']).to_dict()
print("Best IF params:", best_if_params)
```

### 5.2 LOF (sample 20,000 users — novelty=True bắt buộc)

```python
from sklearn.neighbors import LocalOutlierFactor

# Sample để tránh OOM (LOF là O(N²))
N_SAMPLE = 20_000
sample_idx = np.random.choice(len(X_scaled), min(N_SAMPLE, len(X_scaled)), replace=False)
X_sample = X_scaled[sample_idx]

param_grid_lof = {
    'n_neighbors':   [10, 20, 30, 50],
    'contamination': [0.02, 0.05, 0.10],
    'metric':        ['euclidean', 'manhattan']
}

results_lof = []
for params in ParameterGrid(param_grid_lof):
    t0 = time.time()
    # novelty=True BẮT BUỘC để gọi .predict() trên data mới
    model = LocalOutlierFactor(novelty=True, **params)
    model.fit(X_sample)
    scores = -model.score_samples(X_scaled)   # predict trên toàn bộ
    auc = roc_auc_score(y_heuristic, scores)
    results_lof.append({**params, 'roc_auc': auc, 'runtime': time.time()-t0})

results_lof_df = pd.DataFrame(results_lof).sort_values('roc_auc', ascending=False)
best_lof_params = results_lof_df.iloc[0].drop(['roc_auc', 'runtime']).to_dict()
```

### 5.3 One-Class SVM (sample 20,000 users)

```python
from sklearn.svm import OneClassSVM

param_grid_ocsvm = {
    'kernel': ['rbf'],
    'gamma':  ['scale', 'auto', 0.001, 0.01],
    'nu':     [0.01, 0.05, 0.10]
}

results_svm = []
for params in ParameterGrid(param_grid_ocsvm):
    t0 = time.time()
    model = OneClassSVM(**params)
    model.fit(X_sample)                        # train trên sample
    scores = -model.score_samples(X_scaled)    # predict toàn bộ
    auc = roc_auc_score(y_heuristic, scores)
    results_svm.append({**params, 'roc_auc': auc, 'runtime': time.time()-t0})

results_svm_df = pd.DataFrame(results_svm).sort_values('roc_auc', ascending=False)
best_svm_params = results_svm_df.iloc[0].drop(['roc_auc', 'runtime']).to_dict()

# Save tất cả tuning results
pd.concat([
    results_if_df.assign(model='IsolationForest'),
    results_lof_df.assign(model='LOF'),
    results_svm_df.assign(model='OCSVM')
]).to_csv('outputs/tuning_results.csv', index=False)
```

---

## Step 6 — Train Best Models & Collect Scores

```python
# Train final models với best params
best_if = IsolationForest(**best_if_params)
best_if.fit(X_scaled)
if_scores = -best_if.score_samples(X_scaled)

best_lof = LocalOutlierFactor(novelty=True, **best_lof_params)
best_lof.fit(X_sample)
lof_scores = -best_lof.score_samples(X_scaled)

best_svm = OneClassSVM(**best_svm_params)
best_svm.fit(X_sample)
svm_scores = -best_svm.score_samples(X_scaled)
```

---

## Step 7 — Ensemble Voting

```python
from scipy.stats import rankdata

# Percentile Ranking — robust hơn MinMax với outliers
# higher percentile = more suspicious
if_pct  = rankdata(if_scores)  / len(if_scores) * 100
lof_pct = rankdata(lof_scores) / len(lof_scores) * 100
svm_pct = rankdata(svm_scores) / len(svm_scores) * 100

# Weighted composite score (0–100, higher = more suspicious)
composite = 0.5 * if_pct + 0.3 * lof_pct + 0.2 * svm_pct

# Voting: flag nếu trong top 5% của model đó
if_flag  = (if_pct  >= 95).astype(int)
lof_flag = (lof_pct >= 95).astype(int)
svm_flag = (svm_pct >= 95).astype(int)
vote_count = if_flag + lof_flag + svm_flag

# Final: flagged bởi ít nhất 2/3 models
is_anomaly = (vote_count >= 2)

ensemble_results = pd.DataFrame({
    'playerid':        common_ids,
    'composite_score': composite,
    'vote_count':      vote_count,
    'is_anomaly':      is_anomaly,
    'if_flag':         if_flag,
    'lof_flag':        lof_flag,
    'svm_flag':        svm_flag,
    'heuristic_bot':   y_heuristic.values
})
ensemble_results.to_csv('outputs/ensemble_results.csv', index=False)
```

---

## Step 8 — Evaluation

> Không dùng Silhouette/Calinski (clustering metrics, vô nghĩa cho anomaly detection)

### 8.1 ROC-AUC per model

```python
from sklearn.metrics import roc_auc_score, classification_report

metrics = {
    'IsolationForest': roc_auc_score(y_heuristic, if_pct),
    'LOF':             roc_auc_score(y_heuristic, lof_pct),
    'OCSVM':           roc_auc_score(y_heuristic, svm_pct),
    'Ensemble':        roc_auc_score(y_heuristic, composite)
}
print("ROC-AUC (vs heuristic labels):", metrics)
```

### 8.2 Precision@K

```python
def precision_at_k(y_true, scores, k):
    """Tỷ lệ heuristic bots trong top-K accounts bị flag nặng nhất"""
    top_k_idx = np.argsort(scores)[::-1][:k]
    return y_true.values[top_k_idx].mean()

for k in [100, 500, 1000]:
    p = precision_at_k(y_heuristic, composite, k)
    print(f"Precision@{k}: {p:.4f}")
```

### 8.3 Model Comparison Table

```python
flagged_rates = {
    'IsolationForest': if_flag.mean(),
    'LOF':             lof_flag.mean(),
    'OCSVM':           svm_flag.mean(),
    'Ensemble':        is_anomaly.mean()
}

comparison = pd.DataFrame({
    'ROC-AUC':        metrics,
    'Flagged Rate %':  {k: v*100 for k, v in flagged_rates.items()},
    'Precision@100':  {m: precision_at_k(y_heuristic,
                        {'IsolationForest': if_pct,
                         'LOF': lof_pct,
                         'OCSVM': svm_pct,
                         'Ensemble': composite}[m], 100)
                       for m in metrics},
    'Precision@500':  {m: precision_at_k(y_heuristic,
                        {'IsolationForest': if_pct,
                         'LOF': lof_pct,
                         'OCSVM': svm_pct,
                         'Ensemble': composite}[m], 500)
                       for m in metrics},
})
print(comparison.round(4))
comparison.to_csv('outputs/model_comparison.csv')
```

### 8.4 Top-50 Flagged Profile Analysis

```python
top50_ids = ensemble_results.nlargest(50, 'composite_score')['playerid']
flagged_feats  = feature_matrix.loc[feature_matrix.index.isin(top50_ids)]
normal_feats   = feature_matrix.loc[~feature_matrix.index.isin(top50_ids)]

profile_comparison = pd.DataFrame({
    'Flagged (mean)': flagged_feats.mean(),
    'Normal (mean)':  normal_feats.mean(),
    'Ratio':          flagged_feats.mean() / normal_feats.mean().replace(0, np.nan)
}).round(3)

print(profile_comparison)
profile_comparison.to_csv('outputs/top50_flagged_profiles.csv')
```

---

## Step 9 — SHAP Explanation

> ⚠️ Chỉ dùng subset 10,000 samples để tính SHAP — tránh treo máy

```python
import shap
import matplotlib.pyplot as plt

# Subsample để tính SHAP (không cần toàn bộ data)
SHAP_SAMPLE = 10_000
shap_idx = np.random.choice(len(X_scaled), SHAP_SAMPLE, replace=False)
X_shap   = X_scaled[shap_idx]

# TreeExplainer tương thích với IsolationForest
explainer   = shap.TreeExplainer(best_if)
shap_values = explainer.shap_values(X_shap)

# Plot 1: Feature Importance (bar)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    pd.DataFrame(X_shap, columns=feature_names),
    plot_type='bar',
    show=False
)
plt.tight_layout()
plt.savefig('outputs/plots/shap_summary.png', dpi=150)
plt.close()

# Plot 2: Waterfall cho account suspicious nhất
top1_global_idx = np.argmax(composite)
# Map về shap sample nếu top1 nằm trong sample
shap.waterfall_plot(
    shap.Explanation(
        values      = shap_values[0],   # dùng sample đầu tiên làm demo
        base_values = explainer.expected_value,
        data        = X_shap[0],
        feature_names = feature_names
    ),
    show=False
)
plt.savefig('outputs/plots/shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Scatter của top-3 features
top3_features = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(shap_values).mean(0)
}).nlargest(3, 'importance')['feature'].tolist()

for feat in top3_features:
    feat_idx = feature_names.index(feat)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_shap[:, feat_idx], shap_values[:, feat_idx], alpha=0.3, s=5)
    ax.set_xlabel(feat)
    ax.set_ylabel('SHAP value')
    ax.set_title(f'SHAP scatter: {feat}')
    plt.tight_layout()
    plt.savefig(f'outputs/plots/shap_scatter_{feat}.png', dpi=150)
    plt.close()
```

---

## Deliverables Checklist

```
outputs/
├── feature_matrix.csv          ← engineered features per player
├── heuristic_labels.csv        ← pseudo ground truth
├── tuning_results.csv          ← all hyperparameter search results
├── ensemble_results.csv        ← final scores + flags per player
├── model_comparison.csv        ← metrics table (ROC-AUC, Precision@K)
├── top50_flagged_profiles.csv  ← feature comparison flagged vs normal
├── preprocessor.pkl            ← fitted scaler để reuse
└── plots/
    ├── shap_summary.png
    ├── shap_waterfall.png
    └── shap_scatter_<feature>.png  (top 3 features)
```

---

## Dependencies

```txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.10
shap>=0.44
matplotlib>=3.7
joblib>=1.3
```

Install:
```bash
pip install pandas numpy scikit-learn scipy shap matplotlib joblib
```
