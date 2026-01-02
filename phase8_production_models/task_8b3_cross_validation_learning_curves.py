"""
Task 8B.3: Cross-Validation and Learning Curves

Perform time-series cross-validation and generate learning curves
to understand model performance and potential overfitting.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8B.3: CROSS-VALIDATION & LEARNING CURVES")
print("="*120)

# Load data
print(f"\n[1/4] Loading data...")
df = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')

with open('../results/phase8_results/feature_categorization.json', 'r') as f:
    cat = json.load(f)

pre_game_dict = cat['pre_game_features']
pre_game_features = []
for category, features in pre_game_dict.items():
    pre_game_features.extend(features)

unknown_pregame = [
    'OTLosses', 'losses', 'pointsAgainst', 'pointsFor', 'ties', 'winPercent', 
    'winPercentage', 'wins', 'losses_roll3', 'losses_roll5', 'losses_std',
    'winPercent_roll3', 'winPercent_roll5', 'winPercent_std',
    'wins_roll3', 'wins_roll5', 'wins_std',
    'scored_20plus', 'scored_30plus', 'streak_20plus', 'streak_30plus',
    'vsconf_OTLosses', 'vsconf_leagueWinPercent', 'vsconf_losses', 'vsconf_ties', 'vsconf_wins',
    'vsdiv_OTLosses', 'vsdiv_divisionLosses', 'vsdiv_divisionTies', 
    'vsdiv_divisionWinPercent', 'vsdiv_divisionWins', 'vsdiv_losses', 'vsdiv_ties', 'vsdiv_wins',
    'div_game', 'rest_advantage', 'opponent'
]
pre_game_features.extend(unknown_pregame)

df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

# Use train+val for cross-validation (1999-2023)
train_val = df[df['season'] <= 2023].copy()

pregame_cols = []
for feat in pre_game_features:
    home_feat = f'home_{feat}'
    away_feat = f'away_{feat}'
    if home_feat in df.columns:
        pregame_cols.append(home_feat)
    if away_feat in df.columns:
        pregame_cols.append(away_feat)

numeric_pregame = df[pregame_cols].select_dtypes(include=[np.number]).columns.tolist()

train = df[df['season'] <= 2019].copy()
X_train_val = train_val[numeric_pregame].fillna(train[numeric_pregame].median())
y_train_val = train_val['home_win'].values

print(f"  ✅ Loaded train+val set: {len(y_train_val):,} games (1999-2023)")

# Time-series cross-validation
print(f"\n[2/4] Performing time-series cross-validation (5 folds)...")

# Load best hyperparameters
with open('../results/phase8_results/best_hyperparameters.json', 'r') as f:
    best_params = json.load(f)

# Define models with best hyperparameters
models = {
    'XGBoost': XGBClassifier(
        random_state=42,
        tree_method='hist',
        device='cuda:0',
        eval_metric='logloss',
        **best_params['XGBoost']
    ),
    'LightGBM': LGBMClassifier(
        random_state=42,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        verbose=-1,
        **best_params['LightGBM']
    ),
    'CatBoost': CatBoostClassifier(
        random_state=42,
        task_type='GPU',
        devices='0',
        verbose=False,
        **best_params['CatBoost']
    ),
    'RandomForest': RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **best_params['RandomForest']
    )
}

# Perform cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_results = {name: [] for name in models.keys()}

print(f"\n  Running cross-validation...")
for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
    print(f"  Fold {fold_idx + 1}/5...")
    
    X_tr, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_tr, y_val = y_train_val[train_idx], y_train_val[val_idx]
    
    for name, model in models.items():
        # Clone model to avoid reusing fitted model
        if name == 'XGBoost':
            m = XGBClassifier(random_state=42, tree_method='hist', device='cuda:0', 
                             eval_metric='logloss', **best_params['XGBoost'])
        elif name == 'LightGBM':
            m = LGBMClassifier(random_state=42, device='gpu', gpu_platform_id=0, 
                              gpu_device_id=0, verbose=-1, **best_params['LightGBM'])
        elif name == 'CatBoost':
            m = CatBoostClassifier(random_state=42, task_type='GPU', devices='0', 
                                  verbose=False, **best_params['CatBoost'])
        else:  # RandomForest
            m = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params['RandomForest'])
        
        m.fit(X_tr, y_tr)
        val_pred = m.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        cv_results[name].append(val_acc)

print(f"\n  Cross-Validation Results:")
print(f"  {'Model':<15} {'Mean CV Acc':<15} {'Std Dev':<15}")
print(f"  {'-'*45}")
for name, scores in cv_results.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"  {name:<15} {mean_score*100:>6.2f}%          {std_score*100:>6.2f}%")

# Learning curves
print(f"\n[3/4] Generating learning curves...")

# Use different training set sizes
train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_curve_results = {name: {'train_scores': [], 'val_scores': []} for name in models.keys()}

# Split into train (1999-2019) and val (2020-2023)
train_data = df[df['season'] <= 2019].copy()
val_data = df[(df['season'] >= 2020) & (df['season'] <= 2023)].copy()

X_train_full = train_data[numeric_pregame].fillna(train_data[numeric_pregame].median())
y_train_full = train_data['home_win'].values
X_val_full = val_data[numeric_pregame].fillna(train_data[numeric_pregame].median())
y_val_full = val_data['home_win'].values

print(f"  Generating learning curves for different training set sizes...")
for size in train_sizes:
    print(f"  Training with {int(size*100)}% of data...")

    # Sample training data
    n_samples = int(len(X_train_full) * size)
    X_tr = X_train_full.iloc[:n_samples]
    y_tr = y_train_full[:n_samples]

    for name in models.keys():
        # Clone model
        if name == 'XGBoost':
            m = XGBClassifier(random_state=42, tree_method='hist', device='cuda:0',
                             eval_metric='logloss', **best_params['XGBoost'])
        elif name == 'LightGBM':
            m = LGBMClassifier(random_state=42, device='gpu', gpu_platform_id=0,
                              gpu_device_id=0, verbose=-1, **best_params['LightGBM'])
        elif name == 'CatBoost':
            m = CatBoostClassifier(random_state=42, task_type='GPU', devices='0',
                                  verbose=False, **best_params['CatBoost'])
        else:  # RandomForest
            m = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params['RandomForest'])

        m.fit(X_tr, y_tr)

        train_pred = m.predict(X_tr)
        val_pred = m.predict(X_val_full)

        train_acc = accuracy_score(y_tr, train_pred)
        val_acc = accuracy_score(y_val_full, val_pred)

        learning_curve_results[name]['train_scores'].append(train_acc)
        learning_curve_results[name]['val_scores'].append(val_acc)

print(f"  ✅ Learning curves generated")

# Visualizations
print(f"\n[4/4] Creating visualizations...")

# 1. Cross-validation results
fig, ax = plt.subplots(figsize=(12, 6))

model_names = list(cv_results.keys())
mean_scores = [np.mean(cv_results[m]) * 100 for m in model_names]
std_scores = [np.std(cv_results[m]) * 100 for m in model_names]

x_pos = np.arange(len(model_names))
bars = ax.bar(x_pos, mean_scores, yerr=std_scores, capsize=5, color='steelblue', alpha=0.7)

# Highlight best model
best_idx = np.argmax(mean_scores)
bars[best_idx].set_color('darkgreen')
bars[best_idx].set_alpha(1.0)

ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('5-Fold Time-Series Cross-Validation Results', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for i, (mean, std) in enumerate(zip(mean_scores, std_scores)):
    ax.text(i, mean + std + 1, f'{mean:.2f}%\n±{std:.2f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../results/phase8_results/visualizations/cross_validation_results.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: cross_validation_results.png")
plt.close()

# 2. Learning curves
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Learning Curves - All Models', fontsize=16, fontweight='bold')

for idx, name in enumerate(models.keys()):
    ax = axes[idx // 2, idx % 2]

    train_scores = np.array(learning_curve_results[name]['train_scores']) * 100
    val_scores = np.array(learning_curve_results[name]['val_scores']) * 100
    train_sizes_pct = [int(s * 100) for s in train_sizes]

    ax.plot(train_sizes_pct, train_scores, 'o-', label='Training Accuracy', linewidth=2, markersize=8)
    ax.plot(train_sizes_pct, val_scores, 's-', label='Validation Accuracy', linewidth=2, markersize=8)

    ax.set_xlabel('Training Set Size (%)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'{name}')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # Add gap annotation
    final_gap = train_scores[-1] - val_scores[-1]
    ax.text(0.05, 0.95, f'Final Gap: {final_gap:.2f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('../results/phase8_results/visualizations/learning_curves.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: learning_curves.png")
plt.close()

# Save results
cv_results_json = {
    name: {
        'fold_scores': [float(s) for s in scores],
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores))
    }
    for name, scores in cv_results.items()
}

learning_curve_json = {
    name: {
        'train_sizes': train_sizes,
        'train_scores': [float(s) for s in results['train_scores']],
        'val_scores': [float(s) for s in results['val_scores']]
    }
    for name, results in learning_curve_results.items()
}

with open('../results/phase8_results/cross_validation_results.json', 'w') as f:
    json.dump(cv_results_json, f, indent=2)
print(f"  ✅ Saved: cross_validation_results.json")

with open('../results/phase8_results/learning_curves.json', 'w') as f:
    json.dump(learning_curve_json, f, indent=2)
print(f"  ✅ Saved: learning_curves.json")

print(f"\n{'='*120}")
print("✅ CROSS-VALIDATION & LEARNING CURVES COMPLETE")
print(f"{'='*120}")
print(f"\n✅ Performed 5-fold time-series cross-validation")
print(f"✅ Generated learning curves for 5 training set sizes")
print(f"✅ Created 2 visualization plots")
print(f"\n{'='*120}")

