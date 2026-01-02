"""
Task 8A.1: Hyperparameter Tuning for Tree-Based Models

Optimize XGBoost, LightGBM, CatBoost, and RandomForest using RandomizedSearchCV
"""

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8A.1: HYPERPARAMETER TUNING")
print("="*120)

# Load data
print(f"\n[1/6] Loading data...")
df = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')
print(f"  ✅ Loaded: {df.shape[0]:,} games × {df.shape[1]:,} columns")

# Load feature categorization
with open('../results/phase8_results/feature_categorization.json', 'r') as f:
    cat = json.load(f)

# Get pre-game features
pre_game_dict = cat['pre_game_features']
pre_game_features = []
for category, features in pre_game_dict.items():
    pre_game_features.extend(features)

# Add manually classified UNKNOWN features (cumulative stats from prior games)
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

# Create target
df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

# Split by season
print(f"\n[2/6] Splitting data...")
train = df[df['season'] <= 2019].copy()
val = df[(df['season'] >= 2020) & (df['season'] <= 2023)].copy()
test = df[df['season'] == 2024].copy()

# Combine train + val for hyperparameter tuning
train_val = pd.concat([train, val], axis=0)

print(f"  ✅ Train+Val (1999-2023): {len(train_val):,} games")
print(f"  ✅ Test (2024): {len(test):,} games")

# Select pre-game features
pregame_cols = []
for feat in pre_game_features:
    home_feat = f'home_{feat}'
    away_feat = f'away_{feat}'
    if home_feat in df.columns:
        pregame_cols.append(home_feat)
    if away_feat in df.columns:
        pregame_cols.append(away_feat)

numeric_pregame = df[pregame_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"  ✅ Total pre-game features: {len(numeric_pregame)}")

# Prepare data
X_train_val = train_val[numeric_pregame].fillna(train_val[numeric_pregame].median())
X_test = test[numeric_pregame].fillna(train_val[numeric_pregame].median())
y_train_val = train_val['home_win']
y_test = test['home_win']

# Define parameter grids
print(f"\n[3/6] Defining parameter grids...")

param_grids = {
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 40, 50],
        'min_child_samples': [10, 20, 30],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    },
    'CatBoost': {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7],
        'border_count': [32, 64, 128]
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [8, 10, 12, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
}

print(f"  ✅ Parameter grids defined for 4 models")

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Train models
print(f"\n[4/6] Hyperparameter tuning (this may take 1-2 hours)...")

models = {
    'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1)
}

best_params = {}
best_models = {}

for name, model in models.items():
    print(f"\n  Tuning {name}...")
    print(f"    Parameter space: {sum(len(v) for v in param_grids[name].values())} combinations")
    
    # RandomizedSearchCV (faster than GridSearchCV)
    random_search = RandomizedSearchCV(
        model, 
        param_grids[name],
        n_iter=20,  # Try 20 random combinations
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    random_search.fit(X_train_val, y_train_val)
    
    best_params[name] = random_search.best_params_
    best_models[name] = random_search.best_estimator_
    
    # Evaluate on test set
    test_acc = accuracy_score(y_test, random_search.best_estimator_.predict(X_test))
    
    print(f"    ✅ Best CV score: {random_search.best_score_*100:.2f}%")
    print(f"    ✅ Test accuracy: {test_acc*100:.2f}%")
    print(f"    ✅ Best params: {random_search.best_params_}")

# Save results
print(f"\n[5/6] Saving results...")

# Save best parameters
with open('../results/phase8_results/best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print(f"  ✅ Saved: ../results/phase8_results/best_hyperparameters.json")

# Save models
for name, model in best_models.items():
    joblib.dump(model, f'../models/{name.lower()}_tuned.pkl')
    print(f"  ✅ Saved: ../models/{name.lower()}_tuned.pkl")

print(f"\n[6/6] Summary...")
print(f"\n{'='*120}")
print("HYPERPARAMETER TUNING COMPLETE")
print(f"{'='*120}")
print(f"\n✅ All 4 models tuned and saved")
print(f"✅ Best parameters saved to JSON")
print(f"✅ Models ready for ensemble")
print(f"\n{'='*120}")

