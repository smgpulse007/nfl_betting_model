"""
Task 8A.1: GPU-Accelerated Hyperparameter Tuning for Tree-Based Models

Optimize XGBoost, LightGBM, CatBoost using GPU acceleration
RandomForest doesn't support GPU, so we'll use a smaller parameter grid
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
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8A.1: GPU-ACCELERATED HYPERPARAMETER TUNING")
print("="*120)

# Load data
print(f"\n[1/7] Loading data...")
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
print(f"\n[2/7] Splitting data...")
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

# Define parameter grids (smaller for faster tuning)
print(f"\n[3/7] Defining parameter grids...")

param_grids = {
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50],
        'min_child_samples': [20, 30],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'CatBoost': {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5]
    },
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
}

print(f"  ✅ Parameter grids defined for 4 models")

# Time series cross-validation (3 splits for speed)
tscv = TimeSeriesSplit(n_splits=3)

# Train models with GPU acceleration
print(f"\n[4/7] GPU-accelerated hyperparameter tuning...")
print(f"  ℹ️  Using GPU for XGBoost, LightGBM, and CatBoost")
print(f"  ℹ️  Using CPU for RandomForest (no GPU support)")

models = {
    'XGBoost': XGBClassifier(
        random_state=42,
        tree_method='hist',  # Use hist for GPU
        device='cuda:0',  # GPU acceleration (XGBoost 3.1+ syntax)
        eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        random_state=42,
        device='gpu',  # GPU acceleration
        gpu_platform_id=0,
        gpu_device_id=0,
        verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        random_state=42,
        task_type='GPU',  # GPU acceleration
        devices='0',
        verbose=False
    ),
    'RandomForest': RandomForestClassifier(
        random_state=42,
        n_jobs=-1  # CPU parallelization
    )
}

best_params = {}
best_models = {}
tuning_results = {}

for name, model in models.items():
    print(f"\n  Tuning {name}...")

    # CatBoost has sklearn compatibility issues, so we'll use manual grid search
    if name == 'CatBoost':
        from itertools import product

        # Generate all parameter combinations
        param_names = list(param_grids[name].keys())
        param_values = [param_grids[name][k] for k in param_names]
        all_combinations = list(product(*param_values))

        # Randomly sample 15 combinations
        np.random.seed(42)
        sampled_indices = np.random.choice(len(all_combinations), min(15, len(all_combinations)), replace=False)
        sampled_combinations = [all_combinations[i] for i in sampled_indices]

        best_score = 0
        best_params_catboost = None
        best_model_catboost = None

        print(f"    Testing {len(sampled_combinations)} parameter combinations...")

        for combo in sampled_combinations:
            params = dict(zip(param_names, combo))

            # Manual cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train_val):
                X_tr, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
                y_tr, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

                cb_model = CatBoostClassifier(
                    random_state=42,
                    task_type='GPU',
                    devices='0',
                    verbose=False,
                    **params
                )
                cb_model.fit(X_tr, y_tr)
                cv_scores.append(accuracy_score(y_val, cb_model.predict(X_val)))

            avg_score = np.mean(cv_scores)

            if avg_score > best_score:
                best_score = avg_score
                best_params_catboost = params
                best_model_catboost = CatBoostClassifier(
                    random_state=42,
                    task_type='GPU',
                    devices='0',
                    verbose=False,
                    **params
                )
                best_model_catboost.fit(X_train_val, y_train_val)

        best_params[name] = best_params_catboost
        best_models[name] = best_model_catboost

        # Evaluate on test set
        test_acc = accuracy_score(y_test, best_model_catboost.predict(X_test))

        tuning_results[name] = {
            'best_cv_score': best_score,
            'test_accuracy': test_acc,
            'best_params': best_params_catboost
        }

        print(f"    ✅ Best CV score: {best_score*100:.2f}%")
        print(f"    ✅ Test accuracy: {test_acc*100:.2f}%")

    else:
        # RandomizedSearchCV for other models
        random_search = RandomizedSearchCV(
            model,
            param_grids[name],
            n_iter=15,  # Reduced from 20 for speed
            cv=tscv,
            scoring='accuracy',
            n_jobs=1 if name not in ['RandomForest'] else -1,  # GPU models use single job
            random_state=42,
            verbose=1
        )

        random_search.fit(X_train_val, y_train_val)

        best_params[name] = random_search.best_params_
        best_models[name] = random_search.best_estimator_

        # Evaluate on test set
        test_acc = accuracy_score(y_test, random_search.best_estimator_.predict(X_test))

        tuning_results[name] = {
            'best_cv_score': random_search.best_score_,
            'test_accuracy': test_acc,
            'best_params': random_search.best_params_
        }

        print(f"    ✅ Best CV score: {random_search.best_score_*100:.2f}%")
        print(f"    ✅ Test accuracy: {test_acc*100:.2f}%")

print(f"\n[5/7] Saving results...")

# Save best parameters
with open('../results/phase8_results/best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print(f"  ✅ Saved: ../results/phase8_results/best_hyperparameters.json")

# Save tuning results
with open('../results/phase8_results/tuning_results.json', 'w') as f:
    # Convert numpy types to native Python types for JSON serialization
    results_serializable = {}
    for model_name, results in tuning_results.items():
        results_serializable[model_name] = {
            'best_cv_score': float(results['best_cv_score']),
            'test_accuracy': float(results['test_accuracy']),
            'best_params': {k: (int(v) if isinstance(v, (np.integer, np.int64)) else
                               float(v) if isinstance(v, (np.floating, np.float64)) else v)
                           for k, v in results['best_params'].items()}
        }
    json.dump(results_serializable, f, indent=2)
print(f"  ✅ Saved: ../results/phase8_results/tuning_results.json")

# Save models
for name, model in best_models.items():
    joblib.dump(model, f'../models/{name.lower()}_tuned.pkl')
    print(f"  ✅ Saved: ../models/{name.lower()}_tuned.pkl")

print(f"\n[6/7] Performance summary...")
print(f"\n{'='*120}")
print("MODEL PERFORMANCE (AFTER HYPERPARAMETER TUNING)")
print(f"{'='*120}")
print(f"\n{'Model':<15} {'CV Score':<12} {'Test Accuracy':<15}")
print(f"{'-'*42}")
for name, results in tuning_results.items():
    print(f"{name:<15} {results['best_cv_score']*100:>6.2f}%     {results['test_accuracy']*100:>6.2f}%")

print(f"\n[7/7] Summary...")
print(f"\n{'='*120}")
print("✅ HYPERPARAMETER TUNING COMPLETE")
print(f"{'='*120}")
print(f"\n✅ All 4 models tuned using GPU acceleration")
print(f"✅ Best parameters saved to JSON")
print(f"✅ Models saved to ../models/")
print(f"✅ Ready for Task 8A.2 (PyTorch Neural Network)")
print(f"\n{'='*120}")

