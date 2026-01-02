"""
Retrain XGBoost with Injury + Weather + Advanced Features
Test to validate improvement before retraining all models
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import joblib
import json

print("="*120)
print("RETRAIN XGBOOST WITH INJURY + WEATHER + ADVANCED FEATURES")
print("="*120)

# Load data with injury features (clean, no leakage)
print(f"\n[1/8] Loading phase6_game_level with injury + weather features...")
df = pd.read_parquet('../results/phase8_results/phase6_game_level_with_injury_weather_1999_2024.parquet')
print(f"  ✅ Loaded: {len(df):,} games × {len(df.columns):,} columns")

# Check injury features
injury_cols = [c for c in df.columns if 'injury' in c.lower() or 'qb_out' in c.lower()]
weather_cols = [c for c in df.columns if c in ['temp', 'wind', 'temp_extreme', 'wind_high', 'is_outdoor']]
print(f"  ✅ Injury features: {len(injury_cols)}")
print(f"  ✅ Weather features: {len(weather_cols)}")

# Create target
df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

# Split by season
print(f"\n[2/8] Splitting data...")
train = df[df['season'] <= 2019].copy()
val = df[(df['season'] >= 2020) & (df['season'] <= 2023)].copy()
test = df[df['season'] == 2024].copy()

print(f"  ✅ Train (1999-2019): {len(train):,} games")
print(f"  ✅ Val (2020-2023): {len(val):,} games")
print(f"  ✅ Test (2024): {len(test):,} games")

# Load feature categorization to get pre-game features only
with open('../results/phase8_results/feature_categorization.json', 'r') as f:
    cat = json.load(f)

# Get pre-game features
pre_game_dict = cat['pre_game_features']
pre_game_features = []
for category, features in pre_game_dict.items():
    pre_game_features.extend(features)

# Add manually classified UNKNOWN features (from Phase 8A)
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

# Select pre-game features (home_ and away_ prefixed)
pregame_cols = []
for feat in pre_game_features:
    home_feat = f'home_{feat}'
    away_feat = f'away_{feat}'
    if home_feat in df.columns:
        pregame_cols.append(home_feat)
    if away_feat in df.columns:
        pregame_cols.append(away_feat)

# Add injury and weather features
injury_weather_features = [c for c in df.columns if
                          ('injury' in c.lower() or 'qb_out' in c.lower() or
                           c in ['temp', 'wind', 'temp_extreme', 'wind_high', 'is_outdoor'])]

pregame_cols.extend(injury_weather_features)

# Remove duplicates
pregame_cols = list(set(pregame_cols))

# Get numeric features only
numeric_features = df[pregame_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"\n[3/8] Feature selection...")
print(f"  ✅ Total numeric features: {len(numeric_features)}")

# Show injury features being used
injury_features_used = [f for f in numeric_features if 'injury' in f.lower() or 'qb_out' in f.lower()]
weather_features_used = [f for f in numeric_features if f in weather_cols]

print(f"  ✅ Injury features being used: {len(injury_features_used)}")
for feat in sorted(injury_features_used):
    print(f"     • {feat}")

print(f"  ✅ Weather features being used: {len(weather_features_used)}")
for feat in sorted(weather_features_used):
    print(f"     • {feat}")

# Prepare data
print(f"\n[4/8] Preparing data...")
X_train = train[numeric_features].fillna(train[numeric_features].median())
X_val = val[numeric_features].fillna(train[numeric_features].median())
X_test = test[numeric_features].fillna(train[numeric_features].median())
y_train = train['home_win'].values
y_val = val['home_win'].values
y_test = test['home_win'].values

print(f"  ✅ X_train: {X_train.shape}")
print(f"  ✅ X_val: {X_val.shape}")
print(f"  ✅ X_test: {X_test.shape}")

# Train XGBoost
print(f"\n[5/8] Training XGBoost with injury features...")

# Use best hyperparameters from Phase 8A
best_params = {
    'max_depth': 4,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'device': 'cuda'
}

model = xgb.XGBClassifier(**best_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"  ✅ Training complete!")

# Evaluate
print(f"\n[6/8] Evaluating model...")

# Train set
train_pred_proba = model.predict_proba(X_train)[:, 1]
train_pred = (train_pred_proba >= 0.5).astype(int)
train_acc = accuracy_score(y_train, train_pred)
train_logloss = log_loss(y_train, train_pred_proba)
train_auc = roc_auc_score(y_train, train_pred_proba)

# Validation set
val_pred_proba = model.predict_proba(X_val)[:, 1]
val_pred = (val_pred_proba >= 0.5).astype(int)
val_acc = accuracy_score(y_val, val_pred)
val_logloss = log_loss(y_val, val_pred_proba)
val_auc = roc_auc_score(y_val, val_pred_proba)

# Test set
test_pred_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_pred_proba >= 0.5).astype(int)
test_acc = accuracy_score(y_test, test_pred)
test_logloss = log_loss(y_test, test_pred_proba)
test_auc = roc_auc_score(y_test, test_pred_proba)

print(f"\n  Performance Metrics:")
print(f"  {'Set':<12} {'Accuracy':<12} {'Log Loss':<12} {'AUC':<12}")
print(f"  {'-'*48}")
print(f"  {'Train':<12} {train_acc:<12.4f} {train_logloss:<12.4f} {train_auc:<12.4f}")
print(f"  {'Validation':<12} {val_acc:<12.4f} {val_logloss:<12.4f} {val_auc:<12.4f}")
print(f"  {'Test (2024)':<12} {test_acc:<12.4f} {test_logloss:<12.4f} {test_auc:<12.4f}")

# Compare with old model (if exists)
print(f"\n[7/8] Comparing with old model (no injuries)...")
try:
    old_model = joblib.load('../models/xgboost_model.pkl')
    # Get common features between old and new
    common_features = [f for f in old_model.feature_names_in_ if f in numeric_features]
    old_test_pred_proba = old_model.predict_proba(X_test[common_features])[:, 1]
    old_test_pred = (old_test_pred_proba >= 0.5).astype(int)
    old_test_acc = accuracy_score(y_test, old_test_pred)

    print(f"\n  2024 Test Set Comparison:")
    print(f"  {'Model':<30} {'Accuracy':<12} {'Improvement':<12}")
    print(f"  {'-'*54}")
    print(f"  {'Old (no injuries)':<30} {old_test_acc:<12.4f} {'-':<12}")
    print(f"  {'New (with injuries)':<30} {test_acc:<12.4f} {f'+{(test_acc - old_test_acc)*100:.2f}%':<12}")
except FileNotFoundError:
    print(f"  ⚠️  Old model not found, skipping comparison")
    old_test_acc = 0.0

# Save new model
print(f"\n[8/8] Saving model...")
joblib.dump(model, '../models/xgboost_with_injuries.pkl')
print(f"  ✅ Saved: ../models/xgboost_with_injuries.pkl")

# Save feature list
with open('../models/xgboost_with_injuries_features.json', 'w') as f:
    json.dump({
        'features': numeric_features,
        'injury_features': injury_features_used,
        'weather_features': weather_features_used,
        'n_features': len(numeric_features),
        'test_accuracy': float(test_acc),
        'old_test_accuracy': float(old_test_acc),
        'improvement': float(test_acc - old_test_acc)
    }, f, indent=2)

print(f"  ✅ Saved: ../models/xgboost_with_injuries_features.json")

print(f"\n{'='*120}")
print("TRAINING COMPLETE!")
print("="*120)
print(f"\n✅ New XGBoost model trained with {len(numeric_features)} features")
print(f"✅ Includes {len(injury_features_used)} injury features")
print(f"✅ Includes {len(weather_features_used)} weather features")
print(f"✅ Test accuracy: {test_acc:.4f} (vs {old_test_acc:.4f} old)")
print(f"✅ Improvement: +{(test_acc - old_test_acc)*100:.2f}%")
print(f"\n{'='*120}")

