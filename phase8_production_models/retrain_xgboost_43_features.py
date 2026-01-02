"""
Retrain XGBoost with ONLY 43 Features Available in 2025 Dataset

This ensures perfect feature alignment between training and prediction.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import joblib
import json
from pathlib import Path

print("="*120)
print("RETRAIN XGBOOST WITH 43 FEATURES (FEATURE ALIGNMENT FIX)")
print("="*120)

# Load training data with injury + weather
print("\n[1/6] Loading training data...")
df = pd.read_parquet('../results/phase8_results/phase6_game_level_with_injury_weather_1999_2024.parquet')

print(f"  ✅ Loaded: {df.shape}")

# Load 2025 dataset to get exact feature list
print("\n[2/6] Loading 2025 dataset to identify available features...")
df_2025 = pd.read_parquet('../results/phase8_results/2025_complete_dataset_weeks1_17.parquet')

# Exclude metadata columns (scores, teams, etc.)
metadata_cols = ['home_score', 'away_score', 'home_team', 'away_team', 'home_win', 'away_win']

# Get features that exist in 2025 data (exclude metadata)
available_2025_features = [c for c in df_2025.columns
                          if (c.startswith('home_') or c.startswith('away_'))
                          and c not in metadata_cols]
print(f"  ✅ Features in 2025 data: {len(available_2025_features)}")

# Extract base feature names (remove home_/away_ prefix)
base_features = set()
for feat in available_2025_features:
    if feat.startswith('home_'):
        base_features.add(feat[5:])  # Remove 'home_'
    elif feat.startswith('away_'):
        base_features.add(feat[5:])  # Remove 'away_'

base_features = sorted(list(base_features))
print(f"  ✅ Base features: {len(base_features)}")
print(f"     {base_features[:10]}...")

# Build feature list for training (numeric only)
training_features = []
for feat in base_features:
    home_feat = f'home_{feat}'
    away_feat = f'away_{feat}'

    if home_feat in df.columns and df[home_feat].dtype in ['int64', 'float64', 'int32', 'float32']:
        training_features.append(home_feat)
    if away_feat in df.columns and df[away_feat].dtype in ['int64', 'float64', 'int32', 'float32']:
        training_features.append(away_feat)

print(f"  ✅ Training features (numeric only): {len(training_features)}")

# Prepare data
print("\n[3/6] Preparing train/val/test splits...")

# Create target
df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

# Split by year
train_df = df[df['season'] <= 2019].copy()
val_df = df[(df['season'] >= 2020) & (df['season'] <= 2023)].copy()
test_df = df[df['season'] == 2024].copy()

# Prepare X, y
X_train = train_df[training_features].fillna(0)
y_train = train_df['home_win']

X_val = val_df[training_features].fillna(0)
y_val = val_df['home_win']

X_test = test_df[training_features].fillna(0)
y_test = test_df['home_win']

print(f"  ✅ Train: {X_train.shape}, {y_train.value_counts().to_dict()}")
print(f"  ✅ Val:   {X_val.shape}, {y_val.value_counts().to_dict()}")
print(f"  ✅ Test:  {X_test.shape}, {y_test.value_counts().to_dict()}")

# Train model
print("\n[4/6] Training XGBoost...")

params = {
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

model = xgb.XGBClassifier(**params)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)],
    verbose=50
)

print(f"  ✅ Training complete")

# Evaluate
print("\n[5/6] Evaluating model...")

for name, X, y in [('Train', X_train, y_train), ('Val', X_val, y_val), ('Test', X_test, y_test)]:
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    logloss = log_loss(y, y_pred_proba)
    
    print(f"  {name:6s}: Acc={acc:.4f} ({acc:.2%}), AUC={auc:.4f}, LogLoss={logloss:.4f}")

# Save model
print("\n[6/6] Saving model...")

model_path = Path('../models/xgboost_43_features.pkl')
joblib.dump(model, model_path)

features_path = Path('../models/xgboost_43_features_list.json')
with open(features_path, 'w') as f:
    json.dump({
        'features': training_features,
        'n_features': len(training_features),
        'base_features': base_features
    }, f, indent=2)

print(f"  ✅ Model saved: {model_path}")
print(f"  ✅ Features saved: {features_path}")

# Feature importance
print("\n" + "="*120)
print("TOP 20 FEATURE IMPORTANCES")
print("="*120)

importance = model.feature_importances_
feature_importance = list(zip(training_features, importance))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for i, (feat, score) in enumerate(feature_importance[:20], 1):
    print(f"  {i:2d}. {feat:50s}: {score:.0f}")

print(f"\n{'='*120}")
print("RETRAINING COMPLETE - MODEL READY FOR 2025 PREDICTIONS")
print("="*120)

