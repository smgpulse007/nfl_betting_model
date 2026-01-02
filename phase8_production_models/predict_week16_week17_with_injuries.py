"""
Generate predictions for 2025 Week 16 and Week 17 using XGBoost with injury features
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

print("="*120)
print("GENERATE 2025 WEEK 16 & 17 PREDICTIONS WITH INJURY FEATURES")
print("="*120)

# Load model
print("\n[1/6] Loading XGBoost model with injury features...")
model = joblib.load('../models/xgboost_with_injuries.pkl')

with open('../models/xgboost_with_injuries_features.json', 'r') as f:
    feature_info = json.load(f)
    numeric_features = feature_info['features']

print(f"  ✅ Loaded model with {len(numeric_features)} features")

# Load 2025 data - need to merge injury/weather into phase6 format
print("\n[2/6] Loading 2025 data...")

# First, check if we have 2025 data in phase6 format
try:
    df_2025 = pd.read_parquet('../results/phase8_results/phase6_game_level_2025.parquet')
    print(f"  ✅ Loaded phase6_game_level_2025: {len(df_2025)} games")
except:
    print(f"  ⚠️  phase6_game_level_2025.parquet not found")
    print(f"  ℹ️  Need to create 2025 data in phase6 format with injury/weather features")
    print(f"  ℹ️  Using pregame_features as fallback (will need feature mapping)")
    
    # Load pregame_features for 2025
    df_pregame = pd.read_parquet('../results/phase8_results/pregame_features_1999_2025_complete.parquet')
    df_2025 = df_pregame[df_pregame['season'] == 2025].copy()
    print(f"  ✅ Loaded 2025 from pregame_features: {len(df_2025)} games")

# Filter to Week 16 and 17
df_week16_17 = df_2025[df_2025['week'].isin([16, 17])].copy()
print(f"  ✅ Week 16 & 17 games: {len(df_week16_17)}")
print(f"     • Week 16: {len(df_week16_17[df_week16_17['week'] == 16])} games")
print(f"     • Week 17: {len(df_week16_17[df_week16_17['week'] == 17])} games")

# Prepare features
print("\n[3/6] Preparing features...")

# Check which features are available
available_features = [f for f in numeric_features if f in df_week16_17.columns]
missing_features = [f for f in numeric_features if f not in df_week16_17.columns]

print(f"  ✅ Available features: {len(available_features)}/{len(numeric_features)}")
if missing_features:
    print(f"  ⚠️  Missing features: {len(missing_features)}")
    for feat in missing_features[:10]:
        print(f"     • {feat}")
    if len(missing_features) > 10:
        print(f"     ... and {len(missing_features) - 10} more")

# Fill missing features with 0
for feat in missing_features:
    df_week16_17[feat] = 0

# Prepare X
X = df_week16_17[numeric_features].fillna(0)

print(f"  ✅ X shape: {X.shape}")

# Generate predictions
print("\n[4/6] Generating predictions...")

y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of home win
y_pred = (y_pred_proba >= 0.5).astype(int)

df_week16_17['home_win_prob'] = y_pred_proba
df_week16_17['predicted_winner'] = df_week16_17.apply(
    lambda row: row['home_team'] if row['home_win_prob'] >= 0.5 else row['away_team'],
    axis=1
)
df_week16_17['confidence'] = df_week16_17['home_win_prob'].apply(
    lambda p: max(p, 1-p)
)

print(f"  ✅ Predictions generated")
print(f"  ✅ Average confidence: {df_week16_17['confidence'].mean():.1%}")

# Create output
print("\n[5/6] Creating output...")

output_cols = ['game_id', 'season', 'week', 'away_team', 'home_team', 
               'predicted_winner', 'home_win_prob', 'confidence']

# Add actual scores if available
if 'away_score' in df_week16_17.columns and 'home_score' in df_week16_17.columns:
    output_cols.extend(['away_score', 'home_score'])
    df_week16_17['actual_winner'] = df_week16_17.apply(
        lambda row: row['home_team'] if row['home_score'] > row['away_score'] 
                   else row['away_team'] if row['home_score'] < row['away_score']
                   else 'TIE',
        axis=1
    )
    output_cols.append('actual_winner')

df_output = df_week16_17[output_cols].copy()

# Sort by week and confidence
df_output = df_output.sort_values(['week', 'confidence'], ascending=[True, False])

# Save
print("\n[6/6] Saving predictions...")

output_file = Path('../results/phase8_results/2025_week16_17_predictions_with_injuries.csv')
df_output.to_csv(output_file, index=False)

print(f"  ✅ Saved: {output_file}")

# Display summary
print(f"\n{'='*120}")
print("SUMMARY - WEEK 16 & 17 PREDICTIONS")
print("="*120)

for week in [16, 17]:
    df_week = df_output[df_output['week'] == week]
    print(f"\nWEEK {week} ({len(df_week)} games):")
    print(f"  Average confidence: {df_week['confidence'].mean():.1%}")
    print(f"  High confidence (≥65%): {len(df_week[df_week['confidence'] >= 0.65])} games")
    print(f"\n  Top 5 predictions:")
    for idx, row in df_week.head(5).iterrows():
        print(f"    • {row['away_team']} @ {row['home_team']}: {row['predicted_winner']} ({row['confidence']:.1%})")

print(f"\n{'='*120}")

