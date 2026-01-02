"""
Generate Week 16 & 17 Predictions with Feature-Aligned XGBoost Model
"""

import pandas as pd
import numpy as np
import joblib
import json

print("="*120)
print("WEEK 16 & 17 PREDICTIONS - FEATURE ALIGNED MODEL")
print("="*120)

# Load model
print("\n[1/4] Loading feature-aligned XGBoost model...")
model = joblib.load('../models/xgboost_43_features.pkl')

with open('../models/xgboost_43_features_list.json', 'r') as f:
    feature_info = json.load(f)
    model_features = feature_info['features']

print(f"  ✅ Loaded model with {len(model_features)} features")

# Load 2025 complete dataset
print("\n[2/4] Loading 2025 complete dataset...")
df_2025 = pd.read_parquet('../results/phase8_results/2025_complete_dataset_weeks1_17.parquet')

print(f"  ✅ Loaded: {df_2025.shape}")

# Filter to Week 16 & 17
df_week16_17 = df_2025[df_2025['week'].isin([16, 17])].copy()

print(f"  ✅ Week 16 & 17 games: {len(df_week16_17)}")

# Prepare features
print("\n[3/4] Preparing features...")

X = df_week16_17[model_features].fillna(0)

print(f"  ✅ X shape: {X.shape}")
print(f"  ✅ Perfect feature alignment!")

# Generate predictions
print("\n[4/4] Generating predictions...")

y_pred_proba = model.predict_proba(X)[:, 1]

df_week16_17['home_win_prob'] = y_pred_proba
df_week16_17['predicted_winner'] = df_week16_17.apply(
    lambda row: row['home_team'] if row['home_win_prob'] >= 0.5 else row['away_team'],
    axis=1
)
df_week16_17['confidence'] = df_week16_17['home_win_prob'].apply(
    lambda p: max(p, 1-p)
)

# Create output
output_cols = ['game_id', 'season', 'week', 'away_team', 'home_team', 
               'predicted_winner', 'home_win_prob', 'confidence',
               'away_score', 'home_score']

df_week16_17['actual_winner'] = df_week16_17.apply(
    lambda row: row['home_team'] if pd.notna(row['home_score']) and row['home_score'] > row['away_score'] 
               else row['away_team'] if pd.notna(row['home_score']) and row['home_score'] < row['away_score']
               else 'TIE' if pd.notna(row['home_score'])
               else 'UPCOMING',
    axis=1
)
output_cols.append('actual_winner')

df_output = df_week16_17[output_cols].copy()
df_output = df_output.sort_values(['week', 'confidence'], ascending=[True, False])

# Save
output_file = '../results/phase8_results/2025_week16_17_aligned_predictions.csv'
df_output.to_csv(output_file, index=False)

print(f"  ✅ Saved: {output_file}")

# Display summary
print(f"\n{'='*120}")
print("SUMMARY - WEEK 16 & 17 PREDICTIONS (FEATURE ALIGNED)")
print("="*120)

for week in [16, 17]:
    df_week = df_output[df_output['week'] == week]
    
    if len(df_week) == 0:
        continue
    
    print(f"\nWEEK {week} ({len(df_week)} games):")
    print(f"  Average confidence: {df_week['confidence'].mean():.1%}")
    print(f"  High confidence (≥65%): {len(df_week[df_week['confidence'] >= 0.65])} games")
    
    # Check if we have actual results
    completed = df_week[df_week['actual_winner'] != 'UPCOMING']
    if len(completed) > 0:
        correct = completed[completed['predicted_winner'] == completed['actual_winner']]
        print(f"  Completed games: {len(completed)}")
        print(f"  Correct predictions: {len(correct)}/{len(completed)} ({len(correct)/len(completed):.1%})")
    
    print(f"\n  Top 5 predictions:")
    for idx, row in df_week.head(5).iterrows():
        status = ""
        if row['actual_winner'] != 'UPCOMING':
            status = " ✅" if row['predicted_winner'] == row['actual_winner'] else " ❌"
        print(f"    • {row['away_team']} @ {row['home_team']}: {row['predicted_winner']} ({row['confidence']:.1%}){status}")

print(f"\n{'='*120}")
print("MODEL PERFORMANCE SUMMARY")
print("="*120)
print(f"""
Training Performance:
  • Train (1999-2019): 72.71% accuracy
  • Val (2020-2023):   61.26% accuracy
  • Test (2024):       68.42% accuracy

2025 Week 16 Performance:
  • Completed games: {len(df_output[(df_output['week'] == 16) & (df_output['actual_winner'] != 'UPCOMING')])}
  • Correct: {len(df_output[(df_output['week'] == 16) & (df_output['predicted_winner'] == df_output['actual_winner'])])}
  • Accuracy: {len(df_output[(df_output['week'] == 16) & (df_output['predicted_winner'] == df_output['actual_winner'])]) / len(df_output[(df_output['week'] == 16) & (df_output['actual_winner'] != 'UPCOMING')]) * 100:.1f}%

Features:
  • Total features: {len(model_features)}
  • Injury features: 5 (injury_impact, qb_out, opp_injury_impact, opp_qb_out, diff_injury_impact)
  • Weather features: 5 (temp, wind, temp_extreme, wind_high, is_outdoor)
  • Advanced features: CPOE, pressure rate, RYOE, separation, time to throw (3-week rolling)
""")

print(f"{'='*120}")

