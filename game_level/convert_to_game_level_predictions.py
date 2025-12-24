"""
Convert Team-Game Level to Game Level for Predictions
======================================================

Transform the team-game dataset (13,660 rows) to game-level dataset (~6,830 rows)
where each row represents a GAME with both home and away team features.

Input: game_level_features_complete_with_tier_sa.parquet (13,660 team-games × 584 features)
Output: game_level_predictions_dataset.parquet (~6,830 games × ~1,100 features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

print("="*120)
print("CONVERT TEAM-GAME LEVEL TO GAME LEVEL FOR PREDICTIONS")
print("="*120)

# Load team-game dataset
print(f"\n[1/5] Loading team-game dataset...")
df = pd.read_parquet('../results/game_level_features_complete_with_tier_sa.parquet')
print(f"  ✅ Loaded {len(df):,} team-games with {df.shape[1]} features")

# Identify feature columns (exclude metadata)
metadata_cols = ['team', 'game_id', 'opponent', 'game_date', 'year', 'week', 'is_home']
feature_cols = [c for c in df.columns if c not in metadata_cols]
print(f"  ✅ Identified {len(feature_cols)} feature columns")

# Separate home and away teams
print(f"\n[2/5] Separating home and away teams...")
home_df = df[df['is_home'] == True].copy()
away_df = df[df['is_home'] == False].copy()

print(f"  ✅ Home team-games: {len(home_df):,}")
print(f"  ✅ Away team-games: {len(away_df):,}")

# Rename columns with home/away prefix
print(f"\n[3/5] Renaming columns with home/away prefix...")

# Home team features
home_features = {}
for col in feature_cols:
    home_features[col] = f'home_{col}'

home_df = home_df.rename(columns=home_features)
home_df = home_df.rename(columns={'team': 'home_team'})

# Away team features
away_features = {}
for col in feature_cols:
    away_features[col] = f'away_{col}'

away_df = away_df.rename(columns=away_features)
away_df = away_df.rename(columns={'team': 'away_team'})

print(f"  ✅ Renamed {len(feature_cols)} features for home team")
print(f"  ✅ Renamed {len(feature_cols)} features for away team")

# Merge home and away on game_id
print(f"\n[4/5] Merging home and away teams...")

# Keep only necessary columns for merge
home_cols = ['game_id', 'home_team', 'game_date', 'year', 'week'] + list(home_features.values())
away_cols = ['game_id', 'away_team'] + list(away_features.values())

home_df = home_df[home_cols]
away_df = away_df[away_cols]

# Merge
games_df = home_df.merge(away_df, on='game_id', how='inner')

print(f"  ✅ Merged to {len(games_df):,} games")
print(f"  ✅ Total features: {games_df.shape[1]}")

# Add target variable (home_win)
print(f"\n[5/5] Adding target variable...")
games_df['home_win'] = (games_df['home_win'] == 1).astype(int)

# Verify data integrity
print(f"\n  Data Integrity Checks:")
print(f"    ✅ Unique games: {games_df['game_id'].nunique():,}")
print(f"    ✅ Date range: {games_df['game_date'].min()} to {games_df['game_date'].max()}")
print(f"    ✅ Years: {games_df['year'].min()} to {games_df['year'].max()}")
print(f"    ✅ Home wins: {games_df['home_win'].sum():,} ({games_df['home_win'].mean()*100:.1f}%)")

# Sample game
print(f"\n  Sample Game (BAL @ KC, Week 1 2024):")
sample = games_df[games_df['game_id'] == '2024_01_BAL_KC']
if len(sample) > 0:
    print(f"    Away Team: {sample['away_team'].values[0]}")
    print(f"    Home Team: {sample['home_team'].values[0]}")
    print(f"    Home Win: {sample['home_win'].values[0]}")
    print(f"    Away Points: {sample['away_total_pointsFor'].values[0]:.0f}")
    print(f"    Home Points: {sample['home_total_pointsFor'].values[0]:.0f}")

# Save
output_file = Path('../results/game_level_predictions_dataset.parquet')
games_df.to_parquet(output_file, index=False)

print(f"\n{'='*120}")
print("✅ CONVERSION COMPLETE!")
print(f"{'='*120}")
print(f"✅ Input: 13,660 team-games × 584 features")
print(f"✅ Output: {len(games_df):,} games × {games_df.shape[1]} features")
print(f"✅ Saved: {output_file}")
print(f"\n✅ Ready for game outcome predictions!")
print(f"✅ Each row = 1 game with home/away features")
print(f"✅ Target variable: home_win (1 = home team wins, 0 = away team wins)")

# Create summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'input_rows': len(df),
    'input_features': df.shape[1],
    'output_rows': len(games_df),
    'output_features': games_df.shape[1],
    'home_features': len(home_features),
    'away_features': len(away_features),
    'date_range': {
        'min': str(games_df['game_date'].min()),
        'max': str(games_df['game_date'].max())
    },
    'year_range': {
        'min': int(games_df['year'].min()),
        'max': int(games_df['year'].max())
    },
    'home_win_rate': float(games_df['home_win'].mean()),
    'output_file': str(output_file)
}

with open('../results/game_level_conversion_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Summary saved: game_level_conversion_summary.json")

# Show example prediction format
print(f"\n{'='*120}")
print("EXAMPLE: How to predict Week 16 2025 games")
print(f"{'='*120}")
print(f"""
# Load the dataset
games_df = pd.read_parquet('results/game_level_predictions_dataset.parquet')

# Filter to Week 16 2025
week16_2025 = games_df[(games_df['year'] == 2025) & (games_df['week'] == 16)]

# Select features (exclude metadata and target)
feature_cols = [c for c in games_df.columns if c not in ['game_id', 'game_date', 'year', 'week', 
                                                           'home_team', 'away_team', 'home_win']]

# Prepare features
X = week16_2025[feature_cols]

# Train model on historical data (1999-2024)
train_df = games_df[games_df['year'] < 2025]
X_train = train_df[feature_cols]
y_train = train_df['home_win']

# Train XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)

# Predict Week 16 2025
predictions = model.predict_proba(X)[:, 1]  # Probability of home win

# Create results
results = week16_2025[['game_id', 'away_team', 'home_team']].copy()
results['home_win_prob'] = predictions
results['predicted_winner'] = results.apply(
    lambda x: x['home_team'] if x['home_win_prob'] > 0.5 else x['away_team'], axis=1
)
""")

