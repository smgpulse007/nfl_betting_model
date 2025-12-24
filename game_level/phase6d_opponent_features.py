"""
Phase 6D: Opponent-Adjusted Features
=====================================

Add opponent-specific features:
- Opponent's rolling averages (mirror of team's features)
- Matchup differentials (team - opponent)
- Division game indicators
- Rest advantage
- Home/away indicators

Input: game_level_features_engineered.parquet
Output: game_level_features_with_opponents.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
import nfl_data_py as nfl

warnings.filterwarnings('ignore')

print("="*120)
print("PHASE 6D: OPPONENT-ADJUSTED FEATURES")
print("="*120)

# Load engineered dataset
print(f"\n[1/5] Loading engineered dataset...")
df = pd.read_parquet('../results/game_level_features_engineered.parquet')
print(f"  ✅ Loaded {len(df):,} team-games")
print(f"  ✅ Current features: {df.shape[1]}")

# Load schedules to get opponent information
print(f"\n[2/5] Loading schedules to identify opponents...")
years = list(range(1999, 2025))
schedules = nfl.import_schedules(years)

# Create opponent mapping
opponent_map = {}
for _, row in schedules.iterrows():
    game_id = row['game_id']
    home_team = row['home_team']
    away_team = row['away_team']
    
    # Map team to opponent
    opponent_map[f"{game_id}_{home_team}"] = away_team
    opponent_map[f"{game_id}_{away_team}"] = home_team

# Add opponent column
df['opponent'] = df.apply(lambda row: opponent_map.get(f"{row['game_id']}_{row['team']}", None), axis=1)

print(f"  ✅ Identified opponents for {df['opponent'].notna().sum():,} games")
print(f"  ⚠️  Missing opponents: {df['opponent'].isna().sum()}")

# Add home/away indicator
df['is_home'] = df.apply(
    lambda row: row['team'] in row['game_id'].split('_')[-1] if pd.notna(row['game_id']) else False, 
    axis=1
)

print(f"  ✅ Added home/away indicator")
print(f"     Home games: {df['is_home'].sum():,}")
print(f"     Away games: {(~df['is_home']).sum():,}")

# =============================================================================
# OPPONENT FEATURES
# =============================================================================
print(f"\n[3/5] Creating opponent features...")

# Select key features for opponent matching
key_features = [
    'total_winPercent', 'total_pointsFor', 'total_pointsAgainst', 'total_differential',
    'total_pointsFor_roll3', 'total_pointsAgainst_roll3', 'total_differential_roll3',
    'total_pointsFor_roll5', 'total_pointsAgainst_roll5', 'total_differential_roll5',
    'win_streak', 'points_scored_trend', 'points_allowed_trend', 'point_diff_trend'
]

# Filter to features that exist
key_features = [f for f in key_features if f in df.columns]

print(f"  ℹ️  Matching {len(key_features)} key features")

# Create opponent features by merging
opponent_features = df[['team', 'game_id'] + key_features].copy()
opponent_features = opponent_features.rename(columns={'team': 'opponent'})
opponent_features = opponent_features.rename(columns={f: f'opp_{f}' for f in key_features})

# Merge opponent features
df = df.merge(opponent_features, on=['game_id', 'opponent'], how='left')

print(f"  ✅ Added {len(key_features)} opponent features")

# =============================================================================
# MATCHUP DIFFERENTIALS
# =============================================================================
print(f"\n[4/5] Creating matchup differentials...")

differential_features = []
for feature in key_features:
    if f'opp_{feature}' in df.columns:
        diff_name = f'diff_{feature}'
        df[diff_name] = df[feature] - df[f'opp_{feature}']
        differential_features.append(diff_name)

print(f"  ✅ Created {len(differential_features)} differential features")

# =============================================================================
# CONTEXTUAL FEATURES
# =============================================================================
print(f"\n[5/5] Adding contextual features...")

# Division game indicator (from schedules)
div_games = schedules[schedules['game_type'] == 'REG'][['game_id', 'div_game']].drop_duplicates()
df = df.merge(div_games, on='game_id', how='left')
df['div_game'] = df['div_game'].fillna(0).astype(int)

# Rest advantage (will be NaN for now - need schedule data)
df['rest_advantage'] = 0  # Placeholder

print(f"  ✅ Added division game indicator")
print(f"     Division games: {df['div_game'].sum():,}")

# =============================================================================
# SAVE FINAL DATASET
# =============================================================================
print(f"\n[6/6] Saving final dataset...")

output_file = Path('../results/game_level_features_with_opponents.parquet')
df.to_parquet(output_file, index=False)

print(f"  ✅ Saved: {output_file}")
print(f"  ✅ Shape: {df.shape}")
print(f"  ✅ Total features: {df.shape[1]}")

# Summary
opponent_feature_count = len([c for c in df.columns if c.startswith('opp_')])
diff_feature_count = len([c for c in df.columns if c.startswith('diff_')])

summary = {
    'timestamp': datetime.now().isoformat(),
    'total_rows': len(df),
    'total_features': df.shape[1],
    'opponent_features': opponent_feature_count,
    'differential_features': diff_feature_count,
    'contextual_features': 2,  # div_game, is_home
    'output_file': str(output_file)
}

with open('../results/phase6d_opponent_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*120}")
print("✅ PHASE 6D: COMPLETE!")
print(f"{'='*120}")
print(f"✅ Added {opponent_feature_count} opponent features")
print(f"✅ Added {diff_feature_count} differential features")
print(f"✅ Added 2 contextual features (div_game, is_home)")
print(f"✅ Total features: {df.shape[1]}")
print(f"✅ Ready for Phase 6A (TIER S+A integration)")

