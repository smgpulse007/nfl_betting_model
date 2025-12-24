"""
Phase 6: Comprehensive Feature Engineering
===========================================

Integrates all feature engineering phases:
- Phase 6A: TIER S+A features (NGS, PFR data)
- Phase 6B: Rolling averages (3-game, 5-game, season-to-date)
- Phase 6C: Streak features (win/loss, scoring, momentum)
- Phase 6D: Opponent-adjusted metrics (strength, matchups)

Input: game_level_features_1999_2024_complete.parquet (13,564 rows, 191 features)
Output: game_level_features_engineered.parquet (~400-500 features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("="*120)
print("PHASE 6: COMPREHENSIVE FEATURE ENGINEERING")
print("="*120)

# Load game-level dataset
print(f"\n[1/7] Loading game-level dataset...")
df = pd.read_parquet('../results/game_level_features_1999_2024_complete.parquet')
print(f"  ✅ Loaded {len(df):,} team-games")
print(f"  ✅ Base features: {df.shape[1]}")

# Load approved features
with open('../results/approved_features_r085.json') as f:
    approved_data = json.load(f)
    approved_features = approved_data['features']

print(f"  ✅ Approved features: {len(approved_features)}")

# Extract metadata from game_id
df['year'] = df['game_id'].str[:4].astype(int)
df['week'] = df['game_id'].str.split('_').str[1].astype(int)

# Add game date for sorting (approximate - week 1 starts ~Sep 10)
df['game_date'] = pd.to_datetime(df['year'].astype(str) + '-09-01') + pd.to_timedelta((df['week'] - 1) * 7, unit='D')

# Sort by team and date (critical for rolling calculations)
df = df.sort_values(['team', 'game_date']).reset_index(drop=True)

print(f"  ✅ Extracted year, week, game_date")
print(f"  ✅ Sorted by team and date")

# =============================================================================
# PHASE 6A: TIER S+A FEATURES (Placeholder - will be NaN for now)
# =============================================================================
print(f"\n[2/7] Phase 6A: TIER S+A Features...")
print(f"  ⚠️  TIER S+A features require NGS/PFR data (2016+)")
print(f"  ⚠️  Skipping for now - will add in separate script")
print(f"  ℹ️  Expected features: cpoe_3wk, pressure_rate_3wk, injury_impact, qb_out, ryoe_3wk, separation_3wk, time_to_throw_3wk")

# =============================================================================
# PHASE 6B: ROLLING AVERAGES
# =============================================================================
print(f"\n[3/7] Phase 6B: Rolling Averages...")

# Load predictive power results to select top features
pred_df = pd.read_csv('../results/game_level_eda_predictive_power.csv')
top_features = pred_df.head(50)['feature'].tolist()

print(f"  ✅ Selected top 50 features by predictive power")
print(f"  ℹ️  Top 5: {top_features[:5]}")

# Compute rolling averages for top features
rolling_features = []
for feature in top_features:
    if feature in df.columns and feature not in ['team', 'game_id', 'year', 'week', 'game_date']:
        # 3-game rolling average
        df[f'{feature}_roll3'] = df.groupby('team')[feature].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        
        # 5-game rolling average
        df[f'{feature}_roll5'] = df.groupby('team')[feature].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        
        # Season-to-date average
        df[f'{feature}_std'] = df.groupby(['team', 'year'])[feature].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        
        rolling_features.extend([f'{feature}_roll3', f'{feature}_roll5', f'{feature}_std'])

print(f"  ✅ Created {len(rolling_features)} rolling average features")
print(f"     - 3-game rolling: {len([f for f in rolling_features if '_roll3' in f])}")
print(f"     - 5-game rolling: {len([f for f in rolling_features if '_roll5' in f])}")
print(f"     - Season-to-date: {len([f for f in rolling_features if '_std' in f])}")

# =============================================================================
# PHASE 6C: STREAK FEATURES
# =============================================================================
print(f"\n[4/7] Phase 6C: Streak Features...")

# Win indicator
df['win'] = (df['total_pointsFor'] > df['total_pointsAgainst']).astype(int)

# Win/loss streak
def calculate_streak(wins):
    """Calculate current win/loss streak (positive for wins, negative for losses)."""
    streak = []
    current_streak = 0
    
    for win in wins:
        if pd.isna(win):
            streak.append(0)
        elif win == 1:
            current_streak = current_streak + 1 if current_streak >= 0 else 1
            streak.append(current_streak)
        else:
            current_streak = current_streak - 1 if current_streak <= 0 else -1
            streak.append(current_streak)
    
    return streak

df['win_streak'] = df.groupby('team')['win'].transform(
    lambda x: pd.Series(calculate_streak(x.shift(1)), index=x.index)
)

# Scoring streaks
df['scored_20plus'] = (df['total_pointsFor'] >= 20).astype(int)
df['scored_30plus'] = (df['total_pointsFor'] >= 30).astype(int)

df['streak_20plus'] = df.groupby('team')['scored_20plus'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).sum()
)

df['streak_30plus'] = df.groupby('team')['scored_30plus'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).sum()
)

# Points scored/allowed trends (last 3 games)
df['points_scored_trend'] = df.groupby('team')['total_pointsFor'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)

df['points_allowed_trend'] = df.groupby('team')['total_pointsAgainst'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)

# Point differential trend
df['point_diff_trend'] = df['points_scored_trend'] - df['points_allowed_trend']

streak_features = ['win_streak', 'streak_20plus', 'streak_30plus', 
                   'points_scored_trend', 'points_allowed_trend', 'point_diff_trend']

print(f"  ✅ Created {len(streak_features)} streak features")
print(f"     Features: {streak_features}")

print(f"\n[5/7] Phase 6D: Opponent-Adjusted Metrics...")
print(f"  ℹ️  Opponent features require game-level opponent matching")
print(f"  ℹ️  Will be added in post-processing step")

# =============================================================================
# SAVE ENGINEERED DATASET
# =============================================================================
print(f"\n[6/7] Saving engineered dataset...")

output_file = Path('../results/game_level_features_engineered.parquet')
df.to_parquet(output_file, index=False)

print(f"  ✅ Saved: {output_file}")
print(f"  ✅ Shape: {df.shape}")
print(f"  ✅ Total features: {df.shape[1]}")

# Summary
print(f"\n[7/7] Feature Engineering Summary...")
print(f"  Base features: {len(approved_features)}")
print(f"  Rolling features: {len(rolling_features)}")
print(f"  Streak features: {len(streak_features)}")
print(f"  Total features: {df.shape[1]}")

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'total_rows': len(df),
    'total_features': df.shape[1],
    'base_features': len(approved_features),
    'rolling_features': len(rolling_features),
    'streak_features': len(streak_features),
    'top_features_used': top_features[:10],
    'output_file': str(output_file)
}

with open('../results/phase6_engineering_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*120}")
print("✅ PHASE 6B & 6C: COMPLETE!")
print(f"{'='*120}")
print(f"✅ Created {len(rolling_features)} rolling average features")
print(f"✅ Created {len(streak_features)} streak features")
print(f"✅ Total features: {df.shape[1]}")
print(f"✅ Ready for Phase 6D (opponent matching) and EDA")

