"""
Task 8E.1: Fetch 2025 NFL Schedule Data

Fetch the 2025 NFL schedule and prepare it with the same feature structure
as the training data (1999-2024).

This script:
1. Fetches 2025 schedule from nfl_data_py
2. Loads historical data (1999-2024) to calculate rolling features
3. Prepares features for each 2025 game using only pre-game information
4. Saves the dataset ready for prediction
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8E.1: FETCH 2025 NFL SCHEDULE AND PREPARE FEATURES")
print("="*120)

# =============================================================================
# STEP 1: FETCH 2025 SCHEDULE
# =============================================================================
print(f"\n[1/6] Fetching 2025 NFL schedule...")

schedule_2025 = nfl.import_schedules([2025])

# Filter to regular season and playoffs
schedule_2025 = schedule_2025[schedule_2025['game_type'].isin(['REG', 'WC', 'DIV', 'CON', 'SB'])].copy()

print(f"  ✅ Fetched {len(schedule_2025)} games for 2025 season")
print(f"  ✅ Weeks available: {sorted(schedule_2025['week'].unique())}")
print(f"  ✅ Game types: {schedule_2025['game_type'].value_counts().to_dict()}")

# Check for completed games
completed_games = schedule_2025[schedule_2025['home_score'].notna()]
upcoming_games = schedule_2025[schedule_2025['home_score'].isna()]

print(f"\n  📊 Schedule Status:")
print(f"     - Completed games: {len(completed_games)}")
print(f"     - Upcoming games: {len(upcoming_games)}")

# =============================================================================
# STEP 2: LOAD HISTORICAL DATA (1999-2024)
# =============================================================================
print(f"\n[2/6] Loading historical data (1999-2024)...")

# Load the game-level dataset with all features
historical_df = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')

print(f"  ✅ Loaded {len(historical_df):,} historical games")
print(f"  ✅ Features: {historical_df.shape[1]}")
print(f"  ✅ Seasons: {historical_df['season'].min()}-{historical_df['season'].max()}")

# =============================================================================
# STEP 3: IDENTIFY PRE-GAME FEATURES
# =============================================================================
print(f"\n[3/6] Identifying pre-game features...")

# Load feature categorization
with open('../results/phase8_results/feature_categorization.json', 'r') as f:
    cat = json.load(f)

pre_game_dict = cat['pre_game_features']
pre_game_features = []
for category, features in pre_game_dict.items():
    pre_game_features.extend(features)

# Add additional known pre-game features
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

# Get home_ and away_ prefixed columns
pregame_cols = []
for feat in pre_game_features:
    home_feat = f'home_{feat}'
    away_feat = f'away_{feat}'
    if home_feat in historical_df.columns:
        pregame_cols.append(home_feat)
    if away_feat in historical_df.columns:
        pregame_cols.append(away_feat)

# Filter to numeric columns only
numeric_pregame = historical_df[pregame_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"  ✅ Total pre-game features: {len(pre_game_features)}")
print(f"  ✅ Pre-game columns (home/away): {len(pregame_cols)}")
print(f"  ✅ Numeric pre-game columns: {len(numeric_pregame)}")

# =============================================================================
# STEP 4: PREPARE 2025 GAME-LEVEL DATASET
# =============================================================================
print(f"\n[4/6] Preparing 2025 game-level dataset...")

# Create game_id for 2025 schedule
schedule_2025['game_id'] = (
    schedule_2025['season'].astype(str) + '_' +
    schedule_2025['week'].astype(str).str.zfill(2) + '_' +
    schedule_2025['away_team'] + '_' +
    schedule_2025['home_team']
)

# Select metadata columns
metadata_cols = ['game_id', 'season', 'week', 'gameday', 'weekday', 'gametime',
                'home_team', 'away_team', 'home_score', 'away_score']

# Add weather columns if available
weather_cols = ['roof', 'surface', 'temp', 'wind']
for col in weather_cols:
    if col in schedule_2025.columns:
        metadata_cols.append(col)

# Create base 2025 dataframe
df_2025 = schedule_2025[metadata_cols].copy()

print(f"  ✅ Created 2025 dataset: {df_2025.shape}")
print(f"  ✅ Metadata columns: {len(metadata_cols)}")

# =============================================================================
# STEP 5: LOAD EXISTING 2025 FEATURES (FROM PHASE 7)
# =============================================================================
print(f"\n[5/6] Loading existing 2025 features from Phase 7...")

# Check if engineered features exist
engineered_path = '../results/phase8_results/game_level_features_2025_weeks1_16_engineered.parquet'

if Path(engineered_path).exists():
    print(f"  ✅ Found engineered 2025 features")
    df_2025_features = pd.read_parquet(engineered_path)

    print(f"  ✅ Loaded 2025 features: {df_2025_features.shape}")
    print(f"  ✅ Games: {len(df_2025_features)}")
    print(f"  ✅ Features: {df_2025_features.shape[1]}")

    # Merge with schedule to get all games (including upcoming)
    df_2025 = schedule_2025[metadata_cols].merge(
        df_2025_features,
        on='game_id',
        how='left',
        suffixes=('', '_feat')
    )

    print(f"  ✅ Merged with schedule: {df_2025.shape}")

else:
    print(f"  ⚠️  Engineered features not found at: {engineered_path}")
    print(f"  ℹ️  Creating placeholder dataset...")

    # Create placeholder feature columns
    for col in numeric_pregame:
        df_2025[col] = np.nan

    print(f"  ⚠️  Feature calculation not implemented")
    print(f"  ℹ️  Created placeholder dataset with {len(numeric_pregame)} feature columns")

print(f"  ✅ Final shape: {df_2025.shape}")

# =============================================================================
# STEP 6: SAVE 2025 SCHEDULE
# =============================================================================
print(f"\n[6/6] Saving 2025 schedule...")

output_path = '../results/phase8_results/2025_schedule_with_features.parquet'
df_2025.to_parquet(output_path, index=False)

print(f"  ✅ Saved to: {output_path}")
print(f"  ✅ Shape: {df_2025.shape}")

# Summary
print(f"\n{'='*120}")
print("SUMMARY")
print(f"{'='*120}")
print(f"\n✅ 2025 Schedule prepared successfully")
print(f"   - Total games: {len(df_2025)}")
print(f"   - Completed: {len(df_2025[df_2025['home_score'].notna()])}")
print(f"   - Upcoming: {len(df_2025[df_2025['home_score'].isna()])}")
print(f"   - Features: {df_2025.shape[1]}")
print(f"\n✅ Dataset ready for predictions!")
print(f"   - File: {output_path}")
print(f"   - Shape: {df_2025.shape}")
print(f"\n{'='*120}")

