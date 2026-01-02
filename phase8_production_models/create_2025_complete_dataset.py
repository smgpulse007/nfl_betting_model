"""
Create Complete 2025 Dataset (Weeks 1-17) with Injury + Weather Features

Strategy:
1. Load game_level_features_2025_weeks1_16_engineered.parquet (team-game level, has injury)
2. Convert to game-level format (1 row per game with home_/away_ prefixes)
3. Add weather features from schedule
4. Extend to Week 17 using team season averages
5. Save clean dataset for predictions

Output: 2025_complete_dataset_weeks1_17.parquet
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from pathlib import Path
import json

print("="*120)
print("CREATE COMPLETE 2025 DATASET (WEEKS 1-17)")
print("="*120)

# Load team-game level data (Weeks 1-16)
print("\n[1/7] Loading 2025 team-game level data...")
df_team = pd.read_parquet('../results/phase8_results/game_level_features_2025_weeks1_16_engineered.parquet')

print(f"  ✅ Loaded: {df_team.shape}")
print(f"  ✅ Weeks: {sorted(df_team['week'].unique())}")

# Check for injury features
injury_cols = [c for c in df_team.columns if 'injury' in c.lower() or 'qb_out' in c.lower()]
print(f"  ✅ Injury features: {len(injury_cols)}")
print(f"     {injury_cols}")

# Load feature categorization to know which features to use
print("\n[2/7] Loading feature categorization...")

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
    'pointsAgainst_roll3', 'pointsAgainst_roll5', 'pointsAgainst_std',
    'pointsFor_roll3', 'pointsFor_roll5', 'pointsFor_std',
    'ties_roll3', 'ties_roll5', 'ties_std',
    'winPercent_roll3', 'winPercent_roll5', 'winPercent_std',
    'winPercentage_roll3', 'winPercentage_roll5', 'winPercentage_std',
    'wins_roll3', 'wins_roll5', 'wins_std'
]
pre_game_features.extend(unknown_pregame)

# Add injury features
pre_game_features.extend(injury_cols)

print(f"  ✅ Pre-game features: {len(pre_game_features)}")

# Filter to features that exist in data
available_features = [f for f in pre_game_features if f in df_team.columns]
print(f"  ✅ Available in data: {len(available_features)}")

# Convert to game-level format
print("\n[3/7] Converting to game-level format...")

# Load schedule to get home/away info
schedule = nfl.import_schedules([2025])
schedule = schedule[schedule['game_type'] == 'REG'].copy()

# Merge to get home/away designation
df_team = df_team.merge(
    schedule[['game_id', 'home_team', 'away_team']], 
    on='game_id', 
    how='left',
    suffixes=('', '_sched')
)

# Determine if team is home or away
df_team['is_home'] = df_team['team'] == df_team['home_team']

# Split into home and away
df_home = df_team[df_team['is_home']].copy()
df_away = df_team[~df_team['is_home']].copy()

# Rename columns with home_/away_ prefix
home_cols = {f: f'home_{f}' for f in available_features if f in df_home.columns}
away_cols = {f: f'away_{f}' for f in available_features if f in df_away.columns}

df_home = df_home.rename(columns=home_cols)
df_away = df_away.rename(columns=away_cols)

# Merge home and away
df_game = df_home[['game_id'] + list(home_cols.values())].merge(
    df_away[['game_id'] + list(away_cols.values())],
    on='game_id',
    how='inner'
)

# Add metadata
df_game = df_game.merge(
    schedule[['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']],
    on='game_id',
    how='left'
)

print(f"  ✅ Game-level dataset: {df_game.shape}")
print(f"  ✅ Games: {len(df_game)}")

# Add weather features
print("\n[4/7] Adding weather features...")

weather_data = schedule[['game_id', 'roof', 'surface', 'temp', 'wind']].copy()
df_game = df_game.merge(weather_data, on='game_id', how='left')

# Create derived weather features
df_game['is_outdoor'] = (df_game['roof'] == 'outdoors').astype(int)
df_game['temp_extreme'] = ((df_game['temp'] < 32) | (df_game['temp'] > 85)).fillna(False).astype(int)
df_game['wind_high'] = (df_game['wind'] > 15).fillna(False).astype(int)

print(f"  ✅ Added weather features")
print(f"  ✅ Final shape: {df_game.shape}")

# Save Weeks 1-16 dataset
print("\n[5/7] Saving Weeks 1-16 dataset...")

output_file_w16 = Path('../results/phase8_results/2025_complete_dataset_weeks1_16.parquet')
df_game.to_parquet(output_file_w16, index=False)

print(f"  ✅ Saved: {output_file_w16}")

# Extend to Week 17 using team season averages
print("\n[6/7] Extending to Week 17...")

# Get Week 17 schedule
week17 = schedule[schedule['week'] == 17].copy()
week17_upcoming = week17[week17['home_score'].isna()].copy()

print(f"  ✅ Week 17 upcoming games: {len(week17_upcoming)}")

# Calculate team season averages from Weeks 1-16
team_averages = {}

# Get numeric features only
numeric_features = [f for f in available_features if f in df_team.select_dtypes(include=[np.number]).columns]

for team in week17_upcoming['home_team'].unique().tolist() + week17_upcoming['away_team'].unique().tolist():
    team_data = df_team[(df_team['team'] == team) & (df_team['week'] <= 16)]

    team_averages[team] = {}
    for feat in numeric_features:
        if feat in team_data.columns:
            team_averages[team][feat] = team_data[feat].mean()
        else:
            team_averages[team][feat] = 0

print(f"  ✅ Calculated averages for {len(team_averages)} teams")
print(f"  ✅ Numeric features: {len(numeric_features)}")

# Build Week 17 dataset
week17_data = []

for idx, game in week17_upcoming.iterrows():
    game_row = {
        'game_id': game['game_id'],
        'season': 2025,
        'week': 17,
        'home_team': game['home_team'],
        'away_team': game['away_team'],
        'home_score': np.nan,
        'away_score': np.nan
    }
    
    # Add home features
    for feat in numeric_features:
        game_row[f'home_{feat}'] = team_averages.get(game['home_team'], {}).get(feat, 0)

    # Add away features
    for feat in numeric_features:
        game_row[f'away_{feat}'] = team_averages.get(game['away_team'], {}).get(feat, 0)
    
    # Add weather
    game_row['roof'] = game['roof']
    game_row['surface'] = game['surface']
    game_row['temp'] = game['temp']
    game_row['wind'] = game['wind']
    game_row['is_outdoor'] = 1 if game['roof'] == 'outdoors' else 0
    game_row['temp_extreme'] = 1 if (pd.notna(game['temp']) and (game['temp'] < 32 or game['temp'] > 85)) else 0
    game_row['wind_high'] = 1 if (pd.notna(game['wind']) and game['wind'] > 15) else 0
    
    week17_data.append(game_row)

df_week17 = pd.DataFrame(week17_data)

print(f"  ✅ Week 17 dataset: {df_week17.shape}")

# Combine Weeks 1-17
df_complete = pd.concat([df_game, df_week17], ignore_index=True)

print(f"  ✅ Complete dataset: {df_complete.shape}")
print(f"  ✅ Weeks: {sorted(df_complete['week'].unique())}")

# Save complete dataset
print("\n[7/7] Saving complete dataset...")

output_file = Path('../results/phase8_results/2025_complete_dataset_weeks1_17.parquet')
df_complete.to_parquet(output_file, index=False)

print(f"  ✅ Saved: {output_file}")

# Summary
print(f"\n{'='*120}")
print("DATASET CREATION COMPLETE")
print("="*120)

print(f"""
FINAL DATASET: 2025_complete_dataset_weeks1_17.parquet

Structure:
  • Rows: {len(df_complete)} games
  • Columns: {len(df_complete.columns)}
  • Weeks: 1-17
  • Format: Game-level (1 row per game)

Features:
  • Pre-game features: {len(numeric_features)}
  • Injury features: {len(injury_cols)}
  • Weather features: 5 (temp, wind, temp_extreme, wind_high, is_outdoor)
  • Total features: {len([c for c in df_complete.columns if c.startswith('home_') or c.startswith('away_')])}

Coverage:
  • Weeks 1-16: Full Phase 6 engineered features
  • Week 17: Team season averages (Weeks 1-16)

Ready for:
  ✅ XGBoost predictions
  ✅ Week 16 & 17 analysis
  ✅ Model evaluation
""")

print(f"{'='*120}")

