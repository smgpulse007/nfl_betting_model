"""Investigate why some features have low correlations"""
import pandas as pd
import numpy as np

# Load ESPN data
espn_stats = pd.read_parquet('data/espn_raw/team_stats_2024.parquet')
espn_records = pd.read_parquet('data/espn_raw/team_records_2024.parquet')
espn = pd.merge(espn_stats, espn_records, on='team', how='outer')
espn = espn.set_index('team').sort_index()

# Load nfl-data-py
pbp = pd.read_parquet('data/cache/pbp_2024.parquet')
schedules = pd.read_parquet('data/cache/schedules_2024.parquet')

# Filter to regular season
pbp_reg = pbp[pbp['week'] <= 18].copy()
schedules_reg = schedules[schedules['week'] <= 18].copy()

print("=" * 120)
print("INVESTIGATING DISCREPANCIES")
print("=" * 120)

# Sample teams
sample_teams = ['ARI', 'BAL', 'BUF', 'KC', 'DET']

print("\n1. PASSING YARDS INVESTIGATION")
print("-" * 120)

for team in sample_teams:
    # ESPN value
    espn_val = espn.loc[team, 'passing_passingYards']
    
    # My derivation
    team_pbp = pbp_reg[pbp_reg['posteam'] == team]
    pass_plays = team_pbp[team_pbp['play_type'] == 'pass']
    sack_plays = team_pbp[team_pbp['sack'] == 1]
    
    gross_pass_yards = pass_plays['yards_gained'].sum()
    sack_yards_lost = abs(sack_plays['yards_gained'].sum())
    my_val = gross_pass_yards - sack_yards_lost
    
    # Alternative: Maybe ESPN uses NET passing yards (includes sacks in the yards_gained)
    # Let's check what nfl-data-py has
    alt_val = pass_plays['yards_gained'].sum()  # Without subtracting sacks
    
    print(f"{team}: ESPN={espn_val:5.0f}, Derived={my_val:5.0f} (diff={espn_val-my_val:5.0f}), Alt={alt_val:5.0f}")
    print(f"      Gross pass yards: {gross_pass_yards:5.0f}, Sack yards lost: {sack_yards_lost:5.0f}")

print("\n2. PASSING ATTEMPTS INVESTIGATION")
print("-" * 120)

for team in sample_teams:
    # ESPN value
    espn_val = espn.loc[team, 'passing_passingAttempts']
    
    # My derivation
    team_pbp = pbp_reg[pbp_reg['posteam'] == team]
    pass_plays = team_pbp[team_pbp['play_type'] == 'pass']
    
    my_val = pass_plays['pass_attempt'].sum()
    
    # Alternative: Maybe ESPN counts something different
    # Check if there are plays with pass_attempt=1 but play_type != 'pass'
    alt_val = team_pbp['pass_attempt'].sum()
    
    print(f"{team}: ESPN={espn_val:5.0f}, Derived={my_val:5.0f} (diff={espn_val-my_val:5.0f}), Alt={alt_val:5.0f}")

print("\n3. POINTS INVESTIGATION")
print("-" * 120)

for team in sample_teams:
    # ESPN value
    espn_val = espn.loc[team, 'total_pointsFor']
    
    # My derivation from schedules
    team_schedule = schedules_reg[(schedules_reg['home_team'] == team) | (schedules_reg['away_team'] == team)]
    team_schedule['is_home'] = team_schedule['home_team'] == team
    team_schedule['team_score'] = team_schedule.apply(lambda x: x['home_score'] if x['is_home'] else x['away_score'], axis=1)
    my_val = team_schedule['team_score'].sum()
    
    print(f"{team}: ESPN={espn_val:5.0f}, Derived={my_val:5.0f} (diff={espn_val-my_val:5.0f})")
    print(f"      Games: {len(team_schedule)}, Avg PPG: {my_val/len(team_schedule):.1f}")

print("\n4. CHECK IF ESPN IS USING DIFFERENT WEEK RANGES")
print("-" * 120)

# Check what weeks ESPN might be using
print("\nSchedule weeks available:")
print(f"  Min week: {schedules['week'].min()}")
print(f"  Max week: {schedules['week'].max()}")
print(f"  Unique weeks: {sorted(schedules['week'].unique())}")

print("\nRegular season games per team:")
for team in sample_teams:
    team_schedule = schedules_reg[(schedules_reg['home_team'] == team) | (schedules_reg['away_team'] == team)]
    print(f"  {team}: {len(team_schedule)} games")

print("\n5. CHECK PLAY-BY-PLAY DATA SOURCE")
print("-" * 120)

# Check if there are any data quality issues
print("\nPlay-by-play data quality:")
print(f"  Total plays (all weeks): {len(pbp):,}")
print(f"  Total plays (weeks 1-18): {len(pbp_reg):,}")
print(f"  Plays with missing posteam: {pbp_reg['posteam'].isna().sum():,}")
print(f"  Plays with missing play_type: {pbp_reg['play_type'].isna().sum():,}")

print("\n6. SAMPLE GAME COMPARISON")
print("-" * 120)

# Pick a specific game and compare
sample_game = schedules_reg.iloc[0]
game_id = sample_game['game_id']
home_team = sample_game['home_team']
away_team = sample_game['away_team']

print(f"\nGame: {away_team} @ {home_team} (Week {sample_game['week']})")
print(f"  Final score: {away_team} {sample_game['away_score']} - {home_team} {sample_game['home_score']}")

# Get play-by-play for this game
game_pbp = pbp_reg[pbp_reg['game_id'] == game_id]
print(f"  Total plays in game: {len(game_pbp)}")

# Count passing plays for each team
home_pass = game_pbp[(game_pbp['posteam'] == home_team) & (game_pbp['play_type'] == 'pass')]
away_pass = game_pbp[(game_pbp['posteam'] == away_team) & (game_pbp['play_type'] == 'pass')]

print(f"  {home_team} passing plays: {len(home_pass)}, yards: {home_pass['yards_gained'].sum()}")
print(f"  {away_team} passing plays: {len(away_pass)}, yards: {away_pass['yards_gained'].sum()}")

print("\n" + "=" * 120)

