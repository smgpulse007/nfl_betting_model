"""Compare ESPN vs derived values side-by-side"""
import pandas as pd

# Load ESPN data
espn_stats = pd.read_parquet('data/espn_raw/team_stats_2024.parquet')
espn_records = pd.read_parquet('data/espn_raw/team_records_2024.parquet')
espn = pd.merge(espn_stats, espn_records, on='team', how='outer')
espn = espn.set_index('team').sort_index()

# Load nfl-data-py
pbp = pd.read_parquet('data/cache/pbp_2024.parquet')
schedules = pd.read_parquet('data/cache/pbp_2024.parquet')

# Derive for a few teams
teams = ['ARI', 'BAL', 'BUF', 'CIN', 'DET']

print("=" * 100)
print("ESPN vs Derived - Side-by-Side Comparison")
print("=" * 100)

for team in teams:
    print(f"\n{team}:")
    print("-" * 100)
    
    # ESPN values
    espn_pass_yards = espn.loc[team, 'passing_passingYards']
    espn_pass_att = espn.loc[team, 'passing_passingAttempts']
    espn_completions = espn.loc[team, 'passing_completions']
    espn_pass_td = espn.loc[team, 'passing_passingTouchdowns']
    espn_int = espn.loc[team, 'passing_interceptions']
    espn_wins = espn.loc[team, 'total_wins']
    
    # Derived values
    team_pbp = pbp[pbp['posteam'] == team]
    pass_plays = team_pbp[team_pbp['play_type'] == 'pass']
    sack_yards = team_pbp[team_pbp['sack'] == 1]['yards_gained'].sum()
    
    derived_pass_yards = pass_plays['yards_gained'].sum() + sack_yards
    derived_pass_att = pass_plays['pass_attempt'].sum()
    derived_completions = pass_plays['complete_pass'].sum()
    derived_pass_td = pass_plays['pass_touchdown'].sum()
    derived_int = pass_plays['interception'].sum()
    
    print(f"{'Feature':<30} {'ESPN':>15} {'Derived':>15} {'Diff':>15} {'Match':>10}")
    print("-" * 100)
    print(f"{'Passing Yards':<30} {espn_pass_yards:>15.0f} {derived_pass_yards:>15.0f} {espn_pass_yards-derived_pass_yards:>15.0f} {'✅' if abs(espn_pass_yards-derived_pass_yards) < 1 else '❌':>10}")
    print(f"{'Passing Attempts':<30} {espn_pass_att:>15.0f} {derived_pass_att:>15.0f} {espn_pass_att-derived_pass_att:>15.0f} {'✅' if abs(espn_pass_att-derived_pass_att) < 1 else '❌':>10}")
    print(f"{'Completions':<30} {espn_completions:>15.0f} {derived_completions:>15.0f} {espn_completions-derived_completions:>15.0f} {'✅' if abs(espn_completions-derived_completions) < 1 else '❌':>10}")
    print(f"{'Passing TDs':<30} {espn_pass_td:>15.0f} {derived_pass_td:>15.0f} {espn_pass_td-derived_pass_td:>15.0f} {'✅' if abs(espn_pass_td-derived_pass_td) < 1 else '❌':>10}")
    print(f"{'Interceptions':<30} {espn_int:>15.0f} {derived_int:>15.0f} {espn_int-derived_int:>15.0f} {'✅' if abs(espn_int-derived_int) < 1 else '❌':>10}")
    print(f"{'Wins':<30} {espn_wins:>15.0f} {'N/A':>15} {'N/A':>15} {'':>10}")

print("\n" + "=" * 100)

