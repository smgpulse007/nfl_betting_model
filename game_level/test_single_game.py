"""
Phase 5A: Single Game Derivation Test

Test the derive_game_features() function on BAL @ KC, Week 1, 2024.
This is the validation checkpoint before proceeding to Phase 5B.
"""

import pandas as pd
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from game_level.derive_game_features import derive_game_features
from team_abbreviation_mapping import nfl_data_py_to_espn

print("="*120)
print("PHASE 5A: SINGLE GAME DERIVATION TEST")
print("="*120)

# Configuration
GAME_ID = '2024_01_BAL_KC'
WEEK = 1
SEASON = 2024

print(f"\n[1/5] Loading data...")
print(f"  Target game: {GAME_ID}")
print(f"  Week: {WEEK}, Season: {SEASON}")

# Load play-by-play and schedules
pbp = pd.read_parquet('../data/cache/pbp_2024.parquet')
schedules = pd.read_parquet('../data/cache/schedules_2024.parquet')

print(f"  ✅ Loaded {len(pbp):,} plays")
print(f"  ✅ Loaded {len(schedules)} games")

# Get game info
game_info = schedules[schedules['game_id'] == GAME_ID].iloc[0]
home_team = game_info['home_team']
away_team = game_info['away_team']
home_score = game_info['home_score']
away_score = game_info['away_score']

print(f"\n[2/5] Game Information:")
print(f"  Away: {away_team} ({nfl_data_py_to_espn(away_team)})")
print(f"  Home: {home_team} ({nfl_data_py_to_espn(home_team)})")
print(f"  Score: {away_team} {away_score} @ {home_team} {home_score}")
print(f"  Winner: {home_team if home_score > away_score else away_team}")

# Derive features for both teams
print(f"\n[3/5] Deriving features for {away_team}...")
away_features = derive_game_features(
    team=nfl_data_py_to_espn(away_team),
    game_id=GAME_ID,
    pbp=pbp,
    schedules=schedules
)
print(f"  ✅ Derived {len(away_features)} features for {away_team}")

print(f"\n[4/5] Deriving features for {home_team}...")
home_features = derive_game_features(
    team=nfl_data_py_to_espn(home_team),
    game_id=GAME_ID,
    pbp=pbp,
    schedules=schedules
)
print(f"  ✅ Derived {len(home_features)} features for {home_team}")

# Display key features
print(f"\n[5/5] Key Features Summary:")
print(f"\n{away_team} (Away):")
print(f"  Passing Yards: {away_features['passing_passingYards']}")
print(f"  Rushing Yards: {away_features['rushing_rushingYards']}")
print(f"  Total Yards: {away_features['total_totalYards']}")
print(f"  Touchdowns: {away_features['total_totalTouchdowns']}")
print(f"  Turnovers: {away_features['total_turnovers']}")
print(f"  Third Down %: {away_features['total_thirdDownPct']:.1f}%")
print(f"  Points: {away_features['team_score']}")
print(f"  Result: {'WIN' if away_features['win'] else 'LOSS'}")

print(f"\n{home_team} (Home):")
print(f"  Passing Yards: {home_features['passing_passingYards']}")
print(f"  Rushing Yards: {home_features['rushing_rushingYards']}")
print(f"  Total Yards: {home_features['total_totalYards']}")
print(f"  Touchdowns: {home_features['total_totalTouchdowns']}")
print(f"  Turnovers: {home_features['total_turnovers']}")
print(f"  Third Down %: {home_features['total_thirdDownPct']:.1f}%")
print(f"  Points: {home_features['team_score']}")
print(f"  Result: {'WIN' if home_features['win'] else 'LOSS'}")

# Save results
print(f"\n[6/6] Saving results...")
output_dir = Path('../results')
output_dir.mkdir(exist_ok=True)

# Save as JSON for inspection
results = {
    'game_id': GAME_ID,
    'away_team': away_team,
    'home_team': home_team,
    'away_features': away_features,
    'home_features': home_features
}

output_file = output_dir / 'phase5a_single_game_test.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"  ✅ Saved to: {output_file}")

# Create DataFrame for easier inspection
df_away = pd.DataFrame([away_features])
df_home = pd.DataFrame([home_features])
df_combined = pd.concat([df_away, df_home], ignore_index=True)

csv_file = output_dir / 'phase5a_single_game_test.csv'
df_combined.to_csv(csv_file, index=False)
print(f"  ✅ Saved to: {csv_file}")

print(f"\n{'='*120}")
print("✅ PHASE 5A TEST COMPLETE!")
print(f"{'='*120}")
print(f"\nNext Steps:")
print(f"1. Manually validate features against ESPN game logs")
print(f"2. Check for missing values or anomalies")
print(f"3. Compare against season-level aggregations")
print(f"4. If validation passes, proceed to Phase 5B (full 2024 season)")

