"""
Fetch actual 2025 NFL season data for weeks 1-17
"""

import nfl_data_py as nfl
import pandas as pd
from datetime import datetime

print("=" * 100)
print("FETCHING 2025 NFL SEASON DATA")
print("=" * 100)

# Fetch 2025 schedule with scores
print("\n[1/3] Fetching 2025 schedule...")
schedule_2025 = nfl.import_schedules([2025])

print(f"  ✅ Loaded {len(schedule_2025)} total games")
print(f"  Weeks available: {sorted(schedule_2025['week'].unique())}")

# Check which games have scores
schedule_2025['has_score'] = schedule_2025['home_score'].notna()
completed = schedule_2025[schedule_2025['has_score']]
upcoming = schedule_2025[~schedule_2025['has_score']]

print(f"\n  Completed games: {len(completed)}")
print(f"  Upcoming games: {len(upcoming)}")

# Show breakdown by week
print(f"\n  Games by week:")
for week in sorted(schedule_2025['week'].unique()):
    week_games = schedule_2025[schedule_2025['week'] == week]
    week_completed = week_games['has_score'].sum()
    print(f"    Week {week:2d}: {week_completed:3d}/{len(week_games):3d} completed")

# Save to file
print("\n[2/3] Saving data...")
output_path = '../results/phase8_results/2025_schedule_actual.csv'
schedule_2025.to_csv(output_path, index=False)
print(f"  ✅ Saved to: {output_path}")

# Show Week 17 details
print("\n[3/3] Week 17 Details:")
week17 = schedule_2025[schedule_2025['week'] == 17].copy()
week17_completed = week17[week17['has_score']]
week17_upcoming = week17[~week17['has_score']]

print(f"  Total Week 17 games: {len(week17)}")
print(f"  Completed: {len(week17_completed)}")
print(f"  Upcoming: {len(week17_upcoming)}")

if len(week17_completed) > 0:
    print(f"\n  Completed Week 17 games:")
    for idx, row in week17_completed.iterrows():
        print(f"    {row['away_team']} @ {row['home_team']}: {row['away_score']:.0f}-{row['home_score']:.0f}")

if len(week17_upcoming) > 0:
    print(f"\n  Upcoming Week 17 games:")
    for idx, row in week17_upcoming.iterrows():
        gameday = row['gameday'][:10] if pd.notna(row['gameday']) else 'TBD'
        print(f"    {row['away_team']} @ {row['home_team']} ({gameday})")

print("\n" + "=" * 100)
print("DATA FETCH COMPLETE")
print("=" * 100)

