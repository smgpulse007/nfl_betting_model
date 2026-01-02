"""
Fetch Week 17 schedule for 2025
"""

import nfl_data_py as nfl
import pandas as pd

print("="*120)
print("FETCH 2025 WEEK 17 SCHEDULE")
print("="*120)

# Fetch 2025 schedule
print("\n[1/2] Fetching 2025 schedule...")
schedules = nfl.import_schedules([2025])

print(f"  ✅ Loaded {len(schedules)} games")

# Filter to Week 17
week17 = schedules[schedules['week'] == 17].copy()

print(f"\n[2/2] Week 17 games:")
print(f"  Total: {len(week17)} games")

if len(week17) > 0:
    print(f"\n  Games:")
    for idx, row in week17.iterrows():
        status = "✅ Complete" if pd.notna(row['home_score']) else "⏳ Upcoming"
        if pd.notna(row['home_score']):
            print(f"    • {row['away_team']} @ {row['home_team']}: {row['away_score']:.0f}-{row['home_score']:.0f} {status}")
        else:
            print(f"    • {row['away_team']} @ {row['home_team']}: {status}")
    
    # Check how many are completed
    completed = week17[pd.notna(week17['home_score'])]
    print(f"\n  Completed: {len(completed)}/{len(week17)} games")
    print(f"  Upcoming: {len(week17) - len(completed)}/{len(week17)} games")
else:
    print(f"  ⚠️  No Week 17 games found in schedule")

print(f"\n{'='*120}")

