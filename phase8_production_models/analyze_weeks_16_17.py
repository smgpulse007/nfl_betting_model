"""
Analyze Week 16 and Week 17 predictions and results for 2025 season
"""

import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv('../results/phase8_results/2025_predictions.csv')

print("=" * 100)
print("2025 SEASON - WEEK 16 & 17 ANALYSIS")
print("=" * 100)

print(f"\nTotal games in dataset: {len(df)}")
print(f"\nGames by week:")
print(df.groupby('week').size().sort_index())

# Week 16 Analysis
print("\n" + "=" * 100)
print("WEEK 16 ANALYSIS")
print("=" * 100)

w16 = df[df['week'] == 16].copy()
print(f"\nTotal Week 16 games: {len(w16)}")
print(f"Games with scores: {w16['home_score'].notna().sum()}")
print(f"Games without scores: {w16['home_score'].isna().sum()}")

if w16['home_score'].notna().sum() > 0:
    w16_completed = w16[w16['home_score'].notna()].copy()
    print(f"\n--- Week 16 Completed Games ---")
    print(f"Total: {len(w16_completed)}")
    
    # Show all games
    print("\nAll Week 16 games:")
    for idx, row in w16_completed.iterrows():
        print(f"  {row['away_team']} @ {row['home_team']}: {row['away_score']:.0f}-{row['home_score']:.0f} | "
              f"Predicted: {row['predicted_winner']} ({row['confidence']:.1%}) | "
              f"Actual: {row['actual_winner']}")
else:
    print("\nNo completed games in Week 16 yet.")

# Week 17 Analysis
print("\n" + "=" * 100)
print("WEEK 17 ANALYSIS")
print("=" * 100)

w17 = df[df['week'] == 17].copy()
print(f"\nTotal Week 17 games: {len(w17)}")
print(f"Games with scores: {w17['home_score'].notna().sum()}")
print(f"Games without scores: {w17['home_score'].isna().sum()}")

if w17['home_score'].notna().sum() > 0:
    w17_completed = w17[w17['home_score'].notna()].copy()
    print(f"\n--- Week 17 Completed Games ---")
    print(f"Total: {len(w17_completed)}")
    
    # Show all games
    print("\nAll Week 17 games:")
    for idx, row in w17_completed.iterrows():
        print(f"  {row['away_team']} @ {row['home_team']}: {row['away_score']:.0f}-{row['home_score']:.0f} | "
              f"Predicted: {row['predicted_winner']} ({row['confidence']:.1%}) | "
              f"Actual: {row['actual_winner']}")
else:
    print("\nNo completed games in Week 17 yet.")
    print("\nUpcoming Week 17 games:")
    w17_upcoming = w17[w17['home_score'].isna()].copy()
    for idx, row in w17_upcoming.head(10).iterrows():
        print(f"  {row['gameday']}: {row['away_team']} @ {row['home_team']} | "
              f"Predicted: {row['predicted_winner']} ({row['confidence']:.1%})")

print("\n" + "=" * 100)

