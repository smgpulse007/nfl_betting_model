"""Quick inspection of ESPN data"""
import pandas as pd

# Load 2024 stats
print("Loading ESPN 2024 data...")
stats = pd.read_parquet('data/espn_raw/team_stats_2024.parquet')
records = pd.read_parquet('data/espn_raw/team_records_2024.parquet')

print(f"\nTeam Stats: {stats.shape}")
print(f"Team Records: {records.shape}")

# Merge
df = pd.merge(stats, records, on='team', how='outer')
df = df.set_index('team')

print(f"\nMerged: {df.shape}")

# Show Arizona Cardinals
print("\n" + "=" * 80)
print("Arizona Cardinals 2024 - Sample Features")
print("=" * 80)

ari = df.loc['ARI']

print("\nPassing:")
print(f"  passing_passingYards: {ari['passing_passingYards']}")
print(f"  passing_passingAttempts: {ari['passing_passingAttempts']}")
print(f"  passing_completions: {ari['passing_completions']}")
print(f"  passing_passingTouchdowns: {ari['passing_passingTouchdowns']}")
print(f"  passing_interceptions: {ari['passing_interceptions']}")
print(f"  passing_completionPct: {ari['passing_completionPct']}")
print(f"  passing_QBRating: {ari['passing_QBRating']}")

print("\nRushing:")
print(f"  rushing_rushingYards: {ari['rushing_rushingYards']}")
print(f"  rushing_rushingAttempts: {ari['rushing_rushingAttempts']}")
print(f"  rushing_rushingTouchdowns: {ari['rushing_rushingTouchdowns']}")

print("\nRecords:")
print(f"  total_wins: {ari['total_wins']}")
print(f"  total_losses: {ari['total_losses']}")
print(f"  total_pointsFor: {ari['total_pointsFor']}")
print(f"  total_pointsAgainst: {ari['total_pointsAgainst']}")

print("\n" + "=" * 80)
print("All Teams - Passing Yards")
print("=" * 80)
print(df[['passing_passingYards']].sort_values('passing_passingYards', ascending=False).head(10))

print("\n✅ ESPN data loaded successfully!")

