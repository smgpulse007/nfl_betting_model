"""Check weeks in nfl-data-py"""
import pandas as pd

pbp = pd.read_parquet('data/cache/pbp_2024.parquet')

print("Weeks in 2024 data:")
print(pbp['week'].value_counts().sort_index())

print(f"\nMax week: {pbp['week'].max()}")
print(f"Min week: {pbp['week'].min()}")

# Check ARI
ari = pbp[pbp['posteam']=='ARI']
print(f"\nARI weeks: {sorted(ari['week'].unique())}")
print(f"ARI total games: {len(ari['game_id'].unique())}")

# Check if week 18+ exists (playoffs)
playoff_weeks = pbp[pbp['week'] > 18]
if len(playoff_weeks) > 0:
    print(f"\nPlayoff weeks found: {sorted(playoff_weeks['week'].unique())}")
    print(f"Playoff plays: {len(playoff_weeks):,}")
else:
    print("\nNo playoff weeks found (week > 18)")

# Filter to weeks 1-17 only (regular season)
reg_season = pbp[pbp['week'] <= 17]
print(f"\nRegular season (weeks 1-17):")
print(f"  Total plays: {len(reg_season):,}")
print(f"  Total plays (all weeks): {len(pbp):,}")

ari_reg = reg_season[reg_season['posteam'] == 'ARI']
print(f"\nARI regular season:")
print(f"  Games: {len(ari_reg['game_id'].unique())}")
print(f"  Passing yards: {ari_reg[ari_reg['play_type']=='pass']['yards_gained'].sum()}")

