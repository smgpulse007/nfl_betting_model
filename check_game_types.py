"""Check game types in nfl-data-py"""
import pandas as pd

pbp = pd.read_parquet('data/cache/pbp_2024.parquet')

print("Game types in nfl-data-py 2024:")
print(pbp['game_type'].value_counts())

print("\nARI games:")
ari = pbp[pbp['posteam']=='ARI']
games = ari['game_id'].unique()
print(f"Total unique games: {len(games)}")
print("\nGame IDs:")
for game in sorted(games):
    print(f"  {game}")

# Check if there are playoff games
print("\nChecking for playoff games...")
playoff = pbp[pbp['game_type'] != 'REG']
if len(playoff) > 0:
    print(f"Found {len(playoff)} playoff plays")
    print(f"Playoff games: {playoff['game_id'].unique()}")
else:
    print("No playoff games found")

# Filter to regular season only
print("\nFiltering to regular season only...")
reg_season = pbp[pbp['game_type'] == 'REG']
print(f"Regular season plays: {len(reg_season):,}")
print(f"Total plays (all types): {len(pbp):,}")

# Check ARI regular season
ari_reg = reg_season[reg_season['posteam'] == 'ARI']
print(f"\nARI regular season games: {len(ari_reg['game_id'].unique())}")
print(f"ARI all games: {len(ari['game_id'].unique())}")

