"""
Check the year range of our training data
"""
import pandas as pd
from pathlib import Path

print("="*80)
print("📊 Training Data Year Range Analysis")
print("="*80)

# Check all_games.parquet
print("\n[1] All Games Dataset")
print("-"*80)
all_games = pd.read_parquet('data/processed/all_games.parquet')
print(f"Year range: {all_games['season'].min()} - {all_games['season'].max()}")
print(f"Total games: {len(all_games):,}")
print(f"Total columns: {len(all_games.columns)}")
print(f"\nGames per year:")
games_per_year = all_games.groupby('season').size()
print(games_per_year)

# Check train_games.parquet
print("\n[2] Training Dataset")
print("-"*80)
train_games = pd.read_parquet('data/processed/train_games.parquet')
print(f"Year range: {train_games['season'].min()} - {train_games['season'].max()}")
print(f"Total games: {len(train_games):,}")
print(f"Total columns: {len(train_games.columns)}")

# Check test_games.parquet
print("\n[3] Test Dataset")
print("-"*80)
test_games = pd.read_parquet('data/processed/test_games.parquet')
print(f"Year range: {test_games['season'].min()} - {test_games['season'].max()}")
print(f"Total games: {len(test_games):,}")
print(f"Total columns: {len(test_games.columns)}")

# Check config
print("\n[4] Config Settings")
print("-"*80)
try:
    from config import TRAIN_YEARS, TEST_YEARS
    print(f"TRAIN_YEARS: {TRAIN_YEARS}")
    print(f"TEST_YEARS: {TEST_YEARS}")
except:
    print("Could not load config")

# Summary
print("\n" + "="*80)
print("📋 Summary")
print("="*80)
print(f"Historical data spans: {all_games['season'].min()} - {all_games['season'].max()}")
print(f"Total years: {all_games['season'].max() - all_games['season'].min() + 1}")
print(f"Training years: {train_games['season'].min()} - {train_games['season'].max()}")
print(f"Test years: {test_games['season'].min()} - {test_games['season'].max()}")
print(f"\nESPN data collected: 2024, 2025")
print(f"Missing ESPN data years: {all_games['season'].min()} - 2023")
print("="*80)

# Check what features we currently have
print("\n[5] Current Features")
print("-"*80)
print(f"Total features: {len(all_games.columns)}")
print(f"\nSample features:")
for col in all_games.columns[:20]:
    print(f"  - {col}")
print(f"  ... and {len(all_games.columns) - 20} more")

