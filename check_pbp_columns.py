"""Check available columns in play-by-play data"""
import pandas as pd

pbp = pd.read_parquet('data/cache/pbp_2024.parquet')

print(f"Play-by-play columns ({len(pbp.columns)}):")
print()

# Group columns by category
tackle_cols = [c for c in pbp.columns if 'tackle' in c.lower()]
kick_cols = [c for c in pbp.columns if 'kick' in c.lower() or 'punt' in c.lower() or 'field_goal' in c.lower()]
return_cols = [c for c in pbp.columns if 'return' in c.lower()]
interception_cols = [c for c in pbp.columns if 'interception' in c.lower() or 'int_' in c.lower()]

print("TACKLE COLUMNS:")
for c in sorted(tackle_cols):
    print(f"  - {c}")

print("\nKICKING/PUNTING COLUMNS:")
for c in sorted(kick_cols):
    print(f"  - {c}")

print("\nRETURN COLUMNS:")
for c in sorted(return_cols):
    print(f"  - {c}")

print("\nINTERCEPTION COLUMNS:")
for c in sorted(interception_cols):
    print(f"  - {c}")

# Check for specific columns we need
needed_cols = [
    'solo_tackle',
    'assist_tackle',
    'tackle_for_loss',
    'interception_player_id',
    'return_yards',
    'kickoff_returner_player_id',
    'punt_returner_player_id',
    'field_goal_result',
    'kick_distance',
    'extra_point_result',
    'touchback',
    'fair_catch',
    'blocked_player_id'
]

print("\n" + "="*80)
print("CHECKING FOR NEEDED COLUMNS:")
print("="*80)
for col in needed_cols:
    exists = "✅" if col in pbp.columns else "❌"
    print(f"{exists} {col}")

