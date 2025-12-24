"""
Generate complete game-level derivation function from season-level function.

This script reads full_feature_derivation.py and adapts it for game-level derivation.
"""

import re
from pathlib import Path

# Read the season-level function
source_file = Path('../full_feature_derivation.py')
with open(source_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Extract the derive_all_features function
# Find the function definition
func_start = content.find('def derive_all_features(')
func_end = content.find('\n\nprint("  ✅ Feature derivation engine built")')

if func_start == -1 or func_end == -1:
    print("ERROR: Could not find function boundaries")
    exit(1)

function_code = content[func_start:func_end]

print(f"Original function: {len(function_code)} characters")

# Modifications for game-level derivation
modifications = [
    # 1. Change function signature
    (
        r'def derive_all_features\(team: str, pbp_reg: pd\.DataFrame, schedules_reg: pd\.DataFrame\) -> dict:',
        'def derive_game_features_complete(team: str, game_id: str, pbp: pd.DataFrame, schedules: pd.DataFrame) -> dict:'
    ),
    # 2. Update docstring
    (
        r'Derive ALL 319 ESPN features for a team from nfl-data-py',
        'Derive ALL ESPN features for a team\'s performance in a single game'
    ),
    (
        r'pbp_reg: Play-by-play data \(regular season only, uses nfl-data-py abbreviations\)',
        'pbp: Play-by-play data (uses nfl-data-py abbreviations)'
    ),
    (
        r'schedules_reg: Schedule data \(regular season only, uses nfl-data-py abbreviations\)',
        'schedules: Schedule data (uses nfl-data-py abbreviations)'
    ),
    # 3. Add game_id to features dict
    (
        r"features = \{'team': team\}",
        "features = {'team': team, 'game_id': game_id}"
    ),
]

# Apply modifications
modified_code = function_code
for pattern, replacement in modifications:
    modified_code = re.sub(pattern, replacement, modified_code)

# Add game filtering at the start of the function (after features dict)
game_filter_code = '''
    
    # Filter to this specific game
    pbp_game = pbp[pbp['game_id'] == game_id].copy()
    schedule_game = schedules[schedules['game_id'] == game_id].copy()
    
    if len(pbp_game) == 0:
        raise ValueError(f"No play-by-play data found for game_id: {game_id}")
    
    if len(schedule_game) == 0:
        raise ValueError(f"No schedule data found for game_id: {game_id}")
    
    # Use game-filtered data
    pbp_reg = pbp_game
    schedules_reg = schedule_game
'''

# Insert after features dict initialization
insert_pos = modified_code.find("features = {'team': team, 'game_id': game_id}") + len("features = {'team': team, 'game_id': game_id}")
modified_code = modified_code[:insert_pos] + game_filter_code + modified_code[insert_pos:]

# Replace games_played calculation with games_played = 1
modified_code = re.sub(
    r'games_played = len\(team_schedule\)',
    'games_played = 1  # Single game',
    modified_code
)

# Write the new function
output_file = Path('derive_game_features_complete.py')

header = '''"""
Phase 5A: Complete Game-Level Feature Derivation

This function derives ALL ESPN features at the game level (one row per team-game).
It is adapted from full_feature_derivation.py with modifications for game-level data.

Key differences from season-level derivation:
- Filters to a single game_id
- games_played = 1 (single game)
- Per-game stats are the same as raw stats (no division by games_played)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from team_abbreviation_mapping import espn_to_nfl_data_py, nfl_data_py_to_espn

warnings.filterwarnings('ignore')


'''

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(header)
    f.write(modified_code)

print(f"\n✅ Generated: {output_file}")
print(f"   Size: {len(header) + len(modified_code)} characters")
print(f"\nNext steps:")
print(f"1. Review the generated function")
print(f"2. Test on single game (BAL @ KC)")
print(f"3. Verify all 191 features are derived")

