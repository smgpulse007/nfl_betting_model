"""
Check if injury data is available from nfl_data_py
"""

import pandas as pd

print("=" * 120)
print("CHECKING INJURY DATA AVAILABILITY")
print("=" * 120)

print("\n[1/3] Checking nfl_data_py capabilities...")

try:
    import nfl_data_py as nfl
    print("  ✅ nfl_data_py imported successfully")
    
    # Check available functions
    available_functions = [func for func in dir(nfl) if not func.startswith('_')]
    print(f"\n  Available functions ({len(available_functions)}):")
    
    injury_related = [func for func in available_functions if 'injury' in func.lower() or 'health' in func.lower()]
    if injury_related:
        print(f"  ✅ Injury-related functions found:")
        for func in injury_related:
            print(f"     • {func}")
    else:
        print(f"  ⚠️ No injury-specific functions found")
    
    # Check for weekly data functions
    weekly_related = [func for func in available_functions if 'week' in func.lower()]
    if weekly_related:
        print(f"\n  Weekly data functions:")
        for func in weekly_related:
            print(f"     • {func}")
    
    # List all functions for reference
    print(f"\n  All available functions:")
    for func in sorted(available_functions)[:20]:
        print(f"     • {func}")
    if len(available_functions) > 20:
        print(f"     ... and {len(available_functions) - 20} more")
    
except ImportError as e:
    print(f"  ❌ Failed to import nfl_data_py: {e}")
    exit(1)

print(f"\n[2/3] Checking existing data for injury indicators...")

# Load phase 6 data
df_phase6 = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')
print(f"  ✅ Loaded phase 6 data: {len(df_phase6)} games")

# Check all column names
all_cols = df_phase6.columns.tolist()
print(f"  Total columns: {len(all_cols)}")

# Search for injury/health related columns
injury_cols = [col for col in all_cols if any(keyword in col.lower() for keyword in ['injury', 'health', 'out', 'questionable', 'doubtful'])]

if injury_cols:
    print(f"\n  ✅ Found {len(injury_cols)} potential injury-related columns:")
    for col in injury_cols[:20]:
        print(f"     • {col}")
else:
    print(f"\n  ❌ No injury-related columns found in existing data")

# Check for player-related columns
player_cols = [col for col in all_cols if 'player' in col.lower()]
if player_cols:
    print(f"\n  Player-related columns ({len(player_cols)}):")
    for col in player_cols[:10]:
        print(f"     • {col}")

print(f"\n[3/3] Testing injury data import...")

try:
    # Try to import injuries if function exists
    if hasattr(nfl, 'import_injuries'):
        print("  ✅ import_injuries function exists")
        print("  Attempting to fetch 2024 injury data...")
        
        injuries_2024 = nfl.import_injuries([2024])
        print(f"  ✅ Successfully fetched injury data: {len(injuries_2024)} records")
        print(f"\n  Injury data columns:")
        for col in injuries_2024.columns:
            print(f"     • {col}")
        
        print(f"\n  Sample injury data:")
        print(injuries_2024.head(3))
        
    else:
        print("  ⚠️ import_injuries function not found")
        print("  Checking for alternative injury data sources...")
        
        # Try weekly data which might include injury info
        if hasattr(nfl, 'import_weekly_data'):
            print("  ✅ import_weekly_data function exists")
            print("  Attempting to fetch 2024 week 1 data...")
            
            weekly_2024 = nfl.import_weekly_data([2024])
            print(f"  ✅ Successfully fetched weekly data: {len(weekly_2024)} records")
            print(f"\n  Weekly data columns:")
            for col in weekly_2024.columns:
                print(f"     • {col}")
            
            # Check for injury indicators in weekly data
            injury_indicators = [col for col in weekly_2024.columns if any(keyword in col.lower() for keyword in ['injury', 'status', 'active'])]
            if injury_indicators:
                print(f"\n  ✅ Found injury indicators in weekly data:")
                for col in injury_indicators:
                    print(f"     • {col}")
        else:
            print("  ⚠️ import_weekly_data function not found")

except Exception as e:
    print(f"  ❌ Error fetching injury data: {e}")

print(f"\n{'='*120}")
print("SUMMARY")
print("=" * 120)

print(f"\n📊 FINDINGS:")
print(f"   • nfl_data_py library: Available")
print(f"   • Injury data in existing features: {'Yes' if injury_cols else 'No'}")
print(f"   • Injury data from nfl_data_py: {'Available' if hasattr(nfl, 'import_injuries') else 'Not found'}")

print(f"\n💡 RECOMMENDATIONS:")
if not injury_cols:
    print(f"   1. ❌ Current model does NOT use injury data")
    print(f"   2. ⚠️ This is a significant limitation")
    print(f"   3. ✅ Consider adding injury features in future versions")
    print(f"   4. 📊 Injury data could improve late-season predictions")
else:
    print(f"   1. ✅ Injury data is available in features")
    print(f"   2. 📊 Verify injury data is up-to-date for 2025")

print(f"\n{'='*120}")

