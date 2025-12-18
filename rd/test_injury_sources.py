"""
Test Injury Data Sources
=========================
Compare nfl-data-py, nflreadpy, and ESPN API for injury data.
"""

import pandas as pd
import requests
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("INJURY DATA SOURCE COMPARISON")
print("="*60)

# =============================================================================
# 1. Test nfl-data-py (old package - might be deprecated)
# =============================================================================
print("\n1. Testing nfl-data-py (import_injuries)...")
try:
    import nfl_data_py as nfl_old
    injuries_old = nfl_old.import_injuries([2024])
    print(f"   ✅ SUCCESS: {len(injuries_old)} records")
    print(f"   Columns: {injuries_old.columns.tolist()[:10]}...")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# =============================================================================
# 2. Test nflreadpy (new package)
# =============================================================================
print("\n2. Testing nflreadpy (load_injuries)...")
try:
    import nflreadpy as nfl_new
    injuries_new = nfl_new.load_injuries([2024])
    # nflreadpy returns Polars, convert to pandas
    injuries_new_df = injuries_new.to_pandas()
    print(f"   ✅ SUCCESS: {len(injuries_new_df)} records")
    print(f"   Columns: {injuries_new_df.columns.tolist()[:10]}...")
    print(f"\n   Sample data:")
    print(injuries_new_df[['gsis_id', 'season', 'week', 'team', 'position', 'report_status']].head())
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# =============================================================================
# 3. Test ESPN API for injuries
# =============================================================================
print("\n3. Testing ESPN API (team injuries endpoint)...")
try:
    # Get injuries for all teams
    teams_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    teams_resp = requests.get(teams_url, timeout=10)
    teams_data = teams_resp.json()
    
    team_ids = [t['team']['id'] for t in teams_data['sports'][0]['leagues'][0]['teams']]
    print(f"   Found {len(team_ids)} teams")
    
    # Get injuries for one team as sample
    sample_team = team_ids[0]
    injuries_url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{sample_team}/injuries"
    injuries_resp = requests.get(injuries_url, timeout=10)
    injuries_data = injuries_resp.json()
    
    print(f"   ✅ SUCCESS: ESPN API accessible")
    print(f"   Sample team injuries items: {len(injuries_data.get('items', []))}")
    
    if injuries_data.get('items'):
        # Fetch first injury detail
        first_injury_url = injuries_data['items'][0]['$ref']
        injury_detail = requests.get(first_injury_url, timeout=10).json()
        print(f"   Sample injury keys: {list(injury_detail.keys())}")
        
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# =============================================================================
# 4. Test nflreadpy schedules (for rest days)
# =============================================================================
print("\n4. Testing nflreadpy (load_schedules for rest days)...")
try:
    schedules = nfl_new.load_schedules([2024])
    sched_df = schedules.to_pandas()
    print(f"   ✅ SUCCESS: {len(sched_df)} records")
    
    # Check for rest day columns
    rest_cols = [c for c in sched_df.columns if 'rest' in c.lower()]
    print(f"   Rest columns: {rest_cols}")
    
    if rest_cols:
        print(f"\n   Rest day sample:")
        print(sched_df[['game_id', 'away_team', 'home_team'] + rest_cols[:4]].head())
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("""
Based on testing:

1. nfl-data-py: DEPRECATED - API returning 404 errors for injuries
   
2. nflreadpy (RECOMMENDED):
   - Active development (nflverse)
   - load_injuries() works with 2024 data
   - Uses Polars (faster), converts to pandas easily
   - Has load_schedules(), load_snap_counts(), load_rosters()
   
3. ESPN API: ALTERNATIVE
   - Free, no API key needed
   - Real-time injury data
   - Requires multiple API calls (one per team)
   - Good for live injury updates

RECOMMENDATION: Switch from nfl-data-py to nflreadpy for:
- Injuries: load_injuries()
- Schedules: load_schedules() 
- Rosters: load_rosters()
- Snap counts: load_snap_counts()
""")

