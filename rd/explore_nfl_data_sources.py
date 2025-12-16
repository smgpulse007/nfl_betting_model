"""
Explore Available NFL Data Sources
==================================
Check what data is available from nfl-data-py that we're not using.
"""
import nfl_data_py as nfl
import pandas as pd

print('=' * 80)
print('EXPLORING NFL-DATA-PY AVAILABLE DATA')
print('=' * 80)

# 1. Check injury data availability
print("\n1️⃣  INJURY DATA (import_injuries)")
try:
    injuries = nfl.import_injuries([2024])
    print(f"   Loaded {len(injuries)} injury records for 2024")
    print(f"   Columns: {list(injuries.columns)}")
    print(f"\n   Sample:")
    print(injuries[['season','team','full_name','position','report_status']].head())
except Exception as e:
    print(f"   Error: {e}")

# 2. Check QBR data
print("\n2️⃣  QBR DATA (import_qbr)")
try:
    qbr = nfl.import_qbr([2024])
    print(f"   Loaded {len(qbr)} QBR records")
    print(f"   Columns: {list(qbr.columns)}")
except Exception as e:
    print(f"   Error: {e}")

# 3. Check snap counts
print("\n3️⃣  SNAP COUNTS (import_snap_counts)")
try:
    snaps = nfl.import_snap_counts([2024])
    print(f"   Loaded {len(snaps)} snap count records")
    print(f"   Columns: {list(snaps.columns)[:15]}...")
except Exception as e:
    print(f"   Error: {e}")

# 4. Check depth charts
print("\n4️⃣  DEPTH CHARTS (import_depth_charts)")
try:
    depth = nfl.import_depth_charts([2024])
    print(f"   Loaded {len(depth)} depth chart records")
    print(f"   Columns: {list(depth.columns)}")
except Exception as e:
    print(f"   Error: {e}")

# 5. Check Next Gen Stats
print("\n5️⃣  NEXT GEN STATS - Passing (import_ngs_data)")
try:
    ngs_pass = nfl.import_ngs_data('passing', [2024])
    print(f"   Loaded {len(ngs_pass)} NGS passing records")
    print(f"   Key columns: {[c for c in ngs_pass.columns if 'avg' in c.lower() or 'time' in c.lower()]}")
except Exception as e:
    print(f"   Error: {e}")

print("\n6️⃣  NEXT GEN STATS - Rushing")
try:
    ngs_rush = nfl.import_ngs_data('rushing', [2024])
    print(f"   Loaded {len(ngs_rush)} NGS rushing records")
except Exception as e:
    print(f"   Error: {e}")

# 7. Check seasonal data
print("\n7️⃣  SEASONAL DATA (import_seasonal_data)")
try:
    seasonal = nfl.import_seasonal_data([2024])
    print(f"   Loaded {len(seasonal)} seasonal records")
    print(f"   Columns: {list(seasonal.columns)[:15]}...")
except Exception as e:
    print(f"   Error: {e}")

# 8. Check PBP sample for EPA columns
print("\n8️⃣  PLAY-BY-PLAY EPA COLUMNS (sample)")
try:
    pbp = nfl.import_pbp_data([2024], downcast=True)
    epa_cols = [c for c in pbp.columns if 'epa' in c.lower() or 'wpa' in c.lower()]
    print(f"   EPA/WPA related columns: {len(epa_cols)}")
    for col in epa_cols[:10]:
        print(f"      - {col}")
except Exception as e:
    print(f"   Error: {e}")

# 9. Check weekly data for player metrics
print("\n9️⃣  WEEKLY PLAYER DATA (import_weekly_data)")
try:
    weekly = nfl.import_weekly_data([2024])
    print(f"   Loaded {len(weekly)} player-week records")
    qb_cols = [c for c in weekly.columns if 'pass' in c.lower() or 'qb' in c.lower()][:10]
    print(f"   QB-related columns: {qb_cols}")
except Exception as e:
    print(f"   Error: {e}")

# 10. Check win totals / futures
print("\n🔟  WIN TOTALS (import_win_totals)")
try:
    win_totals = nfl.import_win_totals([2024])
    print(f"   Loaded {len(win_totals)} win total records")
    print(f"   Columns: {list(win_totals.columns)}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 80)
print("SUMMARY: Available Data We Could Add")
print("=" * 80)
print("""
   ✅ INJURIES      - Weekly injury reports (Out/Doubtful/Questionable)
   ✅ QBR           - ESPN's Total QBR metric
   ✅ SNAP COUNTS   - Player participation rates
   ✅ DEPTH CHARTS  - Weekly roster depth
   ✅ NEXT GEN      - Advanced tracking (time to throw, separation)
   ✅ PLAY-BY-PLAY  - EPA, WPA, CPOE columns available
   ✅ WEEKLY DATA   - Player-level stats per week
   ✅ WIN TOTALS    - Preseason futures (team strength proxy)
""")

