"""
Comprehensive exploration of ALL nfl-data-py data sources.
This script profiles every available data source to identify features
that could be useful for betting models.
"""

import nfl_data_py as nfl
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def profile_dataframe(df, name):
    """Profile a dataframe and return summary stats."""
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts().to_string())
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        pct = 100 * non_null / len(df)
        print(f"  {i+1:3}. {col:<40} {dtype:<15} ({pct:.0f}% non-null)")
    return df.columns.tolist()

def main():
    print("="*70)
    print("NFL DATA-PY: COMPREHENSIVE DATA SOURCE EXPLORATION")
    print("="*70)
    
    all_sources = {}
    
    # 1. Play-by-Play Data (THE MAIN SOURCE - 372+ columns)
    print("\n\n🏈 1. PLAY-BY-PLAY DATA (Core)")
    try:
        pbp = nfl.import_pbp_data([2024], downcast=True)
        all_sources['pbp'] = profile_dataframe(pbp, "Play-by-Play (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 2. Schedule Data
    print("\n\n📅 2. SCHEDULE DATA")
    try:
        sched = nfl.import_schedules([2024])
        all_sources['schedules'] = profile_dataframe(sched, "Schedules (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 3. Next Gen Stats - Passing
    print("\n\n🎯 3. NEXT GEN STATS - PASSING")
    try:
        ngs_pass = nfl.import_ngs_data('passing', [2024])
        all_sources['ngs_passing'] = profile_dataframe(ngs_pass, "NGS Passing (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 4. Next Gen Stats - Rushing
    print("\n\n🏃 4. NEXT GEN STATS - RUSHING")
    try:
        ngs_rush = nfl.import_ngs_data('rushing', [2024])
        all_sources['ngs_rushing'] = profile_dataframe(ngs_rush, "NGS Rushing (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 5. Next Gen Stats - Receiving
    print("\n\n🙌 5. NEXT GEN STATS - RECEIVING")
    try:
        ngs_rec = nfl.import_ngs_data('receiving', [2024])
        all_sources['ngs_receiving'] = profile_dataframe(ngs_rec, "NGS Receiving (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 6. Weekly Player Stats
    print("\n\n📈 6. WEEKLY PLAYER STATS")
    try:
        weekly = nfl.import_weekly_data([2024])
        all_sources['weekly'] = profile_dataframe(weekly, "Weekly Player Stats (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 7. Seasonal Player Stats
    print("\n\n📊 7. SEASONAL PLAYER STATS")
    try:
        seasonal = nfl.import_seasonal_data([2024])
        all_sources['seasonal'] = profile_dataframe(seasonal, "Seasonal Player Stats (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 8. Injuries
    print("\n\n🏥 8. INJURY REPORTS")
    try:
        injuries = nfl.import_injuries([2024])
        all_sources['injuries'] = profile_dataframe(injuries, "Injuries (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 9. Snap Counts
    print("\n\n⏱️ 9. SNAP COUNTS")
    try:
        snaps = nfl.import_snap_counts([2024])
        all_sources['snap_counts'] = profile_dataframe(snaps, "Snap Counts (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 10. Depth Charts
    print("\n\n📋 10. DEPTH CHARTS")
    try:
        depth = nfl.import_depth_charts([2024])
        all_sources['depth_charts'] = profile_dataframe(depth, "Depth Charts (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 11. QBR (ESPN)
    print("\n\n🏆 11. ESPN QBR")
    try:
        qbr = nfl.import_qbr([2024])
        all_sources['qbr'] = profile_dataframe(qbr, "ESPN QBR (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 12. PFR Passing Stats
    print("\n\n📊 12. PRO-FOOTBALL-REFERENCE - PASSING")
    try:
        pfr_pass = nfl.import_weekly_pfr('pass', [2024])
        all_sources['pfr_passing'] = profile_dataframe(pfr_pass, "PFR Passing (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 13. PFR Rushing Stats
    print("\n\n📊 13. PRO-FOOTBALL-REFERENCE - RUSHING")
    try:
        pfr_rush = nfl.import_weekly_pfr('rush', [2024])
        all_sources['pfr_rushing'] = profile_dataframe(pfr_rush, "PFR Rushing (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 14. PFR Receiving Stats
    print("\n\n📊 14. PRO-FOOTBALL-REFERENCE - RECEIVING")
    try:
        pfr_rec = nfl.import_weekly_pfr('rec', [2024])
        all_sources['pfr_receiving'] = profile_dataframe(pfr_rec, "PFR Receiving (2024)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 15. FTN Charting Data (2022+)
    print("\n\n📝 15. FTN CHARTING DATA")
    try:
        ftn = nfl.import_ftn_data([2024])
        all_sources['ftn_charting'] = profile_dataframe(ftn, "FTN Charting (2024)")
    except Exception as e:
        print(f"Error: {e}")

    # 16. Officials
    print("\n\n👨‍⚖️ 16. OFFICIALS")
    try:
        officials = nfl.import_officials([2024])
        all_sources['officials'] = profile_dataframe(officials, "Officials (2024)")
    except Exception as e:
        print(f"Error: {e}")

    # 17. Rosters
    print("\n\n👥 17. ROSTERS")
    try:
        rosters = nfl.import_rosters([2024])
        all_sources['rosters'] = profile_dataframe(rosters, "Rosters (2024)")
    except Exception as e:
        print(f"Error: {e}")

    # 18. Win Totals (Vegas)
    print("\n\n💰 18. WIN TOTALS (VEGAS)")
    try:
        win_totals = nfl.import_win_totals([2024])
        all_sources['win_totals'] = profile_dataframe(win_totals, "Win Totals (2024)")
    except Exception as e:
        print(f"Error: {e}")

    # 19. Sportsbook Lines
    print("\n\n📈 19. SPORTSBOOK LINES")
    try:
        sc_lines = nfl.import_sc_lines([2024])
        all_sources['sc_lines'] = profile_dataframe(sc_lines, "Sportsbook Lines (2024)")
    except Exception as e:
        print(f"Error: {e}")

    # Summary
    print("\n\n" + "="*70)
    print("📋 SUMMARY: ALL DATA SOURCES")
    print("="*70)
    for name, cols in all_sources.items():
        print(f"  {name:<20}: {len(cols):>4} columns")

    total_cols = sum(len(cols) for cols in all_sources.values())
    print(f"\n  TOTAL UNIQUE COLUMNS: ~{total_cols}")

if __name__ == "__main__":
    main()

