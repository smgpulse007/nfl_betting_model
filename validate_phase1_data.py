"""
Validate Phase 1 Data Collection
Check data quality and completeness
"""
import pandas as pd
from pathlib import Path

def validate_data():
    """Validate all collected ESPN data"""
    
    data_dir = Path('data/espn_raw')
    
    print("="*80)
    print("📋 Phase 1 Data Validation Report")
    print("="*80)
    
    # Check 2024 Team Stats
    print("\n[1/4] 2024 Team Stats")
    print("-" * 80)
    stats_2024 = pd.read_parquet(data_dir / 'team_stats_2024.parquet')
    print(f"✅ Teams: {len(stats_2024)}/32")
    print(f"✅ Columns: {len(stats_2024.columns)}")
    print(f"✅ Missing values: {stats_2024.isnull().sum().sum()} ({stats_2024.isnull().sum().sum() / (len(stats_2024) * len(stats_2024.columns)) * 100:.2f}%)")
    print(f"✅ Sample teams: {', '.join(stats_2024['team'].head(5).tolist())}")
    
    # Check 2024 Team Records
    print("\n[2/4] 2024 Team Records")
    print("-" * 80)
    records_2024 = pd.read_parquet(data_dir / 'team_records_2024.parquet')
    print(f"✅ Teams: {len(records_2024)}/32")
    print(f"✅ Columns: {len(records_2024.columns)}")
    print(f"✅ Missing values: {records_2024.isnull().sum().sum()} ({records_2024.isnull().sum().sum() / (len(records_2024) * len(records_2024.columns)) * 100:.2f}%)")
    print(f"✅ Sample teams: {', '.join(records_2024['team'].head(5).tolist())}")
    
    # Check 2025 Team Stats
    print("\n[3/4] 2025 Team Stats")
    print("-" * 80)
    stats_2025 = pd.read_parquet(data_dir / 'team_stats_2025.parquet')
    print(f"✅ Teams: {len(stats_2025)}/32")
    print(f"✅ Columns: {len(stats_2025.columns)}")
    print(f"✅ Missing values: {stats_2025.isnull().sum().sum()} ({stats_2025.isnull().sum().sum() / (len(stats_2025) * len(stats_2025.columns)) * 100:.2f}%)")
    print(f"✅ Sample teams: {', '.join(stats_2025['team'].head(5).tolist())}")
    
    # Check 2025 Team Records
    print("\n[4/4] 2025 Team Records")
    print("-" * 80)
    records_2025 = pd.read_parquet(data_dir / 'team_records_2025.parquet')
    print(f"✅ Teams: {len(records_2025)}/32")
    print(f"✅ Columns: {len(records_2025.columns)}")
    print(f"✅ Missing values: {records_2025.isnull().sum().sum()} ({records_2025.isnull().sum().sum() / (len(records_2025) * len(records_2025.columns)) * 100:.2f}%)")
    print(f"✅ Sample teams: {', '.join(records_2025['team'].head(5).tolist())}")
    
    # Check 2025 Injuries
    print("\n[BONUS] 2025 Injuries (Weeks 1-16)")
    print("-" * 80)
    injuries_2025 = pd.read_parquet(data_dir / 'injuries_2025_weeks_1-16.parquet')
    print(f"✅ Total injuries: {len(injuries_2025)}")
    print(f"✅ Columns: {len(injuries_2025.columns)}")
    print(f"✅ Weeks covered: {sorted(injuries_2025['week'].unique().tolist())}")
    print(f"✅ Teams with injuries: {injuries_2025['team'].nunique()}")
    
    # Summary Statistics
    print("\n" + "="*80)
    print("📊 Summary Statistics")
    print("="*80)
    
    total_stats_2024 = len(stats_2024) * len(stats_2024.columns)
    total_stats_2025 = len(stats_2025) * len(stats_2025.columns)
    total_records_2024 = len(records_2024) * len(records_2024.columns)
    total_records_2025 = len(records_2025) * len(records_2025.columns)
    total_injuries = len(injuries_2025)
    
    total_data_points = total_stats_2024 + total_stats_2025 + total_records_2024 + total_records_2025 + total_injuries
    
    print(f"\n2024 Team Stats: {total_stats_2024:,} data points")
    print(f"2024 Team Records: {total_records_2024:,} data points")
    print(f"2025 Team Stats: {total_stats_2025:,} data points")
    print(f"2025 Team Records: {total_records_2025:,} data points")
    print(f"2025 Injuries: {total_injuries:,} injury records")
    print(f"\n{'='*80}")
    print(f"🎉 TOTAL DATA POINTS: {total_data_points:,}")
    print(f"{'='*80}")
    
    # Data Quality Assessment
    print("\n📋 Data Quality Assessment")
    print("="*80)
    
    all_teams_present = (len(stats_2024) == 32 and len(records_2024) == 32 and 
                        len(stats_2025) == 32 and len(records_2025) == 32)
    
    total_missing = (stats_2024.isnull().sum().sum() + records_2024.isnull().sum().sum() +
                    stats_2025.isnull().sum().sum() + records_2025.isnull().sum().sum())
    
    missing_pct = total_missing / (total_stats_2024 + total_stats_2025 + total_records_2024 + total_records_2025) * 100
    
    print(f"✅ All 32 teams present: {'YES' if all_teams_present else 'NO'}")
    print(f"✅ Total missing values: {total_missing:,} ({missing_pct:.2f}%)")
    print(f"✅ Injury data collected: YES ({len(injuries_2025):,} records)")
    
    if all_teams_present and missing_pct < 5:
        print(f"\n🎉 DATA QUALITY: EXCELLENT")
        print(f"✅ Phase 1 Data Collection: COMPLETE")
    elif all_teams_present and missing_pct < 10:
        print(f"\n⚠️ DATA QUALITY: GOOD (minor missing values)")
        print(f"✅ Phase 1 Data Collection: COMPLETE")
    else:
        print(f"\n❌ DATA QUALITY: NEEDS REVIEW")
        print(f"⚠️ Phase 1 Data Collection: INCOMPLETE")
    
    print("="*80)
    
    # Sample data preview
    print("\n📊 Sample Data Preview (2024 Team Stats)")
    print("="*80)
    print(stats_2024[['team', 'season']].head(10))
    
    print("\n📊 Sample Column Names (2024 Team Stats)")
    print("="*80)
    print(f"First 10 columns: {list(stats_2024.columns[:10])}")
    
    print("\n📊 Sample Injury Data")
    print("="*80)
    if len(injuries_2025) > 0:
        print(injuries_2025[['team', 'week', 'player_name', 'position', 'status']].head(10))
    
    print("\n" + "="*80)
    print("✅ Validation Complete!")
    print("="*80)

if __name__ == "__main__":
    validate_data()

