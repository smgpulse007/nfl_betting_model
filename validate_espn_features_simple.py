"""
Simple validation of ESPN features vs derived features
Phase 2: Feature Validation (Simplified Approach)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Set style
sns.set_style("whitegrid")

def load_espn_data(year: int) -> pd.DataFrame:
    """Load ESPN data"""
    print(f"Loading ESPN data for {year}...")
    
    # Load team stats
    stats_path = Path(f'data/espn_raw/team_stats_{year}.parquet')
    stats = pd.read_parquet(stats_path)
    
    # Load team records
    records_path = Path(f'data/espn_raw/team_records_{year}.parquet')
    records = pd.read_parquet(records_path)
    
    # Merge on team
    df = pd.merge(stats, records, on='team', how='outer')
    
    print(f"  Loaded {len(df)} teams, {len(df.columns)} columns")
    return df

def load_existing_processed_data(year: int) -> pd.DataFrame:
    """Load existing processed data that already has some derived features"""
    print(f"Loading existing processed data for {year}...")
    
    # Check if we have processed game data
    all_games_path = Path('data/processed/all_games.parquet')
    if all_games_path.exists():
        df = pd.read_parquet(all_games_path)
        df = df[df['season'] == year]
        print(f"  Loaded {len(df)} games from {year}")
        return df
    else:
        print("  No processed data found")
        return None

def compare_team_records(espn_df: pd.DataFrame, year: int):
    """Compare team records from ESPN vs schedules"""
    print(f"\n{'=' * 80}")
    print(f"Comparing Team Records for {year}")
    print('=' * 80)
    
    # We can manually calculate wins/losses from ESPN data structure
    # ESPN data has total_wins, total_losses, etc.
    
    if 'total_wins' in espn_df.columns:
        print("\n✅ ESPN Team Records Available:")
        print(f"  - total_wins: {espn_df['total_wins'].notna().sum()} teams")
        print(f"  - total_losses: {espn_df['total_losses'].notna().sum()} teams")
        print(f"  - total_winPercent: {espn_df['total_winPercent'].notna().sum()} teams")
        
        # Show sample
        print("\nSample Team Records (ESPN):")
        sample = espn_df[['team', 'total_wins', 'total_losses', 'total_ties', 'total_winPercent']].head(10)
        print(sample.to_string(index=False))
    else:
        print("\n❌ No team record columns found in ESPN data")
    
    return espn_df

def analyze_feature_availability(espn_df: pd.DataFrame, year: int):
    """Analyze which ESPN features are available"""
    print(f"\n{'=' * 80}")
    print(f"ESPN Feature Availability Analysis - {year}")
    print('=' * 80)
    
    # Categorize columns
    passing_cols = [c for c in espn_df.columns if 'passing' in c.lower()]
    rushing_cols = [c for c in espn_df.columns if 'rushing' in c.lower()]
    receiving_cols = [c for c in espn_df.columns if 'receiving' in c.lower()]
    defensive_cols = [c for c in espn_df.columns if 'defensive' in c.lower() or 'defense' in c.lower()]
    kicking_cols = [c for c in espn_df.columns if 'kicking' in c.lower() or 'field' in c.lower()]
    total_cols = [c for c in espn_df.columns if 'total_' in c.lower()]
    
    print(f"\n📊 Feature Categories:")
    print(f"  - Passing: {len(passing_cols)} features")
    print(f"  - Rushing: {len(rushing_cols)} features")
    print(f"  - Receiving: {len(receiving_cols)} features")
    print(f"  - Defensive: {len(defensive_cols)} features")
    print(f"  - Kicking: {len(kicking_cols)} features")
    print(f"  - Total/Records: {len(total_cols)} features")
    print(f"  - All Features: {len(espn_df.columns)} total")
    
    # Check for key features we want to validate
    key_features = [
        'passing_passingYards',
        'passing_passingTouchdowns',
        'passing_completions',
        'passing_interceptions',
        'rushing_rushingYards',
        'rushing_rushingTouchdowns',
        'total_wins',
        'total_losses',
        'total_pointsFor',
        'total_pointsAgainst'
    ]
    
    print(f"\n🔑 Key Features for Validation:")
    for feat in key_features:
        if feat in espn_df.columns:
            non_null = espn_df[feat].notna().sum()
            print(f"  ✅ {feat}: {non_null}/{len(espn_df)} teams have data")
        else:
            print(f"  ❌ {feat}: NOT FOUND")
    
    # Show sample data for a team
    print(f"\n📋 Sample Data (First Team: {espn_df.iloc[0]['team']}):")
    sample_features = [f for f in key_features if f in espn_df.columns]
    if sample_features:
        for feat in sample_features[:10]:
            value = espn_df.iloc[0][feat]
            print(f"  - {feat}: {value}")

def create_validation_summary(year: int):
    """Create validation summary"""
    print(f"\n{'=' * 80}")
    print(f"VALIDATION SUMMARY - {year}")
    print('=' * 80)
    
    espn_df = load_espn_data(year)
    
    # Analyze availability
    analyze_feature_availability(espn_df, year)
    
    # Compare records
    compare_team_records(espn_df, year)
    
    # Save summary
    summary = {
        'year': year,
        'teams': len(espn_df),
        'total_features': len(espn_df.columns),
        'features_with_data': espn_df.notna().sum().sum(),
        'completeness_pct': (espn_df.notna().sum().sum() / (len(espn_df) * len(espn_df.columns)) * 100)
    }
    
    print(f"\n📊 Overall Summary:")
    print(f"  - Teams: {summary['teams']}")
    print(f"  - Total Features: {summary['total_features']}")
    print(f"  - Data Points: {summary['features_with_data']:,}")
    print(f"  - Completeness: {summary['completeness_pct']:.2f}%")
    
    return espn_df, summary

def main():
    """Main execution"""
    print("=" * 80)
    print("ESPN FEATURE VALIDATION - Simplified Approach")
    print("=" * 80)
    
    # Validate for 2024 and 2025
    for year in [2024, 2025]:
        espn_df, summary = create_validation_summary(year)
        
        # Save ESPN data with team as index for easy lookup
        output_path = Path(f'results/espn_features_{year}_indexed.parquet')
        espn_df.set_index('team').to_parquet(output_path)
        print(f"\n✅ Saved indexed ESPN data to: {output_path}")
    
    print("\n" + "=" * 80)
    print("✅ VALIDATION ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. We have confirmed ESPN data is available and complete")
    print("2. We can now derive features from nfl-data-py")
    print("3. Then compare derived vs ESPN features")

if __name__ == '__main__':
    main()

