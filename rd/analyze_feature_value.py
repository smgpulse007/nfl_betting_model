"""
Analyze which features have the highest predictive value for betting.
This script computes correlations and feature importance for key metrics.
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def compute_team_game_stats(pbp, schedules):
    """Aggregate play-by-play data to team-game level."""
    # Filter to actual plays
    plays = pbp[pbp['play'] == 1].copy()
    
    # Home team stats
    home_stats = plays[plays['posteam'] == plays['home_team']].groupby('game_id').agg({
        'epa': 'sum',
        'success': 'mean',
        'pass': 'mean',
        'cpoe': 'mean',
    }).rename(columns={
        'epa': 'home_off_epa',
        'success': 'home_success_rate',
        'pass': 'home_pass_rate',
        'cpoe': 'home_cpoe'
    })
    
    # Away team stats
    away_stats = plays[plays['posteam'] == plays['away_team']].groupby('game_id').agg({
        'epa': 'sum',
        'success': 'mean',
        'pass': 'mean',
        'cpoe': 'mean',
    }).rename(columns={
        'epa': 'away_off_epa',
        'success': 'away_success_rate',
        'pass': 'away_pass_rate',
        'cpoe': 'away_cpoe'
    })
    
    # Merge with schedules
    df = schedules.merge(home_stats, on='game_id', how='left')
    df = df.merge(away_stats, on='game_id', how='left')
    
    # Compute differentials
    df['epa_diff'] = df['home_off_epa'] - df['away_off_epa']
    df['cpoe_diff'] = df['home_cpoe'] - df['away_cpoe']
    df['success_diff'] = df['home_success_rate'] - df['away_success_rate']
    
    return df

def analyze_correlations(df):
    """Analyze correlations between features and outcomes."""
    print("\n" + "="*70)
    print("FEATURE CORRELATIONS WITH GAME OUTCOMES")
    print("="*70)
    
    # Define features and targets
    features = [
        'spread_line', 'total_line', 'home_rest', 'away_rest',
        'home_off_epa', 'away_off_epa', 'epa_diff',
        'home_cpoe', 'away_cpoe', 'cpoe_diff',
        'home_success_rate', 'away_success_rate', 'success_diff'
    ]
    
    targets = ['result', 'total']
    
    print("\n📊 Correlation with RESULT (home margin):")
    print("-" * 50)
    correlations = []
    for feat in features:
        if feat in df.columns:
            valid = df[[feat, 'result']].dropna()
            if len(valid) > 10:
                corr, pval = stats.pearsonr(valid[feat], valid['result'])
                correlations.append((feat, corr, pval))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, corr, pval in correlations:
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {feat:<30} r={corr:+.3f} {sig}")
    
    print("\n📊 Correlation with TOTAL (combined score):")
    print("-" * 50)
    correlations = []
    for feat in features:
        if feat in df.columns:
            valid = df[[feat, 'total']].dropna()
            if len(valid) > 10:
                corr, pval = stats.pearsonr(valid[feat], valid['total'])
                correlations.append((feat, corr, pval))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, corr, pval in correlations:
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {feat:<30} r={corr:+.3f} {sig}")

def analyze_ngs_value(years=[2023, 2024]):
    """Analyze Next Gen Stats predictive value."""
    print("\n" + "="*70)
    print("NEXT GEN STATS ANALYSIS")
    print("="*70)
    
    # Load NGS data
    ngs_pass = nfl.import_ngs_data('passing', years)
    ngs_rush = nfl.import_ngs_data('rushing', years)
    ngs_rec = nfl.import_ngs_data('receiving', years)
    
    print(f"\n📊 NGS Passing: {len(ngs_pass)} records")
    print(f"   Key metrics: avg_time_to_throw, aggressiveness, CPOE")
    print(f"   CPOE range: {ngs_pass['completion_percentage_above_expectation'].min():.1f} to {ngs_pass['completion_percentage_above_expectation'].max():.1f}")
    
    print(f"\n📊 NGS Rushing: {len(ngs_rush)} records")
    print(f"   Key metrics: rush_yards_over_expected, efficiency")
    print(f"   RYOE range: {ngs_rush['rush_yards_over_expected'].min():.1f} to {ngs_rush['rush_yards_over_expected'].max():.1f}")
    
    print(f"\n📊 NGS Receiving: {len(ngs_rec)} records")
    print(f"   Key metrics: avg_separation, avg_yac_above_expectation")
    print(f"   Separation range: {ngs_rec['avg_separation'].min():.1f} to {ngs_rec['avg_separation'].max():.1f}")

def analyze_injury_impact(years=[2024]):
    """Analyze injury data availability and potential impact."""
    print("\n" + "="*70)
    print("INJURY DATA ANALYSIS")
    print("="*70)
    
    injuries = nfl.import_injuries(years)
    
    print(f"\n📊 Total injury records: {len(injuries):,}")
    print(f"\n   By Status:")
    print(injuries['report_status'].value_counts().to_string())
    print(f"\n   By Position:")
    print(injuries['position'].value_counts().head(10).to_string())
    
    # QB injuries are most impactful
    qb_injuries = injuries[injuries['position'] == 'QB']
    print(f"\n   QB Injuries: {len(qb_injuries)} records")
    print(f"   QB Out: {len(qb_injuries[qb_injuries['report_status'] == 'Out'])}")

def main():
    print("="*70)
    print("FEATURE VALUE ANALYSIS FOR NFL BETTING")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    years = [2023, 2024]
    pbp = nfl.import_pbp_data(years, downcast=True)
    schedules = nfl.import_schedules(years)
    
    # Filter to completed games
    schedules = schedules[schedules['result'].notna()].copy()
    print(f"Loaded {len(schedules)} completed games")
    
    # Compute team-game stats
    df = compute_team_game_stats(pbp, schedules)
    
    # Analyze correlations
    analyze_correlations(df)
    
    # Analyze NGS
    analyze_ngs_value(years)
    
    # Analyze injuries
    analyze_injury_impact([2024])
    
    print("\n" + "="*70)
    print("SUMMARY: TOP FEATURES FOR BETTING MODELS")
    print("="*70)
    print("""
    1. EPA (Expected Points Added) - Strongest predictor of margin
    2. CPOE (Completion % Over Expected) - Best QB metric
    3. Rest Days - Significant edge for well-rested teams
    4. Vegas Lines - Market efficiency baseline
    5. Injuries (especially QB) - 3-7 point swing potential
    6. NGS Metrics - Separation, RYOE, Time to Throw
    7. Pressure Rate - O-line vs D-line matchup
    """)

if __name__ == "__main__":
    main()

