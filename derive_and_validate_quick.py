"""
Quick derivation and validation for key features
This script focuses on validating a subset of critical features first
"""
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_or_cache_data(year: int):
    """Load nfl-data-py data with caching"""
    cache_dir = Path('data/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    pbp_cache = cache_dir / f'pbp_{year}.parquet'
    sched_cache = cache_dir / f'schedules_{year}.parquet'
    
    # Load or cache play-by-play
    if pbp_cache.exists():
        print(f"  Loading cached play-by-play for {year}...")
        pbp = pd.read_parquet(pbp_cache)
    else:
        print(f"  Downloading play-by-play for {year}...")
        pbp = nfl.import_pbp_data([year])
        pbp.to_parquet(pbp_cache)
    
    # Load or cache schedules
    if sched_cache.exists():
        print(f"  Loading cached schedules for {year}...")
        schedules = pd.read_parquet(sched_cache)
    else:
        print(f"  Downloading schedules for {year}...")
        schedules = nfl.import_schedules([year])
        schedules.to_parquet(sched_cache)
    
    print(f"  ✅ Loaded {len(pbp):,} plays, {len(schedules)} games")
    return pbp, schedules

def derive_key_features(team: str, year: int, pbp: pd.DataFrame, schedules: pd.DataFrame) -> dict:
    """Derive key features for validation"""
    # Filter to team's plays - REGULAR SEASON ONLY (weeks 1-18 for 2024)
    team_pbp = pbp[(pbp['posteam'] == team) & (pbp['season'] == year) & (pbp['week'] <= 18)]
    pass_plays = team_pbp[team_pbp['play_type'] == 'pass']
    rush_plays = team_pbp[team_pbp['play_type'] == 'run']

    features = {'team': team, 'season': year}

    # Passing features
    # ESPN includes sack yards in passing yards, so we need to add them back
    sack_yards = team_pbp[team_pbp['sack'] == 1]['yards_gained'].sum()  # This is negative
    features['passing_passingYards'] = pass_plays['yards_gained'].sum() + sack_yards
    features['passing_passingAttempts'] = pass_plays['pass_attempt'].sum()
    features['passing_completions'] = pass_plays['complete_pass'].sum()
    features['passing_passingTouchdowns'] = pass_plays['pass_touchdown'].sum()
    features['passing_interceptions'] = pass_plays['interception'].sum()

    if features['passing_passingAttempts'] > 0:
        features['passing_completionPct'] = (
            features['passing_completions'] / features['passing_passingAttempts'] * 100
        )
    else:
        features['passing_completionPct'] = 0

    # Rushing features
    features['rushing_rushingYards'] = rush_plays['yards_gained'].sum()
    features['rushing_rushingAttempts'] = rush_plays['rush_attempt'].sum()
    features['rushing_rushingTouchdowns'] = rush_plays['rush_touchdown'].sum()
    
    # Team records
    team_games = schedules[
        ((schedules['home_team'] == team) | (schedules['away_team'] == team)) &
        (schedules['season'] == year) &
        (schedules['game_type'] == 'REG')
    ]
    
    home_games = team_games[team_games['home_team'] == team]
    away_games = team_games[team_games['away_team'] == team]
    
    home_wins = len(home_games[home_games['home_score'] > home_games['away_score']])
    away_wins = len(away_games[away_games['away_score'] > away_games['home_score']])
    
    features['total_wins'] = home_wins + away_wins
    
    home_points = home_games['home_score'].sum()
    away_points = away_games['away_score'].sum()
    features['total_pointsFor'] = home_points + away_points
    
    return features

def validate_features(year: int):
    """Validate derived features against ESPN"""
    print(f"\n{'=' * 80}")
    print(f"VALIDATION FOR {year}")
    print('=' * 80)
    
    # Load ESPN data
    print("\nLoading ESPN data...")
    espn_stats = pd.read_parquet(f'data/espn_raw/team_stats_{year}.parquet')
    espn_records = pd.read_parquet(f'data/espn_raw/team_records_{year}.parquet')
    espn = pd.merge(espn_stats, espn_records, on='team', how='outer')
    espn = espn.set_index('team')
    
    # Load nfl-data-py data
    print("\nLoading nfl-data-py data...")
    pbp, schedules = load_or_cache_data(year)
    
    # Get teams
    teams = sorted(espn.index.tolist())
    
    # Derive features for all teams
    print(f"\nDeriving features for {len(teams)} teams...")
    derived_list = []
    for i, team in enumerate(teams, 1):
        print(f"  [{i}/{len(teams)}] {team}...", end=' ')
        try:
            features = derive_key_features(team, year, pbp, schedules)
            derived_list.append(features)
            print("✅")
        except Exception as e:
            print(f"❌ {e}")
    
    derived = pd.DataFrame(derived_list).set_index('team')
    
    # Calculate correlations
    print(f"\n{'=' * 80}")
    print("CORRELATION ANALYSIS")
    print('=' * 80)
    
    key_features = [
        'passing_passingYards',
        'passing_passingAttempts',
        'passing_completions',
        'passing_passingTouchdowns',
        'passing_interceptions',
        'passing_completionPct',
        'rushing_rushingYards',
        'rushing_rushingAttempts',
        'rushing_rushingTouchdowns',
        'total_wins',
        'total_pointsFor'
    ]
    
    results = []
    for feature in key_features:
        if feature in espn.columns and feature in derived.columns:
            # Get common teams
            common_teams = espn.index.intersection(derived.index)
            espn_vals = espn.loc[common_teams, feature].values
            derived_vals = derived.loc[common_teams, feature].values
            
            # Remove NaN
            mask = ~(pd.isna(espn_vals) | pd.isna(derived_vals))
            if mask.sum() > 1:
                r, p = pearsonr(espn_vals[mask], derived_vals[mask])
                results.append({
                    'feature': feature,
                    'correlation': r,
                    'p_value': p,
                    'n_teams': mask.sum()
                })
                
                status = "✅" if r > 0.95 else "⚠️" if r > 0.85 else "❌"
                print(f"{status} {feature:35s} r = {r:.4f} (p = {p:.4e}, n = {mask.sum()})")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = Path(f'results/validation_results_{year}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved validation results to: {output_path}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print('=' * 80)
    high_corr = len(results_df[results_df['correlation'] > 0.95])
    med_corr = len(results_df[(results_df['correlation'] > 0.85) & (results_df['correlation'] <= 0.95)])
    low_corr = len(results_df[results_df['correlation'] <= 0.85])
    
    print(f"  High correlation (r > 0.95): {high_corr}/{len(results_df)} features")
    print(f"  Medium correlation (r > 0.85): {med_corr}/{len(results_df)} features")
    print(f"  Low correlation (r ≤ 0.85): {low_corr}/{len(results_df)} features")
    
    return results_df

def main():
    """Main execution"""
    print("=" * 80)
    print("QUICK FEATURE DERIVATION & VALIDATION")
    print("=" * 80)
    
    # Validate for 2024
    results_2024 = validate_features(2024)
    
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    main()

