"""
TIER S+A Feature Engineering
=============================
High-value features for NFL betting model:

TIER S (Highest Value):
- Rolling EPA (3-week, 5-week) - already in tier1
- CPOE from NGS Passing data
- Pressure Rate from PFR Passing data
- Rest Days from Schedule data
- Injury Impact Scoring (QB + key positions)

TIER A (High Value):
- NGS Metrics: avg_separation, avg_time_to_throw, rush_yards_over_expected
- Snap Count Workload indicators
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))

# Try nflreadpy first (newer, actively maintained), fallback to nfl-data-py
try:
    import nflreadpy as nfl_new
    HAS_NFLREADPY = True
except ImportError:
    HAS_NFLREADPY = False

try:
    import nfl_data_py as nfl
except ImportError:
    if not HAS_NFLREADPY:
        raise ImportError("Please install nflreadpy: pip install nflreadpy")


# =============================================================================
# TIER S: CPOE from NGS Passing
# =============================================================================

def compute_team_cpoe(years: list) -> pd.DataFrame:
    """
    Compute team-level CPOE (Completion % Over Expected) from NGS Passing data.
    Returns rolling 3-week CPOE for starting QB.
    """
    print(f"Loading NGS Passing data for CPOE ({min(years)}-{max(years)})...")
    
    # NGS data only available from 2016+
    valid_years = [y for y in years if y >= 2016]
    if not valid_years:
        return pd.DataFrame()
    
    try:
        ngs_pass = nfl.import_ngs_data('passing', valid_years)
    except Exception as e:
        print(f"Error loading NGS passing: {e}")
        return pd.DataFrame()
    
    if len(ngs_pass) == 0:
        return pd.DataFrame()
    
    # Get weekly data (not season aggregates)
    ngs_weekly = ngs_pass[ngs_pass['week'] > 0].copy()
    
    # Key columns
    cpoe_cols = ['player_gsis_id', 'player_display_name', 'team_abbr', 'season', 'week',
                 'completion_percentage_above_expectation', 'avg_time_to_throw',
                 'aggressiveness', 'attempts']
    
    cpoe_cols = [c for c in cpoe_cols if c in ngs_weekly.columns]
    ngs_weekly = ngs_weekly[cpoe_cols].copy()
    
    if 'completion_percentage_above_expectation' not in ngs_weekly.columns:
        return pd.DataFrame()
    
    # Rename for clarity
    ngs_weekly = ngs_weekly.rename(columns={
        'completion_percentage_above_expectation': 'cpoe',
        'avg_time_to_throw': 'time_to_throw',
        'team_abbr': 'team'
    })
    
    # Sort and compute rolling averages (3-week)
    ngs_weekly = ngs_weekly.sort_values(['team', 'season', 'week'])
    
    result = []
    for (team, season), group in ngs_weekly.groupby(['team', 'season']):
        group = group.sort_values('week').copy()
        # Shift to get prior week values (no look-ahead bias)
        group['cpoe_3wk'] = group['cpoe'].shift(1).rolling(3, min_periods=1).mean()
        group['time_to_throw_3wk'] = group['time_to_throw'].shift(1).rolling(3, min_periods=1).mean() if 'time_to_throw' in group.columns else np.nan
        result.append(group)
    
    if not result:
        return pd.DataFrame()
    
    ngs_result = pd.concat(result, ignore_index=True)
    return ngs_result[['team', 'season', 'week', 'cpoe_3wk', 'time_to_throw_3wk']].drop_duplicates()


# =============================================================================
# TIER S: Pressure Rate from PFR
# =============================================================================

def compute_team_pressure_rate(years: list) -> pd.DataFrame:
    """
    Compute team-level pressure rate from PFR Passing data.
    Returns rolling 3-week pressure rate for QBs.
    """
    print(f"Loading PFR Passing data for pressure rate ({min(years)}-{max(years)})...")
    
    # PFR data only available from 2018+
    valid_years = [y for y in years if y >= 2018]
    if not valid_years:
        return pd.DataFrame()
    
    try:
        pfr_pass = nfl.import_weekly_pfr('pass', valid_years)
    except Exception as e:
        print(f"Error loading PFR passing: {e}")
        return pd.DataFrame()
    
    if len(pfr_pass) == 0:
        return pd.DataFrame()
    
    # Key columns
    if 'times_pressured_pct' not in pfr_pass.columns:
        print("times_pressured_pct not in PFR data")
        return pd.DataFrame()
    
    pfr_pass = pfr_pass.rename(columns={'pfr_game_id': 'game_id'})
    
    # Get team-game level (aggregate if multiple QBs played)
    team_pressure = pfr_pass.groupby(['team', 'season', 'week']).agg({
        'times_pressured_pct': 'mean',
        'passing_bad_throw_pct': 'mean' if 'passing_bad_throw_pct' in pfr_pass.columns else 'first'
    }).reset_index()
    
    # Compute rolling averages
    team_pressure = team_pressure.sort_values(['team', 'season', 'week'])
    
    result = []
    for (team, season), group in team_pressure.groupby(['team', 'season']):
        group = group.sort_values('week').copy()
        group['pressure_rate_3wk'] = group['times_pressured_pct'].shift(1).rolling(3, min_periods=1).mean()
        if 'passing_bad_throw_pct' in group.columns:
            group['bad_throw_pct_3wk'] = group['passing_bad_throw_pct'].shift(1).rolling(3, min_periods=1).mean()
        result.append(group)
    
    if not result:
        return pd.DataFrame()

    pfr_result = pd.concat(result, ignore_index=True)
    keep_cols = ['team', 'season', 'week', 'pressure_rate_3wk']
    if 'bad_throw_pct_3wk' in pfr_result.columns:
        keep_cols.append('bad_throw_pct_3wk')
    return pfr_result[keep_cols].drop_duplicates()


# =============================================================================
# TIER S: Injury Impact Scoring
# =============================================================================

# Position impact weights (how much a position injury affects team performance)
POSITION_IMPACT = {
    'QB': 1.0,      # Most impactful
    'LT': 0.6,      # Left tackle protects blind side
    'RT': 0.5,
    'WR': 0.4,
    'RB': 0.35,
    'TE': 0.3,
    'G': 0.3,
    'C': 0.3,
    'CB': 0.4,
    'EDGE': 0.4,
    'DE': 0.35,
    'DT': 0.3,
    'LB': 0.3,
    'S': 0.3,
    'K': 0.2,
    'P': 0.1,
}

# Status impact (probability of missing game)
STATUS_IMPACT = {
    'Out': 1.0,
    'Doubtful': 0.75,
    'Questionable': 0.5,
    'Probable': 0.1,
    'IR': 1.0,
}

def compute_injury_impact(years: list) -> pd.DataFrame:
    """
    Compute team-level injury impact score from injury reports.
    Higher score = more impactful injuries.

    Uses nflreadpy (preferred) or nfl-data-py as fallback.
    nflreadpy API: load_injuries(seasons=None) returns Polars DataFrame

    Note: 2025 data may not be available yet - we load year by year to handle this.
    """
    print(f"Loading injury data ({min(years)}-{max(years)})...")

    # Injuries available from 2009+, exclude current year if data not ready
    valid_years = [y for y in years if y >= 2009 and y <= 2024]  # 2025 not available yet
    if not valid_years:
        return pd.DataFrame()

    all_injuries = []

    # Try loading each year individually to handle missing years gracefully
    for year in valid_years:
        injuries_year = None

        # Try nflreadpy first
        if HAS_NFLREADPY:
            try:
                injuries_polars = nfl_new.load_injuries(seasons=[year])
                injuries_year = injuries_polars.to_pandas()
            except Exception:
                pass

        # Fallback to nfl-data-py
        if injuries_year is None:
            try:
                injuries_year = nfl.import_injuries([year])
            except Exception:
                pass

        if injuries_year is not None and len(injuries_year) > 0:
            all_injuries.append(injuries_year)

    if not all_injuries:
        print("   No injury data available")
        return pd.DataFrame()

    injuries = pd.concat(all_injuries, ignore_index=True)
    print(f"   Loaded {len(injuries)} injury records for {len(all_injuries)} seasons")

    if len(injuries) == 0:
        return pd.DataFrame()

    # Map position to impact
    injuries['position_impact'] = injuries['position'].map(POSITION_IMPACT).fillna(0.2)

    # Map status to probability
    injuries['status_prob'] = injuries['report_status'].map(STATUS_IMPACT).fillna(0.25)

    # Compute injury impact per player
    injuries['player_impact'] = injuries['position_impact'] * injuries['status_prob']

    # Aggregate to team-week level
    team_injuries = injuries.groupby(['team', 'season', 'week']).agg({
        'player_impact': 'sum',
        'gsis_id': 'count'
    }).reset_index()
    team_injuries.columns = ['team', 'season', 'week', 'injury_impact', 'injured_count']

    # QB injury flag (most impactful)
    qb_injuries = injuries[injuries['position'] == 'QB'].copy()
    qb_out = qb_injuries[qb_injuries['report_status'].isin(['Out', 'Doubtful', 'IR'])]
    qb_out_agg = qb_out.groupby(['team', 'season', 'week']).size().reset_index(name='qb_out')

    team_injuries = team_injuries.merge(qb_out_agg, on=['team', 'season', 'week'], how='left')
    team_injuries['qb_out'] = team_injuries['qb_out'].fillna(0).astype(int)
    team_injuries['qb_out'] = (team_injuries['qb_out'] > 0).astype(int)

    return team_injuries


# =============================================================================
# TIER A: NGS Rushing (RYOE)
# =============================================================================

def compute_team_ryoe(years: list) -> pd.DataFrame:
    """
    Compute team-level Rush Yards Over Expected from NGS Rushing data.
    """
    print(f"Loading NGS Rushing data for RYOE ({min(years)}-{max(years)})...")

    valid_years = [y for y in years if y >= 2016]
    if not valid_years:
        return pd.DataFrame()

    try:
        ngs_rush = nfl.import_ngs_data('rushing', valid_years)
    except Exception as e:
        print(f"Error loading NGS rushing: {e}")
        return pd.DataFrame()

    if len(ngs_rush) == 0:
        return pd.DataFrame()

    # Get weekly data
    ngs_weekly = ngs_rush[ngs_rush['week'] > 0].copy()

    if 'rush_yards_over_expected_per_att' not in ngs_weekly.columns:
        if 'rush_yards_over_expected' in ngs_weekly.columns and 'rush_attempts' in ngs_weekly.columns:
            ngs_weekly['rush_yards_over_expected_per_att'] = ngs_weekly['rush_yards_over_expected'] / ngs_weekly['rush_attempts'].clip(1)
        else:
            return pd.DataFrame()

    ngs_weekly = ngs_weekly.rename(columns={'team_abbr': 'team'})

    # Aggregate to team-week
    team_ryoe = ngs_weekly.groupby(['team', 'season', 'week']).agg({
        'rush_yards_over_expected_per_att': 'mean',
        'efficiency': 'mean' if 'efficiency' in ngs_weekly.columns else 'first'
    }).reset_index()

    team_ryoe = team_ryoe.rename(columns={
        'rush_yards_over_expected_per_att': 'ryoe_per_att'
    })

    # Compute rolling averages
    team_ryoe = team_ryoe.sort_values(['team', 'season', 'week'])

    result = []
    for (team, season), group in team_ryoe.groupby(['team', 'season']):
        group = group.sort_values('week').copy()
        group['ryoe_3wk'] = group['ryoe_per_att'].shift(1).rolling(3, min_periods=1).mean()
        result.append(group)

    if not result:
        return pd.DataFrame()

    ryoe_result = pd.concat(result, ignore_index=True)
    return ryoe_result[['team', 'season', 'week', 'ryoe_3wk']].drop_duplicates()


# =============================================================================
# TIER A: NGS Receiving (Separation)
# =============================================================================

def compute_team_separation(years: list) -> pd.DataFrame:
    """
    Compute team-level receiver separation from NGS Receiving data.
    """
    print(f"Loading NGS Receiving data for separation ({min(years)}-{max(years)})...")

    valid_years = [y for y in years if y >= 2016]
    if not valid_years:
        return pd.DataFrame()

    try:
        ngs_rec = nfl.import_ngs_data('receiving', valid_years)
    except Exception as e:
        print(f"Error loading NGS receiving: {e}")
        return pd.DataFrame()

    if len(ngs_rec) == 0:
        return pd.DataFrame()

    # Get weekly data
    ngs_weekly = ngs_rec[ngs_rec['week'] > 0].copy()

    if 'avg_separation' not in ngs_weekly.columns:
        return pd.DataFrame()

    ngs_weekly = ngs_weekly.rename(columns={'team_abbr': 'team'})

    # Aggregate to team-week (weighted by targets)
    if 'targets' in ngs_weekly.columns:
        team_sep = ngs_weekly.groupby(['team', 'season', 'week']).apply(
            lambda x: np.average(x['avg_separation'], weights=x['targets'].clip(1))
        ).reset_index(name='avg_separation')
    else:
        team_sep = ngs_weekly.groupby(['team', 'season', 'week'])['avg_separation'].mean().reset_index()

    # Compute rolling averages
    team_sep = team_sep.sort_values(['team', 'season', 'week'])

    result = []
    for (team, season), group in team_sep.groupby(['team', 'season']):
        group = group.sort_values('week').copy()
        group['separation_3wk'] = group['avg_separation'].shift(1).rolling(3, min_periods=1).mean()
        result.append(group)

    if not result:
        return pd.DataFrame()

    sep_result = pd.concat(result, ignore_index=True)
    return sep_result[['team', 'season', 'week', 'separation_3wk']].drop_duplicates()


# =============================================================================
# MASTER FUNCTION: Compute All TIER S+A Features
# =============================================================================

def compute_all_tier_sa_features(years: list) -> Dict[str, pd.DataFrame]:
    """
    Compute all TIER S+A features and return as dictionary of DataFrames.
    Each DataFrame is keyed by team-season-week.
    """
    features = {}

    # TIER S Features
    print("\n" + "="*60)
    print("COMPUTING TIER S FEATURES")
    print("="*60)

    features['cpoe'] = compute_team_cpoe(years)
    features['pressure'] = compute_team_pressure_rate(years)
    features['injuries'] = compute_injury_impact(years)

    # TIER A Features
    print("\n" + "="*60)
    print("COMPUTING TIER A FEATURES")
    print("="*60)

    features['ryoe'] = compute_team_ryoe(years)
    features['separation'] = compute_team_separation(years)

    # Summary
    print("\n" + "="*60)
    print("TIER S+A FEATURE SUMMARY")
    print("="*60)
    for name, df in features.items():
        if len(df) > 0:
            print(f"  {name}: {len(df)} records, {df.columns.tolist()}")
        else:
            print(f"  {name}: NO DATA")

    return features


def merge_features_to_games(games: pd.DataFrame, features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all TIER S+A features onto games dataframe.
    Matches home_team and away_team to respective features.
    """
    df = games.copy()

    for name, feat_df in features.items():
        if len(feat_df) == 0:
            continue

        # Merge for home team
        home_cols = {c: f'home_{c}' for c in feat_df.columns if c not in ['team', 'season', 'week']}
        home_feat = feat_df.rename(columns={'team': 'home_team', **home_cols})
        df = df.merge(home_feat, on=['home_team', 'season', 'week'], how='left')

        # Merge for away team
        away_cols = {c: f'away_{c}' for c in feat_df.columns if c not in ['team', 'season', 'week']}
        away_feat = feat_df.rename(columns={'team': 'away_team', **away_cols})
        df = df.merge(away_feat, on=['away_team', 'season', 'week'], how='left')

    # Compute differentials
    diff_pairs = [
        ('home_cpoe_3wk', 'away_cpoe_3wk', 'cpoe_diff'),
        ('home_pressure_rate_3wk', 'away_pressure_rate_3wk', 'pressure_diff'),
        ('home_injury_impact', 'away_injury_impact', 'injury_diff'),
        ('home_ryoe_3wk', 'away_ryoe_3wk', 'ryoe_diff'),
        ('home_separation_3wk', 'away_separation_3wk', 'separation_diff'),
    ]

    for home_col, away_col, diff_col in diff_pairs:
        if home_col in df.columns and away_col in df.columns:
            df[diff_col] = df[home_col] - df[away_col]

    return df


def add_rest_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest day advantage features from schedule data."""
    df = df.copy()

    # Rest advantage (already computed in data_loader but ensure it exists)
    if 'rest_advantage' not in df.columns:
        if 'home_rest' in df.columns and 'away_rest' in df.columns:
            df['rest_advantage'] = df['home_rest'] - df['away_rest']

    # Short week flags
    if 'home_rest' in df.columns:
        df['home_short_week'] = (df['home_rest'] <= 6).astype(int)
    if 'away_rest' in df.columns:
        df['away_short_week'] = (df['away_rest'] <= 6).astype(int)

    # Bye week advantage
    if 'home_rest' in df.columns:
        df['home_off_bye'] = (df['home_rest'] >= 10).astype(int)
    if 'away_rest' in df.columns:
        df['away_off_bye'] = (df['away_rest'] >= 10).astype(int)

    return df


if __name__ == "__main__":
    # Test the feature computation
    years = [2023, 2024]
    features = compute_all_tier_sa_features(years)

    # Show sample
    for name, df in features.items():
        if len(df) > 0:
            print(f"\n{name} sample:")
            print(df.head())

