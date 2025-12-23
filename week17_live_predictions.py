"""
Week 17 Live Predictions with Latest Data

Pulls:
1. Latest 2025 schedule (Week 17)
2. Live ESPN odds
3. Live ESPN injuries
4. Trains XGBoost on all data including Week 16 results
5. Generates predictions for Week 17
6. Deep dive on SF vs IND
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

import nfl_data_py as nfl
from xgboost import XGBClassifier
from run_tier_sa_backtest import load_and_prepare_data
from version import FEATURES
from espn_data_fetcher import ESPNDataFetcher

def get_live_week17_data():
    """Fetch latest Week 17 data from all sources."""
    print("="*100)
    print("FETCHING LIVE WEEK 17 DATA")
    print("="*100)
    
    # 1. Get latest schedule
    print("\n[1/3] Fetching latest 2025 schedule...")
    schedule_2025 = nfl.import_schedules([2025])
    week17 = schedule_2025[schedule_2025['week'] == 17].copy()
    print(f"Found {len(week17)} Week 17 games")
    
    # Show game status
    completed = week17[week17['home_score'].notna()]
    upcoming = week17[week17['home_score'].isna()]
    print(f"  Completed: {len(completed)}")
    print(f"  Upcoming: {len(upcoming)}")
    
    # 2. Get live ESPN odds
    print("\n[2/3] Fetching live ESPN odds...")
    fetcher = ESPNDataFetcher()
    try:
        live_odds = fetcher.get_current_odds()
        print(f"  Fetched odds for {len(live_odds)} games")
        
        # Merge with schedule
        week17 = week17.merge(
            live_odds[['home_team', 'away_team', 'spread', 'home_ml', 'away_ml']],
            on=['home_team', 'away_team'],
            how='left',
            suffixes=('', '_live')
        )
        
        # Update with live odds where available
        week17['spread_line'] = week17['spread'].fillna(week17['spread_line'])
        week17['home_moneyline'] = week17['home_ml'].fillna(week17['home_moneyline'])
        week17['away_moneyline'] = week17['away_ml'].fillna(week17['away_moneyline'])
        
    except Exception as e:
        print(f"  Warning: Could not fetch live odds: {e}")
    
    # 3. Get live ESPN injuries
    print("\n[3/3] Fetching live ESPN injuries...")
    try:
        live_injuries = fetcher.get_all_injuries()
        print(f"  Fetched injuries for {len(live_injuries['team'].unique())} teams")
    except Exception as e:
        print(f"  Warning: Could not fetch live injuries: {e}")
        live_injuries = pd.DataFrame()
    
    return week17, live_injuries


def train_xgboost_with_week16():
    """Train XGBoost on all data including Week 16 results."""
    print("\n" + "="*100)
    print("TRAINING XGBOOST WITH WEEK 16 RESULTS")
    print("="*100)
    
    # Load all data
    games_df, _ = load_and_prepare_data()
    
    # Get features
    available_features = [f for f in FEATURES if f in games_df.columns]
    
    # Train on 2018-2024 + Week 16 2025
    train_df = games_df[
        ((games_df['season'] >= 2018) & (games_df['season'] <= 2024)) |
        ((games_df['season'] == 2025) & (games_df['week'] == 16))
    ].dropna(subset=available_features + ['home_win'])
    
    print(f"\nTraining data: {len(train_df)} games")
    print(f"  2018-2024: {len(train_df[train_df['season'] <= 2024])} games")
    print(f"  Week 16 2025: {len(train_df[(train_df['season'] == 2025) & (train_df['week'] == 16)])} games")
    
    X_train = train_df[available_features].copy()
    y_train = train_df['home_win'].astype(int)
    
    # Train XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0, random_state=42)
    xgb.fit(X_train, y_train)
    
    print(f"\n✅ XGBoost trained on {len(train_df)} games")
    
    return xgb, available_features, X_train.median()


def predict_week17(xgb, features, train_medians):
    """Generate Week 17 predictions."""
    print("\n" + "="*100)
    print("GENERATING WEEK 17 PREDICTIONS")
    print("="*100)
    
    # Load data
    games_df, _ = load_and_prepare_data()
    week17 = games_df[(games_df['season'] == 2025) & (games_df['week'] == 17)].copy()
    
    # Prepare features
    X_week17 = week17[features].copy()
    for col in X_week17.columns:
        if X_week17[col].isna().any():
            X_week17[col] = X_week17[col].fillna(train_medians.get(col, 0))
    
    # Get predictions
    xgb_proba = xgb.predict_proba(X_week17)[:, 1]
    
    # Create results
    results = pd.DataFrame({
        'away_team': week17['away_team'].values,
        'home_team': week17['home_team'].values,
        'gameday': week17['gameday'].values,
        'spread': week17['spread_line'].values,
        'vegas_home_prob': week17['home_implied_prob'].values,
        'xgb_home_prob': xgb_proba,
    })
    
    results['xgb_pick'] = results.apply(
        lambda x: x['home_team'] if x['xgb_home_prob'] > 0.5 else x['away_team'],
        axis=1
    )
    results['xgb_confidence'] = results.apply(
        lambda x: max(x['xgb_home_prob'], 1 - x['xgb_home_prob']),
        axis=1
    )
    results['xgb_vegas_deviation'] = results['xgb_home_prob'] - results['vegas_home_prob']
    
    return results, week17, X_week17


def main():
    print(f"\n{'='*100}")
    print(f"WEEK 17 LIVE PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")
    
    # Get live data
    week17_schedule, live_injuries = get_live_week17_data()
    
    # Train model
    xgb, features, train_medians = train_xgboost_with_week16()
    
    # Generate predictions
    predictions, week17_data, X_week17 = predict_week17(xgb, features, train_medians)
    
    # Display predictions
    print("\n" + "="*100)
    print("WEEK 17 PREDICTIONS (Sorted by Confidence)")
    print("="*100)
    print(f"\n{'Matchup':<18} {'Date':<12} {'XGB Pick':<8} {'Conf':>6} {'Vegas':>7} {'XGB':>7} {'Dev':>7} {'Spread':>8}")
    print("-"*100)
    
    for _, row in predictions.sort_values('xgb_confidence', ascending=False).iterrows():
        matchup = f"{row['away_team']}@{row['home_team']}"
        date = pd.to_datetime(row['gameday']).strftime('%a %m/%d')
        
        marker = ""
        if abs(row['xgb_vegas_deviation']) > 0.15:
            marker = " 🔥🔥"
        elif abs(row['xgb_vegas_deviation']) > 0.10:
            marker = " 🔥"
        
        print(f"{matchup:<18} {date:<12} {row['xgb_pick']:<8} {row['xgb_confidence']:>5.1%} "
              f"{row['vegas_home_prob']:>6.1%} {row['xgb_home_prob']:>6.1%} "
              f"{row['xgb_vegas_deviation']:>+6.1%} {row['spread']:>8.1f}{marker}")
    
    # Save results
    results_dir = Path("results")
    predictions.to_csv(results_dir / "week17_predictions.csv", index=False)
    print(f"\n✅ Saved predictions to results/week17_predictions.csv")
    
    return predictions, week17_data, X_week17, live_injuries, xgb, features


if __name__ == "__main__":
    predictions, week17_data, X_week17, live_injuries, xgb, features = main()

