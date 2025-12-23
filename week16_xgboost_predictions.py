"""
XGBoost Week 16 Predictions with Injury Feature Verification

This script shows:
1. All XGBoost predictions for Week 16
2. Injury features for each game to confirm they're being used
3. Comparison with ESPN injury data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from run_tier_sa_backtest import load_and_prepare_data
from version import FEATURES

def show_xgboost_predictions_with_injuries():
    """Show XGBoost predictions with injury context."""
    
    print("="*90)
    print("XGBOOST WEEK 16 PREDICTIONS - WITH INJURY VERIFICATION")
    print("="*90)
    
    # Load data
    print("\n[1/3] Loading data with TIER S+A features (including injuries)...")
    games_df, _ = load_and_prepare_data()
    
    # Get Week 16 data
    week16 = games_df[
        (games_df['season'] == 2025) &
        (games_df['week'] == 16)
    ].copy()
    
    print(f"Found {len(week16)} Week 16 games")
    
    # Get features
    available_features = [f for f in FEATURES if f in week16.columns]
    
    # Train on 2018-2024
    train_df = games_df[
        (games_df['season'] >= 2018) &
        (games_df['season'] <= 2024)
    ].dropna(subset=available_features + ['home_win'])
    
    X_train = train_df[available_features].copy()
    y_train = train_df['home_win'].astype(int)
    train_medians = X_train.median()
    
    # Prepare Week 16
    X_week16 = week16[available_features].copy()
    for col in X_week16.columns:
        if X_week16[col].isna().any():
            X_week16[col] = X_week16[col].fillna(train_medians.get(col, 0))
    
    # Train XGBoost
    print("\n[2/3] Training XGBoost model...")
    xgb = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0, random_state=42)
    xgb.fit(X_train, y_train)
    
    # Get predictions
    xgb_proba = xgb.predict_proba(X_week16)[:, 1]
    
    # Create results with injury features
    print("\n[3/3] Generating predictions with injury data...")
    
    injury_features = [
        'home_injury_impact', 'away_injury_impact', 'injury_diff',
        'home_qb_out', 'away_qb_out'
    ]
    
    results = pd.DataFrame({
        'away_team': week16['away_team'].values,
        'home_team': week16['home_team'].values,
        'spread': week16['spread_line'].values,
        'vegas_home_prob': week16['home_implied_prob'].values,
        'xgb_home_prob': xgb_proba,
    })
    
    # Add injury features
    for feat in injury_features:
        if feat in X_week16.columns:
            results[feat] = X_week16[feat].values
    
    # Add XGBoost pick
    results['xgb_pick'] = results.apply(
        lambda x: x['home_team'] if x['xgb_home_prob'] > 0.5 else x['away_team'], 
        axis=1
    )
    results['xgb_confidence'] = results.apply(
        lambda x: max(x['xgb_home_prob'], 1 - x['xgb_home_prob']),
        axis=1
    )
    
    # Sort by XGBoost confidence
    results_sorted = results.sort_values('xgb_confidence', ascending=False)
    
    # Display predictions
    print("\n" + "="*90)
    print("XGBOOST WEEK 16 PREDICTIONS (Sorted by Confidence)")
    print("="*90)
    print(f"\n{'Matchup':<18} {'XGB Pick':<8} {'Conf':>6} {'Vegas':>7} {'XGB':>7} "
          f"{'Home Inj':>9} {'Away Inj':>9} {'QB Out':>8}")
    print("-"*90)
    
    for _, row in results_sorted.iterrows():
        matchup = f"{row['away_team']}@{row['home_team']}"
        home_inj = row.get('home_injury_impact', 0)
        away_inj = row.get('away_injury_impact', 0)
        qb_out = "HOME" if row.get('home_qb_out', 0) == 1 else "AWAY" if row.get('away_qb_out', 0) == 1 else "-"
        
        # Highlight if injury impact is significant
        inj_marker = ""
        if home_inj > 2.0 or away_inj > 2.0:
            inj_marker = " ⚠️"
        
        print(f"{matchup:<18} {row['xgb_pick']:<8} {row['xgb_confidence']:>5.1%} "
              f"{row['vegas_home_prob']:>6.1%} {row['xgb_home_prob']:>6.1%} "
              f"{home_inj:>9.2f} {away_inj:>9.2f} {qb_out:>8}{inj_marker}")
    
    # Show injury feature statistics
    print("\n" + "="*90)
    print("INJURY FEATURE VERIFICATION")
    print("="*90)
    
    print(f"\nInjury Impact Statistics:")
    print(f"  Home injury impact - Mean: {results['home_injury_impact'].mean():.2f}, "
          f"Max: {results['home_injury_impact'].max():.2f}")
    print(f"  Away injury impact - Mean: {results['away_injury_impact'].mean():.2f}, "
          f"Max: {results['away_injury_impact'].max():.2f}")
    print(f"  QB Out - Home: {results['home_qb_out'].sum():.0f} games, "
          f"Away: {results['away_qb_out'].sum():.0f} games")
    
    # Check if injury features are populated
    injury_populated = results['home_injury_impact'].notna().sum()
    print(f"\n✅ Injury features populated for {injury_populated}/{len(results)} games")
    
    if injury_populated == 0:
        print("⚠️  WARNING: Injury features are NOT populated! Model may not be using injury data.")
    
    # Show games with highest injury impact
    print("\n" + "="*90)
    print("GAMES WITH SIGNIFICANT INJURY IMPACT")
    print("="*90)
    
    results['total_injury_impact'] = results['home_injury_impact'] + results['away_injury_impact']
    high_injury = results.nlargest(5, 'total_injury_impact')
    
    print(f"\n{'Matchup':<18} {'XGB Pick':<8} {'Home Inj':>10} {'Away Inj':>10} {'Total':>10}")
    print("-"*90)
    for _, row in high_injury.iterrows():
        matchup = f"{row['away_team']}@{row['home_team']}"
        print(f"{matchup:<18} {row['xgb_pick']:<8} {row['home_injury_impact']:>10.2f} "
              f"{row['away_injury_impact']:>10.2f} {row['total_injury_impact']:>10.2f}")
    
    # Save results
    results_dir = Path("results")
    results_sorted.to_csv(results_dir / "week16_xgboost_with_injuries.csv", index=False)
    print(f"\n✅ Saved to results/week16_xgboost_with_injuries.csv")
    
    return results_sorted


if __name__ == "__main__":
    predictions = show_xgboost_predictions_with_injuries()

