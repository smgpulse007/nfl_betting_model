"""
Diagnostic Analysis of SF @ IND Prediction

Shows exactly what data was used and recalculates with correct odds.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from run_tier_sa_backtest import load_and_prepare_data
from version import FEATURES

def diagnose_sf_ind():
    print("="*100)
    print("SF @ IND DIAGNOSTIC ANALYSIS")
    print("="*100)
    
    # Load data
    print("\n[1/5] Loading data...")
    games_df, _ = load_and_prepare_data()
    
    # Find SF @ IND game
    sf_ind = games_df[
        ((games_df['home_team'] == 'IND') & (games_df['away_team'] == 'SF') & (games_df['season'] == 2025)) |
        ((games_df['home_team'] == 'SF') & (games_df['away_team'] == 'IND') & (games_df['season'] == 2025))
    ]
    
    if len(sf_ind) == 0:
        print("❌ No SF @ IND game found!")
        return
    
    game = sf_ind.iloc[0]
    
    print(f"\n[2/5] Game Details from NFL Data:")
    print(f"  Week: {game['week']}")
    print(f"  Home: {game['home_team']}")
    print(f"  Away: {game['away_team']}")
    print(f"  Spread (from nfl_data_py): {game['home_team']} {game['spread_line']:+.1f}")
    print(f"  Total: {game['total_line']:.1f}")
    print(f"  Home ML: {game.get('home_moneyline', 'N/A')}")
    print(f"  Away ML: {game.get('away_moneyline', 'N/A')}")
    
    print(f"\n⚠️  ACTUAL LIVE ODDS (from user):")
    print(f"  Spread: SF -3.5 (SF is FAVORITE)")
    print(f"  SF Moneyline: 0.69 (implies ~-145 odds, 59% probability)")
    print(f"  O/U: 45.5")
    
    print(f"\n🔥 DISCREPANCY DETECTED!")
    if game['home_team'] == 'IND':
        print(f"  NFL Data says: IND {game['spread_line']:+.1f} (IND favorite)")
        print(f"  Live odds say: SF -3.5 (SF favorite)")
        print(f"  This is COMPLETELY BACKWARDS!")
    
    # Get features
    available_features = [f for f in FEATURES if f in games_df.columns]
    
    print(f"\n[3/5] Features used in prediction ({len(available_features)} total):")
    print("-"*100)
    
    # Train model
    train_df = games_df[
        (games_df['season'] >= 2018) & (games_df['season'] <= 2024)
    ].dropna(subset=available_features + ['home_win'])
    
    X_train = train_df[available_features].copy()
    y_train = train_df['home_win'].astype(int)
    train_medians = X_train.median()
    
    # Get SF @ IND features
    X_game = game[available_features].copy()
    
    # Fill missing values
    for col in available_features:
        if pd.isna(X_game[col]):
            X_game[col] = train_medians.get(col, 0)
    
    # Display features in categories
    print("\n📊 VEGAS/ODDS FEATURES:")
    odds_features = ['spread_line', 'total_line', 'home_implied_prob', 'away_implied_prob', 
                     'home_moneyline', 'away_moneyline']
    for feat in odds_features:
        if feat in available_features:
            val = X_game[feat]
            median = train_medians.get(feat, 0)
            diff = val - median
            print(f"  {feat:<25} = {val:>8.2f}  (median: {median:>6.2f}, diff: {diff:>+7.2f})")
    
    print("\n📊 ELO FEATURES:")
    elo_features = [f for f in available_features if 'elo' in f.lower()]
    for feat in elo_features:
        val = X_game[feat]
        median = train_medians.get(feat, 0)
        diff = val - median
        print(f"  {feat:<25} = {val:>8.2f}  (median: {median:>6.2f}, diff: {diff:>+7.2f})")
    
    print("\n📊 REST/VENUE FEATURES:")
    rest_features = ['rest_advantage', 'home_rest', 'away_rest', 'home_short_week', 'away_short_week']
    for feat in rest_features:
        if feat in available_features:
            val = X_game[feat]
            median = train_medians.get(feat, 0)
            diff = val - median
            print(f"  {feat:<25} = {val:>8.2f}  (median: {median:>6.2f}, diff: {diff:>+7.2f})")
    
    print("\n📊 INJURY FEATURES:")
    injury_features = [f for f in available_features if 'injury' in f or 'qb_out' in f]
    for feat in injury_features:
        val = X_game[feat]
        median = train_medians.get(feat, 0)
        diff = val - median
        marker = " ⚠️" if abs(diff) > 1.0 else ""
        print(f"  {feat:<25} = {val:>8.2f}  (median: {median:>6.2f}, diff: {diff:>+7.2f}){marker}")
    
    print("\n📊 PASSING FEATURES (TIER S):")
    passing_features = [f for f in available_features if 'cpoe' in f or 'pressure' in f or 'time_to_throw' in f]
    for feat in passing_features:
        val = X_game[feat]
        median = train_medians.get(feat, 0)
        diff = val - median
        print(f"  {feat:<25} = {val:>8.2f}  (median: {median:>6.2f}, diff: {diff:>+7.2f})")
    
    print("\n📊 RUSHING/RECEIVING FEATURES (TIER A):")
    rush_features = [f for f in available_features if 'ryoe' in f or 'separation' in f]
    for feat in rush_features:
        val = X_game[feat]
        median = train_medians.get(feat, 0)
        diff = val - median
        print(f"  {feat:<25} = {val:>8.2f}  (median: {median:>6.2f}, diff: {diff:>+7.2f})")
    
    # Train XGBoost
    print(f"\n[4/5] Training XGBoost and making prediction with LOADED data...")
    xgb = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0, random_state=42)
    xgb.fit(X_train, y_train)
    
    # Predict with loaded data
    X_game_array = X_game.values.reshape(1, -1)
    prob_loaded = xgb.predict_proba(X_game_array)[0, 1]
    
    print(f"\n  Prediction with LOADED data:")
    print(f"    Home ({game['home_team']}) win probability: {prob_loaded:.1%}")
    print(f"    Away ({game['away_team']}) win probability: {1-prob_loaded:.1%}")
    
    # Now recalculate with CORRECT odds
    print(f"\n[5/5] Recalculating with CORRECT live odds...")
    
    X_corrected = X_game.copy()
    
    # Correct the odds based on user input
    # SF -3.5 means SF is favorite, so if IND is home, spread should be +3.5
    if game['home_team'] == 'IND':
        X_corrected['spread_line'] = 3.5  # IND getting 3.5 points
        # SF ML 0.69 implies SF ~59% probability
        X_corrected['home_implied_prob'] = 0.41  # IND ~41%
        X_corrected['away_implied_prob'] = 0.59  # SF ~59%
    else:  # SF is home
        X_corrected['spread_line'] = -3.5  # SF giving 3.5 points
        X_corrected['home_implied_prob'] = 0.59  # SF ~59%
        X_corrected['away_implied_prob'] = 0.41  # IND ~41%
    
    X_corrected['total_line'] = 45.5
    
    print(f"\n  Corrected odds:")
    print(f"    Spread: {game['home_team']} {X_corrected['spread_line']:+.1f}")
    print(f"    Home implied prob: {X_corrected['home_implied_prob']:.1%}")
    print(f"    Away implied prob: {X_corrected['away_implied_prob']:.1%}")
    print(f"    Total: {X_corrected['total_line']:.1f}")
    
    # Predict with corrected data
    X_corrected_array = X_corrected.values.reshape(1, -1)
    prob_corrected = xgb.predict_proba(X_corrected_array)[0, 1]
    
    print(f"\n  Prediction with CORRECTED data:")
    print(f"    Home ({game['home_team']}) win probability: {prob_corrected:.1%}")
    print(f"    Away ({game['away_team']}) win probability: {1-prob_corrected:.1%}")
    
    # Show the difference
    print(f"\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"\nWith LOADED data (IND -4.5):")
    print(f"  {game['away_team']} win probability: {1-prob_loaded:.1%}")
    print(f"\nWith CORRECTED data (SF -3.5):")
    print(f"  {game['away_team']} win probability: {1-prob_corrected:.1%}")
    print(f"\nDifference: {abs((1-prob_corrected) - (1-prob_loaded)):.1%}")
    
    # Feature importance
    print(f"\n📊 TOP 10 MOST IMPORTANT FEATURES:")
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    for _, row in feature_importance.iterrows():
        feat = row['feature']
        imp = row['importance']
        val = X_game[feat]
        print(f"  {feat:<25} importance: {imp:.4f}  value: {val:>8.2f}")
    
    return game, X_game, X_corrected, prob_loaded, prob_corrected


if __name__ == "__main__":
    game, X_loaded, X_corrected, prob_loaded, prob_corrected = diagnose_sf_ind()

