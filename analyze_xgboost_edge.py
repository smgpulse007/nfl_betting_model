"""
Analyze why XGBoost disagrees with Vegas on certain games.

XGBoost has the LOWEST correlation with Vegas (r=0.891) compared to:
- Logistic: r=0.968
- CatBoost: r=0.962  
- RandomForest: r=0.974

This script investigates what features XGBoost is weighting differently.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from run_tier_sa_backtest import load_and_prepare_data
from version import FEATURES

def analyze_xgboost_decisions():
    """Deep dive into XGBoost's decision-making process."""
    
    print("="*70)
    print("ANALYZING XGBOOST'S INDEPENDENT PREDICTIONS")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data with TIER S+A features...")
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
    
    # Compute medians for imputation
    train_medians = X_train.median()
    
    # Prepare Week 16 data
    X_week16 = week16[available_features].copy()
    for col in X_week16.columns:
        if X_week16[col].isna().any():
            X_week16[col] = X_week16[col].fillna(train_medians.get(col, 0))
    
    # Train XGBoost and Logistic for comparison
    print("\n[2/5] Training XGBoost and Logistic models...")
    xgb = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0, random_state=42)
    xgb.fit(X_train, y_train)
    
    logistic = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    logistic.fit(X_train, y_train)
    
    # Get predictions
    xgb_proba = xgb.predict_proba(X_week16)[:, 1]
    log_proba = logistic.predict_proba(X_week16)[:, 1]
    
    # Feature importance
    print("\n[3/5] Extracting feature importance...")
    xgb_importance = pd.DataFrame({
        'feature': available_features,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    log_importance = pd.DataFrame({
        'feature': available_features,
        'coefficient': logistic.coef_[0],
        'abs_coef': np.abs(logistic.coef_[0])
    }).sort_values('abs_coef', ascending=False)
    
    print("\n" + "="*70)
    print("TOP 15 FEATURES - XGBOOST vs LOGISTIC")
    print("="*70)
    print(f"\n{'Feature':<25} {'XGB Importance':>15} {'Logistic |Coef|':>18}")
    print("-"*70)
    
    # Merge and display
    merged = xgb_importance.merge(log_importance, on='feature')
    for _, row in merged.head(15).iterrows():
        print(f"{row['feature']:<25} {row['importance']:>15.4f} {row['abs_coef']:>18.4f}")
    
    # Identify games where XGBoost disagrees with Vegas
    print("\n[4/5] Identifying XGBoost disagreements with Vegas...")
    
    results = week16[['home_team', 'away_team', 'spread_line', 'home_implied_prob']].copy()
    results['xgb_proba'] = xgb_proba
    results['log_proba'] = log_proba
    results['vegas_pick_home'] = results['home_implied_prob'] > 0.5
    results['xgb_pick_home'] = results['xgb_proba'] > 0.5
    results['log_pick_home'] = results['log_proba'] > 0.5
    results['xgb_disagrees'] = results['xgb_pick_home'] != results['vegas_pick_home']
    
    disagreements = results[results['xgb_disagrees']].copy()
    
    print(f"\nXGBoost disagrees with Vegas on {len(disagreements)} games:")
    print("-"*70)
    for _, row in disagreements.iterrows():
        vegas_pick = "HOME" if row['vegas_pick_home'] else "AWAY"
        xgb_pick = "HOME" if row['xgb_pick_home'] else "AWAY"
        print(f"\n{row['away_team']} @ {row['home_team']}")
        print(f"  Vegas: {vegas_pick} ({row['home_implied_prob']:.1%})")
        print(f"  XGBoost: {xgb_pick} ({row['xgb_proba']:.1%})")
        print(f"  Logistic: {'HOME' if row['log_pick_home'] else 'AWAY'} ({row['log_proba']:.1%})")
    
    # Deep dive into feature values for disagreement games
    print("\n[5/5] Feature analysis for XGBoost disagreements...")
    
    for idx, row in disagreements.iterrows():
        print("\n" + "="*70)
        print(f"{row['away_team']} @ {row['home_team']} - FEATURE BREAKDOWN")
        print("="*70)
        
        # Get feature values for this game
        game_features = X_week16.loc[idx]
        
        # Show top features by XGBoost importance
        print(f"\n{'Feature':<25} {'Value':>12} {'XGB Imp':>12} {'Train Median':>15}")
        print("-"*70)
        
        for _, feat_row in xgb_importance.head(20).iterrows():
            feat_name = feat_row['feature']
            feat_value = game_features[feat_name]
            feat_imp = feat_row['importance']
            train_median = train_medians[feat_name]
            
            # Highlight if significantly different from median
            diff_marker = ""
            if abs(feat_value - train_median) > train_medians[feat_name] * 0.5:
                diff_marker = " ⚠️"
            
            print(f"{feat_name:<25} {feat_value:>12.3f} {feat_imp:>12.4f} {train_median:>15.3f}{diff_marker}")
    
    return xgb_importance, disagreements, results


if __name__ == "__main__":
    xgb_imp, disagreements, all_results = analyze_xgboost_decisions()
    
    # Save results
    results_dir = Path("results")
    xgb_imp.to_csv(results_dir / "xgboost_feature_importance.csv", index=False)
    all_results.to_csv(results_dir / "xgboost_week16_analysis.csv", index=False)
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print("="*70)
    print(f"Saved to:")
    print(f"  - results/xgboost_feature_importance.csv")
    print(f"  - results/xgboost_week16_analysis.csv")

