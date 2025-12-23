"""
Deep dive into XGBoost's predictions for games where it shows different confidence than Vegas.

Focus on understanding:
1. Why XGBoost has lower correlation with Vegas (0.891 vs 0.962-0.974)
2. What features drive XGBoost's unique predictions
3. Whether XGBoost is finding genuine signal or just noise
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

def main():
    print("="*80)
    print("XGBOOST DEEP DIVE: Why Lower Correlation with Vegas?")
    print("="*80)
    
    # Load data
    games_df, _ = load_and_prepare_data()
    
    week16 = games_df[
        (games_df['season'] == 2025) &
        (games_df['week'] == 16)
    ].copy()
    
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
    
    # Train models
    xgb = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0, random_state=42)
    xgb.fit(X_train, y_train)
    
    logistic = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    logistic.fit(X_train, y_train)
    
    # Predictions
    xgb_proba = xgb.predict_proba(X_week16)[:, 1]
    log_proba = logistic.predict_proba(X_week16)[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame({
        'away_team': week16['away_team'].values,
        'home_team': week16['home_team'].values,
        'spread': week16['spread_line'].values,
        'vegas_home_prob': week16['home_implied_prob'].values,
        'xgb_home_prob': xgb_proba,
        'log_home_prob': log_proba,
    })
    
    # Calculate differences
    results['xgb_vs_vegas'] = results['xgb_home_prob'] - results['vegas_home_prob']
    results['log_vs_vegas'] = results['log_home_prob'] - results['vegas_home_prob']
    results['xgb_vs_log'] = results['xgb_home_prob'] - results['log_home_prob']
    
    # Sort by XGBoost deviation from Vegas
    results['abs_xgb_deviation'] = abs(results['xgb_vs_vegas'])
    results_sorted = results.sort_values('abs_xgb_deviation', ascending=False)
    
    print("\n" + "="*80)
    print("GAMES RANKED BY XGBOOST DEVIATION FROM VEGAS")
    print("="*80)
    print(f"\n{'Matchup':<20} {'Vegas':>8} {'XGB':>8} {'Logistic':>10} {'XGB-Vegas':>12} {'Log-Vegas':>12}")
    print("-"*80)
    
    for _, row in results_sorted.iterrows():
        matchup = f"{row['away_team']}@{row['home_team']}"
        print(f"{matchup:<20} {row['vegas_home_prob']:>7.1%} {row['xgb_home_prob']:>7.1%} "
              f"{row['log_home_prob']:>9.1%} {row['xgb_vs_vegas']:>11.1%} {row['log_vs_vegas']:>11.1%}")
    
    # Identify games with biggest XGBoost deviations
    print("\n" + "="*80)
    print("TOP 5 XGBOOST DEVIATIONS - FEATURE ANALYSIS")
    print("="*80)

    # Reset index to ensure alignment
    X_week16_reset = X_week16.reset_index(drop=True)
    results_sorted_reset = results_sorted.reset_index(drop=True)

    # Get feature importance
    xgb_importance = pd.DataFrame({
        'feature': available_features,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False)

    for i in range(min(5, len(results_sorted_reset))):
        row = results_sorted_reset.iloc[i]
        matchup = f"{row['away_team']} @ {row['home_team']}"

        print(f"\n{'='*80}")
        print(f"{matchup}")
        print(f"Vegas: {row['vegas_home_prob']:.1%} | XGB: {row['xgb_home_prob']:.1%} | "
              f"Deviation: {row['xgb_vs_vegas']:+.1%}")
        print(f"{'='*80}")

        # Get feature values - use the same index i
        game_features = X_week16_reset.iloc[results_sorted.index[i]]
        
        # Focus on top 10 most important features
        print(f"\n{'Feature':<25} {'Value':>10} {'Importance':>12} {'vs Median':>12}")
        print("-"*80)
        
        for _, feat_row in xgb_importance.head(10).iterrows():
            feat_name = feat_row['feature']
            feat_value = game_features[feat_name]
            feat_imp = feat_row['importance']
            median_val = train_medians[feat_name]
            
            # Calculate deviation from median
            if median_val != 0:
                pct_diff = ((feat_value - median_val) / abs(median_val)) * 100
            else:
                pct_diff = 0
            
            marker = ""
            if abs(pct_diff) > 50:
                marker = " ⚠️⚠️"
            elif abs(pct_diff) > 25:
                marker = " ⚠️"
            
            print(f"{feat_name:<25} {feat_value:>10.3f} {feat_imp:>12.4f} {pct_diff:>11.1f}%{marker}")
    
    # Calculate correlations
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    xgb_corr = results['xgb_home_prob'].corr(results['vegas_home_prob'])
    log_corr = results['log_home_prob'].corr(results['vegas_home_prob'])
    
    print(f"\nCorrelation with Vegas implied probability:")
    print(f"  XGBoost:  r = {xgb_corr:.4f}")
    print(f"  Logistic: r = {log_corr:.4f}")
    print(f"\nXGBoost has {(log_corr - xgb_corr)*100:.2f}% lower correlation")
    print(f"This suggests XGBoost is finding {(1-xgb_corr)*100:.1f}% independent signal")
    
    # Save detailed results
    results_dir = Path("results")
    results_sorted.to_csv(results_dir / "xgboost_deviation_analysis.csv", index=False)
    
    print(f"\n✅ Saved to results/xgboost_deviation_analysis.csv")
    
    return results_sorted, xgb_importance


if __name__ == "__main__":
    results, importance = main()

