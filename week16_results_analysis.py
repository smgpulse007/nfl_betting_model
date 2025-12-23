"""
Week 16 Results Analysis - XGBoost vs Vegas Performance

Compare XGBoost predictions against actual results to validate the model's edge.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import nfl_data_py as nfl

def analyze_week16_results():
    print("="*100)
    print("WEEK 16 RESULTS ANALYSIS - XGBOOST VS VEGAS")
    print("="*100)
    
    # Load actual Week 16 results
    print("\n[1/4] Loading Week 16 actual results...")
    schedule_2025 = nfl.import_schedules([2025])
    week16_actual = schedule_2025[
        (schedule_2025['week'] == 16) & 
        (schedule_2025['home_score'].notna())
    ].copy()
    
    print(f"Found {len(week16_actual)} completed Week 16 games")
    
    # Load our predictions
    print("\n[2/4] Loading XGBoost predictions...")
    predictions = pd.read_csv("results/week16_2025_all_models.csv")
    
    # Merge predictions with actual results
    results = week16_actual.merge(
        predictions,
        on=['away_team', 'home_team'],
        how='inner',
        suffixes=('_actual', '_pred')
    )
    
    print(f"Matched {len(results)} games")
    
    # Calculate actual outcomes
    results['actual_home_win'] = (results['home_score'] > results['away_score']).astype(int)
    results['actual_margin'] = results['home_score'] - results['away_score']
    
    # Rename columns for consistency
    results = results.rename(columns={
        'XGBoost_proba': 'xgb_home_prob',
        'Logistic_proba': 'log_home_prob',
        'CatBoost_proba': 'cat_home_prob',
        'RandomForest_proba': 'rf_home_prob'
    })

    # Calculate predictions
    results['xgb_pick_home'] = (results['xgb_home_prob'] > 0.5).astype(int)
    results['vegas_pick_home'] = (results['home_implied_prob'] > 0.5).astype(int)
    results['log_pick_home'] = (results['log_home_prob'] > 0.5).astype(int)

    # Calculate correct predictions
    results['xgb_correct'] = (results['xgb_pick_home'] == results['actual_home_win']).astype(int)
    results['vegas_correct'] = (results['vegas_pick_home'] == results['actual_home_win']).astype(int)
    results['log_correct'] = (results['log_pick_home'] == results['actual_home_win']).astype(int)

    # Calculate deviation from Vegas
    results['xgb_vegas_deviation'] = results['xgb_home_prob'] - results['home_implied_prob']
    results['abs_deviation'] = results['xgb_vegas_deviation'].abs()
    
    print("\n[3/4] Calculating performance metrics...")
    
    # Overall accuracy
    xgb_accuracy = results['xgb_correct'].mean()
    vegas_accuracy = results['vegas_correct'].mean()
    log_accuracy = results['log_correct'].mean()
    
    print("\n" + "="*100)
    print("OVERALL ACCURACY")
    print("="*100)
    print(f"XGBoost:  {xgb_accuracy:.1%} ({results['xgb_correct'].sum()}/{len(results)})")
    print(f"Vegas:    {vegas_accuracy:.1%} ({results['vegas_correct'].sum()}/{len(results)})")
    print(f"Logistic: {log_accuracy:.1%} ({results['log_correct'].sum()}/{len(results)})")
    
    edge = xgb_accuracy - vegas_accuracy
    print(f"\n🎯 XGBoost Edge over Vegas: {edge:+.1%}")
    
    # High confidence picks (>70%)
    high_conf = results[results['xgb_home_prob'].apply(lambda x: max(x, 1-x)) > 0.7]
    if len(high_conf) > 0:
        high_conf_acc = high_conf['xgb_correct'].mean()
        print(f"\n📊 High Confidence Picks (>70%): {high_conf_acc:.1%} ({high_conf['xgb_correct'].sum()}/{len(high_conf)})")
    
    # Games where XGBoost disagreed with Vegas (>5% deviation)
    disagreements = results[results['abs_deviation'] > 0.05].copy()
    if len(disagreements) > 0:
        print("\n" + "="*100)
        print(f"XGBOOST DISAGREEMENTS WITH VEGAS (>{5}% deviation)")
        print("="*100)
        
        disagree_xgb_acc = disagreements['xgb_correct'].mean()
        disagree_vegas_acc = disagreements['vegas_correct'].mean()
        
        print(f"XGBoost on disagreements: {disagree_xgb_acc:.1%} ({disagreements['xgb_correct'].sum()}/{len(disagreements)})")
        print(f"Vegas on disagreements:   {disagree_vegas_acc:.1%} ({disagreements['vegas_correct'].sum()}/{len(disagreements)})")
        print(f"\n🎯 XGBoost Edge on disagreements: {disagree_xgb_acc - disagree_vegas_acc:+.1%}")
    
    # Detailed game-by-game results
    print("\n[4/4] Generating detailed results...")
    print("\n" + "="*100)
    print("GAME-BY-GAME RESULTS")
    print("="*100)
    
    # Sort by absolute deviation to see biggest disagreements first
    results_sorted = results.sort_values('abs_deviation', ascending=False)
    
    print(f"\n{'Matchup':<18} {'Score':<10} {'Winner':<6} {'XGB':<6} {'Vegas':<6} "
          f"{'XGB%':>6} {'Vegas%':>7} {'Dev':>7} {'XGB✓':>5} {'Vegas✓':>7}")
    print("-"*100)
    
    for _, row in results_sorted.iterrows():
        matchup = f"{row['away_team']}@{row['home_team']}"
        score = f"{int(row['away_score'])}-{int(row['home_score'])}"
        winner = row['home_team'] if row['actual_home_win'] else row['away_team']
        
        xgb_pick = row['home_team'] if row['xgb_pick_home'] else row['away_team']
        vegas_pick = row['home_team'] if row['vegas_pick_home'] else row['away_team']
        
        xgb_check = "✅" if row['xgb_correct'] else "❌"
        vegas_check = "✅" if row['vegas_correct'] else "❌"
        
        deviation_marker = ""
        if abs(row['xgb_vegas_deviation']) > 0.10:
            deviation_marker = " 🔥"
        
        print(f"{matchup:<18} {score:<10} {winner:<6} {xgb_pick:<6} {vegas_pick:<6} "
              f"{row['xgb_home_prob']:>5.1%} {row['home_implied_prob']:>6.1%} "
              f"{row['xgb_vegas_deviation']:>+6.1%} {xgb_check:>5} {vegas_check:>7}{deviation_marker}")
    
    # Calibration analysis
    print("\n" + "="*100)
    print("CALIBRATION ANALYSIS (Were probabilities accurate?)")
    print("="*100)
    
    # Bin predictions by probability
    bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
    results['xgb_conf_bin'] = pd.cut(
        results['xgb_home_prob'].apply(lambda x: max(x, 1-x)),
        bins=bins,
        labels=['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    )
    
    calibration = results.groupby('xgb_conf_bin', observed=True).agg({
        'xgb_correct': ['count', 'sum', 'mean']
    }).round(3)
    
    if len(calibration) > 0:
        print("\nXGBoost Calibration:")
        print(f"{'Confidence':<12} {'Games':<8} {'Correct':<10} {'Accuracy':<10}")
        print("-"*50)
        for idx, row in calibration.iterrows():
            count = int(row[('xgb_correct', 'count')])
            correct = int(row[('xgb_correct', 'sum')])
            accuracy = row[('xgb_correct', 'mean')]
            print(f"{idx:<12} {count:<8} {correct:<10} {accuracy:<10.1%}")
    
    # Save detailed results
    results_dir = Path("results")
    output_cols = [
        'away_team', 'home_team', 'away_score', 'home_score', 'actual_margin',
        'xgb_home_prob', 'home_implied_prob', 'log_home_prob',
        'xgb_vegas_deviation', 'actual_home_win', 'xgb_correct', 'vegas_correct', 'log_correct'
    ]
    results[output_cols].to_csv(results_dir / "week16_results_analysis.csv", index=False)
    print(f"\n✅ Saved detailed results to results/week16_results_analysis.csv")
    
    return results


if __name__ == "__main__":
    results = analyze_week16_results()

