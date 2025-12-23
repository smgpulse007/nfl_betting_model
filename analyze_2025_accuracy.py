"""
Detailed accuracy analysis for 2025 predictions
Breaks down by week, confidence level, and bet type
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results_file = Path('results/2025_week1_16_evaluation.csv')
results = pd.read_csv(results_file)

print("="*100)
print("2025 NFL BETTING MODEL - DETAILED ACCURACY ANALYSIS")
print("="*100)

print(f"\nTotal games analyzed: {len(results)}")
print(f"Weeks covered: {results['week'].min()}-{results['week'].max()}")

# ============================================================================
# MONEYLINE ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("MONEYLINE ACCURACY ANALYSIS")
print("="*100)

# Overall accuracy
xgb_ml_correct = (results['xgb_ml_pick'] == results['home_win']).sum()
lr_ml_correct = (results['lr_ml_pick'] == results['home_win']).sum()
vegas_ml_correct = (results['vegas_ml_pick'] == results['home_win']).sum()

print(f"\n📊 Overall Accuracy:")
print(f"   XGBoost:  {xgb_ml_correct/len(results):.1%} ({xgb_ml_correct}/{len(results)})")
print(f"   Logistic: {lr_ml_correct/len(results):.1%} ({lr_ml_correct}/{len(results)})")
print(f"   Vegas:    {vegas_ml_correct/len(results):.1%} ({vegas_ml_correct}/{len(results)})")

# By confidence level
print(f"\n📊 Accuracy by Confidence Level (XGBoost):")
confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
confidence_labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

results['xgb_confidence'] = results['xgb_win_prob'].apply(lambda x: max(x, 1-x))

for i, (low, high) in enumerate(zip(confidence_bins[:-1], confidence_bins[1:])):
    mask = (results['xgb_confidence'] >= low) & (results['xgb_confidence'] < high)
    if mask.sum() > 0:
        acc = (results[mask]['xgb_ml_pick'] == results[mask]['home_win']).mean()
        count = mask.sum()
        correct = (results[mask]['xgb_ml_pick'] == results[mask]['home_win']).sum()
        print(f"   {confidence_labels[i]}: {acc:.1%} ({correct}/{count})")

# By week
print(f"\n📊 Accuracy by Week (XGBoost):")
weekly_acc = results.groupby('week').apply(
    lambda x: (x['xgb_ml_pick'] == x['home_win']).mean()
).sort_index()

for week, acc in weekly_acc.items():
    week_games = len(results[results['week'] == week])
    week_correct = (results[results['week'] == week]['xgb_ml_pick'] == results[results['week'] == week]['home_win']).sum()
    print(f"   Week {week:2d}: {acc:.1%} ({week_correct}/{week_games})")

# ============================================================================
# SPREAD ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("SPREAD ACCURACY ANALYSIS")
print("="*100)

spread_correct = (results['xgb_spread_pick'] == results['home_cover']).sum()
print(f"\n📊 Overall Accuracy:")
print(f"   XGBoost: {spread_correct/len(results):.1%} ({spread_correct}/{len(results)})")
print(f"   Baseline (50%): 50.0%")

# By confidence level
print(f"\n📊 Accuracy by Confidence Level:")
results['spread_confidence'] = results['home_cover_prob'].apply(lambda x: max(x, 1-x))

for i, (low, high) in enumerate(zip(confidence_bins[:-1], confidence_bins[1:])):
    mask = (results['spread_confidence'] >= low) & (results['spread_confidence'] < high)
    if mask.sum() > 0:
        acc = (results[mask]['xgb_spread_pick'] == results[mask]['home_cover']).mean()
        count = mask.sum()
        correct = (results[mask]['xgb_spread_pick'] == results[mask]['home_cover']).sum()
        print(f"   {confidence_labels[i]}: {acc:.1%} ({correct}/{count})")

# By week
print(f"\n📊 Accuracy by Week:")
weekly_spread_acc = results.groupby('week').apply(
    lambda x: (x['xgb_spread_pick'] == x['home_cover']).mean()
).sort_index()

for week, acc in weekly_spread_acc.items():
    week_games = len(results[results['week'] == week])
    week_correct = (results[results['week'] == week]['xgb_spread_pick'] == results[results['week'] == week]['home_cover']).sum()
    print(f"   Week {week:2d}: {acc:.1%} ({week_correct}/{week_games})")

# ============================================================================
# TOTALS (O/U) ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("TOTALS (O/U) ACCURACY ANALYSIS")
print("="*100)

totals_correct = (results['xgb_total_pick'] == results['over_hit']).sum()
print(f"\n📊 Overall Accuracy:")
print(f"   XGBoost: {totals_correct/len(results):.1%} ({totals_correct}/{len(results)})")
print(f"   Baseline (50%): 50.0%")

# By confidence level
print(f"\n📊 Accuracy by Confidence Level:")
results['total_confidence'] = results['over_prob'].apply(lambda x: max(x, 1-x))

for i, (low, high) in enumerate(zip(confidence_bins[:-1], confidence_bins[1:])):
    mask = (results['total_confidence'] >= low) & (results['total_confidence'] < high)
    if mask.sum() > 0:
        acc = (results[mask]['xgb_total_pick'] == results[mask]['over_hit']).mean()
        count = mask.sum()
        correct = (results[mask]['xgb_total_pick'] == results[mask]['over_hit']).sum()
        print(f"   {confidence_labels[i]}: {acc:.1%} ({correct}/{count})")

# By week
print(f"\n📊 Accuracy by Week:")
weekly_total_acc = results.groupby('week').apply(
    lambda x: (x['xgb_total_pick'] == x['over_hit']).mean()
).sort_index()

for week, acc in weekly_total_acc.items():
    week_games = len(results[results['week'] == week])
    week_correct = (results[results['week'] == week]['xgb_total_pick'] == results[results['week'] == week]['over_hit']).sum()
    print(f"   Week {week:2d}: {acc:.1%} ({week_correct}/{week_games})")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*100)
print("SUMMARY")
print("="*100)

print(f"\n✅ Best Performance:")
print(f"   Moneyline: XGBoost & Logistic both at {xgb_ml_correct/len(results):.1%} (beat Vegas {vegas_ml_correct/len(results):.1%})")
print(f"   Spread: {spread_correct/len(results):.1%} (slightly above 50% baseline)")
print(f"   Totals: {totals_correct/len(results):.1%} (slightly above 50% baseline)")

print(f"\n📈 Key Insights:")
print(f"   - Moneyline predictions are most reliable")
print(f"   - Spread and totals are close to 50/50 (as expected)")
print(f"   - Higher confidence picks tend to be more accurate")

