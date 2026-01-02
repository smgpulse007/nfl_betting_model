"""
Backtest 2025 predictions against actual results for weeks 1-16
"""

import pandas as pd
import numpy as np

print("=" * 120)
print("2025 SEASON BACKTEST - WEEKS 1-16")
print("=" * 120)

# Load our predictions
print("\n[1/4] Loading predictions...")
df_pred = pd.read_csv('../results/phase8_results/2025_predictions.csv')
print(f"  ✅ Loaded {len(df_pred)} predictions")

# Load actual results
print("\n[2/4] Loading actual results...")
df_actual = pd.read_csv('../results/phase8_results/2025_schedule_actual.csv')
print(f"  ✅ Loaded {len(df_actual)} games")

# Filter to completed games (weeks 1-16)
df_actual_completed = df_actual[df_actual['home_score'].notna()].copy()
print(f"  ✅ {len(df_actual_completed)} completed games")

# Create game_id for matching (with zero-padded week for consistency)
df_actual_completed['game_id'] = (
    df_actual_completed['season'].astype(str) + '_' +
    df_actual_completed['week'].astype(str).str.zfill(2) + '_' +
    df_actual_completed['away_team'] + '_' +
    df_actual_completed['home_team']
)

# Merge predictions with actual results
print("\n[3/4] Matching predictions with results...")
df_backtest = df_pred.merge(
    df_actual_completed[['game_id', 'home_score', 'away_score']],
    on='game_id',
    how='inner',
    suffixes=('_pred', '_actual')
)

print(f"  ✅ Matched {len(df_backtest)} games")

# Calculate actual winner
df_backtest['actual_winner'] = df_backtest.apply(
    lambda row: row['home_team'] if row['home_score_actual'] > row['away_score_actual'] else row['away_team'],
    axis=1
)

# Check if prediction was correct
df_backtest['correct'] = (df_backtest['predicted_winner'] == df_backtest['actual_winner']).astype(int)

# Calculate overall accuracy
overall_accuracy = df_backtest['correct'].mean()

print(f"\n{'='*120}")
print("OVERALL PERFORMANCE")
print("=" * 120)
print(f"Total Games: {len(df_backtest)}")
print(f"Correct Predictions: {df_backtest['correct'].sum()}")
print(f"Accuracy: {overall_accuracy:.1%}")
print(f"Average Confidence: {df_backtest['confidence'].mean():.1%}")

# Performance by week
print(f"\n{'='*120}")
print("PERFORMANCE BY WEEK")
print("=" * 120)
print(f"{'Week':<6} {'Games':<8} {'Correct':<10} {'Accuracy':<12} {'Avg Confidence':<15}")
print("-" * 120)

weekly_stats = []
for week in sorted(df_backtest['week'].unique()):
    week_data = df_backtest[df_backtest['week'] == week]
    week_acc = week_data['correct'].mean()
    week_conf = week_data['confidence'].mean()
    
    print(f"{week:<6} {len(week_data):<8} {week_data['correct'].sum():<10} {week_acc:<12.1%} {week_conf:<15.1%}")
    
    weekly_stats.append({
        'week': week,
        'games': len(week_data),
        'correct': week_data['correct'].sum(),
        'accuracy': week_acc,
        'avg_confidence': week_conf
    })

# Performance by confidence level
print(f"\n{'='*120}")
print("PERFORMANCE BY CONFIDENCE LEVEL")
print("=" * 120)

high_conf = df_backtest[df_backtest['confidence'] >= 0.65]
med_conf = df_backtest[(df_backtest['confidence'] >= 0.60) & (df_backtest['confidence'] < 0.65)]
low_conf = df_backtest[df_backtest['confidence'] < 0.60]

print(f"High Confidence (≥65%): {high_conf['correct'].mean():.1%} ({high_conf['correct'].sum()}/{len(high_conf)} games)")
print(f"Medium Confidence (60-65%): {med_conf['correct'].mean():.1%} ({med_conf['correct'].sum()}/{len(med_conf)} games)")
print(f"Low Confidence (<60%): {low_conf['correct'].mean():.1%} ({low_conf['correct'].sum()}/{len(low_conf)} games)")

# Save backtest results
print(f"\n[4/4] Saving results...")
output_path = '../results/phase8_results/2025_backtest_weeks1_16.csv'
df_backtest.to_csv(output_path, index=False)
print(f"  ✅ Saved to: {output_path}")

# Save weekly stats
weekly_df = pd.DataFrame(weekly_stats)
weekly_output = '../results/phase8_results/2025_weekly_performance.csv'
weekly_df.to_csv(weekly_output, index=False)
print(f"  ✅ Saved weekly stats to: {weekly_output}")

# Show best and worst weeks
print(f"\n{'='*120}")
print("BEST AND WORST WEEKS")
print("=" * 120)

weekly_df_sorted = weekly_df.sort_values('accuracy', ascending=False)
print(f"\nTop 3 Weeks:")
for i, (idx, row) in enumerate(weekly_df_sorted.head(3).iterrows(), 1):
    print(f"  {i}. Week {row['week']}: {row['accuracy']:.1%} ({row['correct']:.0f}/{row['games']:.0f})")

print(f"\nBottom 3 Weeks:")
for i, (idx, row) in enumerate(weekly_df_sorted.tail(3).iterrows(), 1):
    print(f"  {i}. Week {row['week']}: {row['accuracy']:.1%} ({row['correct']:.0f}/{row['games']:.0f})")

print(f"\n{'='*120}")
print("BACKTEST COMPLETE")
print("=" * 120)

