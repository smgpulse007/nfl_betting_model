"""
Fix team abbreviation mismatches between predictions and actual data
"""

import pandas as pd
import numpy as np

print("=" * 120)
print("FIXING TEAM ABBREVIATION MISMATCHES")
print("=" * 120)

# Load data
df_pred = pd.read_csv('../results/phase8_results/2025_predictions.csv')
df_actual = pd.read_csv('../results/phase8_results/2025_schedule_actual.csv')

# Get unique teams from each dataset
pred_teams = set(df_pred['home_team'].unique()) | set(df_pred['away_team'].unique())
actual_teams = set(df_actual['home_team'].unique()) | set(df_actual['away_team'].unique())

print(f"\n[1/3] Analyzing team abbreviations...")
print(f"  Teams in predictions: {len(pred_teams)}")
print(f"  Teams in actual: {len(actual_teams)}")

# Find differences
only_in_pred = pred_teams - actual_teams
only_in_actual = actual_teams - pred_teams

print(f"\n  Teams only in predictions: {sorted(only_in_pred)}")
print(f"  Teams only in actual: {sorted(only_in_actual)}")

# Common team abbreviation mappings
team_mapping = {
    'LA': 'LAR',  # Los Angeles Rams
    'WAS': 'WSH',  # Washington Commanders (changed abbreviation)
}

# Reverse mapping
reverse_mapping = {v: k for k, v in team_mapping.items()}

print(f"\n[2/3] Applying team abbreviation fixes...")

# Check which mapping to use
if 'LA' in only_in_pred and 'LAR' in only_in_actual:
    print(f"  ✅ Detected LA -> LAR mapping needed")
    use_mapping = team_mapping
elif 'LAR' in only_in_pred and 'LA' in only_in_actual:
    print(f"  ✅ Detected LAR -> LA mapping needed")
    use_mapping = reverse_mapping
else:
    use_mapping = {}

if 'WAS' in only_in_pred and 'WSH' in only_in_actual:
    print(f"  ✅ Detected WAS -> WSH mapping needed")
    use_mapping.update({'WAS': 'WSH'})
elif 'WSH' in only_in_pred and 'WAS' in only_in_actual:
    print(f"  ✅ Detected WSH -> WAS mapping needed")
    use_mapping.update({'WSH': 'WAS'})

print(f"\n  Final mapping: {use_mapping}")

# Apply mapping to predictions
df_pred_fixed = df_pred.copy()
df_pred_fixed['home_team'] = df_pred_fixed['home_team'].replace(use_mapping)
df_pred_fixed['away_team'] = df_pred_fixed['away_team'].replace(use_mapping)

# Recreate game_id
df_pred_fixed['game_id'] = (
    df_pred_fixed['season'].astype(str) + '_' +
    df_pred_fixed['week'].astype(str) + '_' +
    df_pred_fixed['away_team'] + '_' +
    df_pred_fixed['home_team']
)

# Create game_id for actual if missing
if 'game_id' not in df_actual.columns:
    df_actual['game_id'] = (
        df_actual['season'].astype(str) + '_' +
        df_actual['week'].astype(str) + '_' +
        df_actual['away_team'] + '_' +
        df_actual['home_team']
    )

# Test matching
df_actual_completed = df_actual[df_actual['home_score'].notna()].copy()

# Merge to test
df_test_merge = df_pred_fixed.merge(
    df_actual_completed[['game_id', 'home_score', 'away_score']],
    on='game_id',
    how='inner',
    suffixes=('_pred', '_actual')
)

print(f"\n[3/3] Testing new matching...")
print(f"  Original matches: 99 games")
print(f"  New matches: {len(df_test_merge)} games")
print(f"  Improvement: +{len(df_test_merge) - 99} games")

if len(df_test_merge) > 99:
    print(f"\n  ✅ SUCCESS! Fixed team abbreviation mismatches")
    
    # Save fixed predictions
    output_path = '../results/phase8_results/2025_predictions_fixed.csv'
    df_pred_fixed.to_csv(output_path, index=False)
    print(f"\n  💾 Saved fixed predictions to: {output_path}")
    
    # Re-run backtest with fixed data
    print(f"\n  🔄 Re-running backtest with fixed data...")
    
    # Calculate actual winner
    df_test_merge['actual_winner'] = df_test_merge.apply(
        lambda row: row['home_team'] if row['home_score_actual'] > row['away_score_actual'] else row['away_team'],
        axis=1
    )
    
    # Check if prediction was correct
    df_test_merge['correct'] = (df_test_merge['predicted_winner'] == df_test_merge['actual_winner']).astype(int)
    
    # Calculate overall accuracy
    overall_accuracy = df_test_merge['correct'].mean()
    
    print(f"\n{'='*120}")
    print("UPDATED BACKTEST RESULTS")
    print("=" * 120)
    print(f"Total Games: {len(df_test_merge)}")
    print(f"Correct Predictions: {df_test_merge['correct'].sum()}")
    print(f"Accuracy: {overall_accuracy:.1%}")
    print(f"Average Confidence: {df_test_merge['confidence'].mean():.1%}")
    
    # Performance by week
    print(f"\n{'Week':<6} {'Games':<8} {'Correct':<10} {'Accuracy':<12}")
    print("-" * 120)
    
    for week in sorted(df_test_merge['week'].unique()):
        week_data = df_test_merge[df_test_merge['week'] == week]
        week_acc = week_data['correct'].mean()
        print(f"{week:<6} {len(week_data):<8} {week_data['correct'].sum():<10} {week_acc:<12.1%}")
    
    # Save updated backtest
    backtest_output = '../results/phase8_results/2025_backtest_weeks1_16_fixed.csv'
    df_test_merge.to_csv(backtest_output, index=False)
    print(f"\n  💾 Saved updated backtest to: {backtest_output}")
    
else:
    print(f"\n  ⚠️ No improvement - issue may be elsewhere")

print(f"\n{'='*120}")

