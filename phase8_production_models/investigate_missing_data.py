"""
Investigate why we only matched 99/243 games in 2025 backtest
"""

import pandas as pd
import numpy as np

print("=" * 120)
print("INVESTIGATING MISSING 2025 DATA")
print("=" * 120)

# Load all relevant datasets
print("\n[1/5] Loading datasets...")

# Our predictions
df_pred = pd.read_csv('../results/phase8_results/2025_predictions.csv')
print(f"  ✅ Our predictions: {len(df_pred)} games")
print(f"     Weeks covered: {sorted(df_pred['week'].unique())}")

# Actual 2025 schedule
df_actual = pd.read_csv('../results/phase8_results/2025_schedule_actual.csv')
df_actual_completed = df_actual[df_actual['home_score'].notna()].copy()
print(f"  ✅ Actual schedule: {len(df_actual)} total, {len(df_actual_completed)} completed")

# Backtest results
df_backtest = pd.read_csv('../results/phase8_results/2025_backtest_weeks1_16.csv')
print(f"  ✅ Backtest matched: {len(df_backtest)} games")

# Feature datasets
df_phase6 = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')
print(f"  ✅ Phase 6 data: {len(df_phase6)} games (1999-2024)")

df_pregame = pd.read_parquet('../results/phase8_results/pregame_features_1999_2025_complete.parquet')
print(f"  ✅ Pregame features: {len(df_pregame)} games")
df_pregame_2025 = df_pregame[df_pregame['season'] == 2025]
print(f"     2025 games: {len(df_pregame_2025)}")
print(f"     2025 weeks: {sorted(df_pregame_2025['week'].unique())}")

# Analyze the gap
print(f"\n{'='*120}")
print("GAP ANALYSIS")
print("=" * 120)

print(f"\n1. PREDICTION COVERAGE:")
print(f"   • We generated predictions for: {len(df_pred)} games")
print(f"   • Actual completed games: {len(df_actual_completed)} games")
print(f"   • Gap: {len(df_actual_completed) - len(df_pred)} games missing")
print(f"   • Coverage: {len(df_pred)/len(df_actual_completed)*100:.1f}%")

print(f"\n2. BACKTEST MATCHING:")
print(f"   • Predictions: {len(df_pred)} games")
print(f"   • Matched in backtest: {len(df_backtest)} games")
print(f"   • Unmatched predictions: {len(df_pred) - len(df_backtest)} games")
print(f"   • Match rate: {len(df_backtest)/len(df_pred)*100:.1f}%")

# Check game_id format
print(f"\n3. GAME_ID FORMAT CHECK:")
print(f"   Sample prediction game_id: {df_pred['game_id'].iloc[0]}")
print(f"   Sample actual game_id: {df_actual['game_id'].iloc[0] if 'game_id' in df_actual.columns else 'NOT FOUND'}")

# Create game_id for actual if missing
if 'game_id' not in df_actual.columns:
    df_actual['game_id'] = (
        df_actual['season'].astype(str) + '_' +
        df_actual['week'].astype(str) + '_' +
        df_actual['away_team'] + '_' +
        df_actual['home_team']
    )
    print(f"   Created game_id for actual schedule")

# Find unmatched games
pred_game_ids = set(df_pred['game_id'])
actual_game_ids = set(df_actual_completed['game_id'])

unmatched_in_pred = pred_game_ids - actual_game_ids
unmatched_in_actual = actual_game_ids - pred_game_ids

print(f"\n4. UNMATCHED GAMES:")
print(f"   • Predictions not in actual: {len(unmatched_in_pred)}")
print(f"   • Actual not in predictions: {len(unmatched_in_actual)}")

if len(unmatched_in_actual) > 0:
    print(f"\n   Sample games we MISSED (first 10):")
    for i, game_id in enumerate(list(unmatched_in_actual)[:10], 1):
        game = df_actual_completed[df_actual_completed['game_id'] == game_id].iloc[0]
        print(f"   {i}. Week {game['week']}: {game['away_team']} @ {game['home_team']}")

# Check by week
print(f"\n5. COVERAGE BY WEEK:")
print(f"{'Week':<6} {'Actual':<10} {'Predicted':<12} {'Matched':<10} {'Coverage':<10}")
print("-" * 120)

for week in sorted(df_actual_completed['week'].unique()):
    actual_week = df_actual_completed[df_actual_completed['week'] == week]
    pred_week = df_pred[df_pred['week'] == week] if week in df_pred['week'].values else pd.DataFrame()
    backtest_week = df_backtest[df_backtest['week'] == week] if week in df_backtest['week'].values else pd.DataFrame()
    
    coverage = len(backtest_week) / len(actual_week) * 100 if len(actual_week) > 0 else 0
    
    print(f"{week:<6} {len(actual_week):<10} {len(pred_week):<12} {len(backtest_week):<10} {coverage:<10.1f}%")

# Check feature availability
print(f"\n{'='*120}")
print("FEATURE DATA ANALYSIS")
print("=" * 120)

print(f"\n6. PREGAME FEATURES COVERAGE:")
print(f"   • Total 2025 games in pregame features: {len(df_pregame_2025)}")
print(f"   • Total 2025 completed games: {len(df_actual_completed)}")
print(f"   • Gap: {len(df_actual_completed) - len(df_pregame_2025)} games")

# Check if pregame features have all weeks
print(f"\n   Pregame features by week:")
for week in sorted(df_pregame_2025['week'].unique()):
    week_data = df_pregame_2025[df_pregame_2025['week'] == week]
    print(f"   Week {week}: {len(week_data)} games")

# Check injury data
print(f"\n{'='*120}")
print("INJURY DATA CHECK")
print("=" * 120)

# Check if we have injury-related columns
injury_cols = [col for col in df_phase6.columns if 'injury' in col.lower() or 'health' in col.lower()]
print(f"\n7. INJURY-RELATED FEATURES:")
if len(injury_cols) > 0:
    print(f"   Found {len(injury_cols)} injury-related columns:")
    for col in injury_cols[:10]:
        print(f"   • {col}")
else:
    print(f"   ❌ NO injury-related features found in dataset!")

# Summary
print(f"\n{'='*120}")
print("SUMMARY & RECOMMENDATIONS")
print("=" * 120)

print(f"\n📊 KEY FINDINGS:")
print(f"   1. We only have predictions for {len(df_pred)} games out of {len(df_actual_completed)} completed")
print(f"   2. Only {len(df_backtest)} predictions matched with actual results")
print(f"   3. Pregame features dataset has {len(df_pregame_2025)} 2025 games")
print(f"   4. Missing {len(unmatched_in_actual)} completed games from our predictions")
print(f"   5. Injury features: {'Found' if len(injury_cols) > 0 else 'NOT FOUND'}")

print(f"\n⚠️ LIKELY CAUSES:")
print(f"   • Pregame features dataset incomplete for 2025")
print(f"   • Feature engineering pipeline didn't process all 2025 games")
print(f"   • Some weeks may have been excluded during data preparation")

print(f"\n✅ NEXT STEPS:")
print(f"   1. Re-run feature engineering for ALL 2025 games (weeks 1-18)")
print(f"   2. Generate predictions for missing games")
print(f"   3. Check if injury data is being updated")
print(f"   4. Verify team abbreviations match between datasets")

print(f"\n{'='*120}")

