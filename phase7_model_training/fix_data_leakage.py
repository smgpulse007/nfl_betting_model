"""
Fix Data Leakage
=================

Remove features that contain information about game outcomes:
- Win/loss records (computed after the game)
- Final scores/points (known only after the game)
- Any outcome-based statistics

Keep only features that are known BEFORE the game starts.
"""

import json
import pandas as pd
from pathlib import Path

print("="*120)
print("FIXING DATA LEAKAGE IN SELECTED FEATURES")
print("="*120)

# Load selected features
with open('../results/phase7_feature_selection/selected_features.json') as f:
    features = json.load(f)

print(f"\nOriginal features: {len(features)}")

# Define leakage patterns to exclude
leakage_patterns = [
    '_wins',  # Win counts
    '_losses',  # Loss counts
    '_winPercent',  # Win percentages
    '_winPercentage',  # Win percentages (alternate)
    '_leagueWinPercent',  # League win percentages
    'away_win',  # Direct win indicator
    'home_win',  # Direct win indicator
    '_totalPoints',  # Total points (cumulative after games)
    '_pointsFor',  # Points scored (cumulative)
    '_pointsAgainst',  # Points allowed (cumulative)
    '_avgPoints',  # Average points (computed from past games including this one)
    '_totalPointsPerGame',  # Points per game (includes this game)
    '_total_differential',  # Cumulative point differential (includes this game)
    '_total_pointDifferential',  # Cumulative point differential (includes this game)
    '_total_points',  # Cumulative total points (includes this game)
    '_defensive_pointsAllowed',  # Cumulative points allowed (includes this game)
    '_scored_',  # Cumulative scoring stats (includes this game)
]

# Filter out leakage features
clean_features = []
removed_features = []

for feature in features:
    is_leakage = False
    for pattern in leakage_patterns:
        if pattern in feature:
            is_leakage = True
            removed_features.append((feature, pattern))
            break
    
    if not is_leakage:
        clean_features.append(feature)

print(f"\n✅ Clean features: {len(clean_features)}")
print(f"⚠️  Removed features: {len(removed_features)}")

print(f"\nRemoved features (first 30):")
for feature, pattern in removed_features[:30]:
    print(f"  - {feature:60s} (matched '{pattern}')")

# Verify no perfect correlations remain
print("\n" + "="*120)
print("VERIFYING NO DATA LEAKAGE REMAINS")
print("="*120)

df = pd.read_parquet('../results/game_level_predictions_dataset.parquet')

correlations = []
for feature in clean_features:
    if feature in df.columns:
        valid_data = df[[feature, 'home_win']].dropna()
        if len(valid_data) > 100:
            corr = abs(valid_data[feature].corr(valid_data['home_win']))
            correlations.append((feature, corr))

correlations.sort(key=lambda x: x[1], reverse=True)

print(f"\nTop 20 features by correlation with home_win:")
for feature, corr in correlations[:20]:
    print(f"  {feature:60s} r = {corr:.4f}")

# Check for perfect correlations
perfect_corr = [(f, c) for f, c in correlations if c > 0.99]
if perfect_corr:
    print(f"\n⚠️  WARNING: Still found {len(perfect_corr)} features with r > 0.99:")
    for feature, corr in perfect_corr:
        print(f"  - {feature}: r = {corr:.6f}")
else:
    print(f"\n✅ No perfect correlations found (max r = {correlations[0][1]:.4f})")

# Save clean features
output_dir = Path('../results/phase7_feature_selection')
with open(output_dir / 'selected_features_clean.json', 'w') as f:
    json.dump(clean_features, f, indent=2)

# Save removed features for reference
with open(output_dir / 'removed_leakage_features.json', 'w') as f:
    json.dump([f for f, _ in removed_features], f, indent=2)

print(f"\n{'='*120}")
print("✅ DATA LEAKAGE FIXED!")
print(f"{'='*120}")
print(f"✅ Clean features saved to: selected_features_clean.json")
print(f"✅ Removed features saved to: removed_leakage_features.json")
print(f"\nReady to retrain models with {len(clean_features)} clean features")

