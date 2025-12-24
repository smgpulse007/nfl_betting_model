"""Check for data leakage in selected features."""

import json
import pandas as pd

# Load selected features
with open('../results/phase7_feature_selection/selected_features.json') as f:
    features = json.load(f)

print(f"Total selected features: {len(features)}\n")

# Check for suspicious features
leakage_keywords = ['win', 'result', 'score', 'points', 'outcome', 'final']

suspicious_features = []
for feature in features:
    feature_lower = feature.lower()
    for keyword in leakage_keywords:
        if keyword in feature_lower:
            suspicious_features.append((feature, keyword))
            break

if suspicious_features:
    print(f"⚠️  FOUND {len(suspicious_features)} SUSPICIOUS FEATURES:")
    for feature, keyword in suspicious_features[:30]:
        print(f"  - {feature} (contains '{keyword}')")
else:
    print("✅ No obvious data leakage detected")

# Load dataset and check correlations
print("\n" + "="*80)
print("Checking correlations with target...")
print("="*80)

df = pd.read_parquet('../results/game_level_predictions_dataset.parquet')

# Check correlation of each feature with home_win
correlations = []
for feature in features:
    if feature in df.columns:
        valid_data = df[[feature, 'home_win']].dropna()
        if len(valid_data) > 100:
            corr = abs(valid_data[feature].corr(valid_data['home_win']))
            correlations.append((feature, corr))

correlations.sort(key=lambda x: x[1], reverse=True)

print("\nTop 20 features by correlation with home_win:")
for feature, corr in correlations[:20]:
    print(f"  {feature:60s} r = {corr:.4f}")

# Check for perfect correlations (r > 0.99)
perfect_corr = [(f, c) for f, c in correlations if c > 0.99]
if perfect_corr:
    print(f"\n⚠️  FOUND {len(perfect_corr)} FEATURES WITH PERFECT CORRELATION (r > 0.99):")
    for feature, corr in perfect_corr:
        print(f"  - {feature}: r = {corr:.6f}")

