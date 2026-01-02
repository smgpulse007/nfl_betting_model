"""
Check for data leakage in pregame_features dataset
"""

import pandas as pd
import numpy as np

print("=" * 120)
print("CHECKING FOR DATA LEAKAGE IN PREGAME_FEATURES")
print("=" * 120)

# Load data
print("\n[1/3] Loading data...")
df = pd.read_parquet('../results/phase8_results/pregame_features_1999_2025_complete.parquet')
print(f"  ✅ Loaded: {len(df)} games")

# Check a specific game
print("\n[2/3] Checking specific game for leakage...")

# Get a 2024 game
sample_game = df[(df['season'] == 2024) & (df['week'] == 1)].iloc[0]

print(f"\n  Sample Game:")
print(f"    game_id: {sample_game['game_id']}")
print(f"    {sample_game['away_team']} @ {sample_game['home_team']}")
print(f"    Score: {sample_game['away_score']} - {sample_game['home_score']}")
print(f"    Winner: {'Home' if sample_game['home_score'] > sample_game['away_score'] else 'Away'}")

# Check if any features contain the game score
print(f"\n  Checking for score leakage...")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
metadata = ['season', 'week', 'home_score', 'away_score', 'home_win']
feature_cols = [c for c in numeric_cols if c not in metadata]

print(f"  Total numeric features: {len(feature_cols)}")

# Check if any feature values match the game scores
away_score = sample_game['away_score']
home_score = sample_game['home_score']

suspicious_features = []
for col in feature_cols:
    val = sample_game[col]
    if pd.notna(val):
        if abs(val - away_score) < 0.01 or abs(val - home_score) < 0.01:
            suspicious_features.append((col, val))

if suspicious_features:
    print(f"\n  ⚠️  Found {len(suspicious_features)} features with values matching game scores:")
    for feat, val in suspicious_features[:10]:
        print(f"      • {feat}: {val}")
else:
    print(f"  ✅ No features match game scores")

# Check for perfect correlation with outcome
print(f"\n[3/3] Checking for perfect predictors...")

# Create target
df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

# Split 2024 data
df_2024 = df[df['season'] == 2024].copy()

# Check each feature for perfect correlation
perfect_predictors = []
for col in feature_cols:
    if col in df_2024.columns:
        # Check if feature perfectly predicts outcome
        df_test = df_2024[[col, 'home_win']].dropna()
        if len(df_test) > 0:
            # Try simple threshold
            for threshold in np.percentile(df_test[col], [25, 50, 75]):
                pred = (df_test[col] > threshold).astype(int)
                acc = (pred == df_test['home_win']).mean()
                if acc > 0.99:  # 99%+ accuracy
                    perfect_predictors.append((col, threshold, acc))
                    break

if perfect_predictors:
    print(f"\n  🚨 Found {len(perfect_predictors)} features with >99% accuracy:")
    for feat, thresh, acc in perfect_predictors[:10]:
        print(f"      • {feat} (threshold={thresh:.2f}): {acc:.4f} accuracy")
else:
    print(f"  ✅ No perfect predictors found")

# Check for features that shouldn't exist
print(f"\n  Checking for suspicious feature names...")

suspicious_names = []
for col in feature_cols:
    col_lower = col.lower()
    if any(word in col_lower for word in ['score', 'result', 'final', 'outcome']):
        suspicious_names.append(col)

if suspicious_names:
    print(f"\n  ⚠️  Found {len(suspicious_names)} features with suspicious names:")
    for feat in suspicious_names[:20]:
        print(f"      • {feat}")
else:
    print(f"  ✅ No suspicious feature names")

print(f"\n{'='*120}")
print("SUMMARY")
print("=" * 120)

if suspicious_features or perfect_predictors or suspicious_names:
    print(f"\n⚠️  POTENTIAL DATA LEAKAGE DETECTED!")
    print(f"   • Features matching scores: {len(suspicious_features)}")
    print(f"   • Perfect predictors: {len(perfect_predictors)}")
    print(f"   • Suspicious names: {len(suspicious_names)}")
else:
    print(f"\n✅ No obvious data leakage detected")
    print(f"   • But model still achieves 100% accuracy")
    print(f"   • Need to investigate further...")

print(f"\n{'='*120}")

