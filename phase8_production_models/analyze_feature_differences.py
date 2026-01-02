"""
Analyze feature differences between the two datasets
"""

import pandas as pd
import json

print("=" * 120)
print("FEATURE COMPARISON: phase6_game_level vs pregame_features")
print("=" * 120)

# Load datasets
print("\n[1/4] Loading datasets...")
df_phase6 = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')
df_pregame = pd.read_parquet('../results/phase8_results/pregame_features_1999_2025_complete.parquet')

print(f"  ✅ phase6_game_level: {len(df_phase6.columns)} columns")
print(f"  ✅ pregame_features: {len(df_pregame.columns)} columns")

# Get feature lists (exclude metadata)
metadata = ['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'result', 'home_win']

phase6_features = [c for c in df_phase6.columns if c not in metadata]
pregame_features = [c for c in df_pregame.columns if c not in metadata]

print(f"\n  phase6_game_level features: {len(phase6_features)}")
print(f"  pregame_features features: {len(pregame_features)}")

# Find common and unique features
print(f"\n[2/4] Analyzing feature overlap...")

common = set(phase6_features) & set(pregame_features)
only_phase6 = set(phase6_features) - set(pregame_features)
only_pregame = set(pregame_features) - set(phase6_features)

print(f"\n  Common features: {len(common)}")
print(f"  Only in phase6_game_level: {len(only_phase6)}")
print(f"  Only in pregame_features: {len(only_pregame)}")

# Categorize unique features
print(f"\n[3/4] Categorizing unique features...")

# Features only in phase6
print(f"\n  Features ONLY in phase6_game_level ({len(only_phase6)}):")
print(f"    (These would be LOST if we switch to pregame_features)")

# Sample some
sample_phase6 = sorted(list(only_phase6))[:20]
for feat in sample_phase6:
    print(f"      • {feat}")
if len(only_phase6) > 20:
    print(f"      ... and {len(only_phase6) - 20} more")

# Features only in pregame
print(f"\n  Features ONLY in pregame_features ({len(only_pregame)}):")
print(f"    (These would be GAINED if we switch to pregame_features)")

# Categorize by type
injury_features = [f for f in only_pregame if 'injury' in f.lower() or 'qb_out' in f.lower()]
weather_features = [f for f in only_pregame if 'weather' in f.lower() or 'temp' in f.lower() or 'wind' in f.lower() or 'outdoor' in f.lower()]
other_features = [f for f in only_pregame if f not in injury_features and f not in weather_features]

print(f"\n    Injury features ({len(injury_features)}):")
for feat in sorted(injury_features):
    print(f"      • {feat}")

print(f"\n    Weather features ({len(weather_features)}):")
for feat in sorted(weather_features):
    print(f"      • {feat}")

if other_features:
    print(f"\n    Other features ({len(other_features)}):")
    for feat in sorted(other_features)[:10]:
        print(f"      • {feat}")
    if len(other_features) > 10:
        print(f"      ... and {len(other_features) - 10} more")

# Check current training features
print(f"\n[4/4] Checking current training features...")

import torch
checkpoint = torch.load('../models/pytorch_nn.pth', map_location='cpu', weights_only=False)
training_features = checkpoint['input_features']

print(f"\n  Current model uses: {len(training_features)} features")

# How many of current features are in pregame?
current_in_pregame = [f for f in training_features if f in pregame_features]
current_not_in_pregame = [f for f in training_features if f not in pregame_features]

print(f"  Available in pregame_features: {len(current_in_pregame)} ({len(current_in_pregame)/len(training_features)*100:.1f}%)")
print(f"  NOT in pregame_features: {len(current_not_in_pregame)} ({len(current_not_in_pregame)/len(training_features)*100:.1f}%)")

if current_not_in_pregame:
    print(f"\n  Features we'd LOSE:")
    for feat in sorted(current_not_in_pregame)[:20]:
        print(f"    • {feat}")
    if len(current_not_in_pregame) > 20:
        print(f"    ... and {len(current_not_in_pregame) - 20} more")

# Summary
print(f"\n{'='*120}")
print("RECOMMENDATION")
print("=" * 120)

if len(current_not_in_pregame) == 0:
    print(f"\n✅ SAFE TO SWITCH:")
    print(f"   • All {len(training_features)} current features exist in pregame_features")
    print(f"   • We'd GAIN {len(injury_features)} injury features")
    print(f"   • We'd GAIN {len(weather_features)} weather features")
    print(f"   • No features would be lost")
elif len(current_not_in_pregame) < 10:
    print(f"\n⚠️ MOSTLY SAFE TO SWITCH:")
    print(f"   • {len(current_in_pregame)}/{len(training_features)} current features exist in pregame_features")
    print(f"   • We'd LOSE {len(current_not_in_pregame)} features (minor)")
    print(f"   • We'd GAIN {len(injury_features)} injury features (major)")
    print(f"   • We'd GAIN {len(weather_features)} weather features (major)")
    print(f"   • Net benefit: POSITIVE")
else:
    print(f"\n❌ RISKY TO SWITCH:")
    print(f"   • We'd LOSE {len(current_not_in_pregame)} features (significant)")
    print(f"   • We'd GAIN {len(injury_features)} injury features")
    print(f"   • Need to investigate what we're losing")

print(f"\n{'='*120}")

