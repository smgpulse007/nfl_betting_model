"""
Check if injury data actually exists in our datasets
"""

import pandas as pd

print("=" * 120)
print("CHECKING FOR INJURY DATA IN ACTUAL DATASETS")
print("=" * 120)

# Load phase 6 data
print("\n[1/3] Checking phase6_game_level_1999_2024.parquet...")
df_phase6 = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')

# Search for injury columns
injury_keywords = ['injury', 'qb_out', 'injured', 'health', 'out', 'questionable', 'doubtful']
injury_cols = []

for col in df_phase6.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in injury_keywords):
        injury_cols.append(col)

print(f"  Total columns: {len(df_phase6.columns)}")
print(f"  Injury-related columns: {len(injury_cols)}")

if injury_cols:
    print(f"\n  ✅ FOUND {len(injury_cols)} injury-related columns:")
    for col in injury_cols:
        print(f"     • {col}")
    
    # Show sample data
    print(f"\n  Sample data (first 10 rows):")
    print(df_phase6[injury_cols].head(10))
    
    # Check data availability
    print(f"\n  Data availability:")
    for col in injury_cols:
        non_null = df_phase6[col].notna().sum()
        pct = non_null / len(df_phase6) * 100
        print(f"     • {col}: {non_null:,}/{len(df_phase6):,} ({pct:.1f}%)")
else:
    print(f"\n  ❌ NO injury-related columns found")

# Check pregame features
print(f"\n[2/3] Checking pregame_features_1999_2025_complete.parquet...")
df_pregame = pd.read_parquet('../results/phase8_results/pregame_features_1999_2025_complete.parquet')

injury_cols_pregame = []
for col in df_pregame.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in injury_keywords):
        injury_cols_pregame.append(col)

print(f"  Total columns: {len(df_pregame.columns)}")
print(f"  Injury-related columns: {len(injury_cols_pregame)}")

if injury_cols_pregame:
    print(f"\n  ✅ FOUND {len(injury_cols_pregame)} injury-related columns:")
    for col in injury_cols_pregame:
        print(f"     • {col}")
    
    # Check data availability
    print(f"\n  Data availability:")
    for col in injury_cols_pregame:
        non_null = df_pregame[col].notna().sum()
        pct = non_null / len(df_pregame) * 100
        print(f"     • {col}: {non_null:,}/{len(df_pregame):,} ({pct:.1f}%)")
else:
    print(f"\n  ❌ NO injury-related columns found")

# Check if we used injury features in training
print(f"\n[3/3] Checking which features were used in model training...")

# Load PyTorch checkpoint to see which features were used
import torch
pytorch_path = '../models/pytorch_nn.pth'
checkpoint = torch.load(pytorch_path, map_location='cpu', weights_only=False)
input_features = checkpoint['input_features']

print(f"  Total features used in training: {len(input_features)}")

# Check if any injury features were used
injury_features_used = [f for f in input_features if any(keyword in f.lower() for keyword in injury_keywords)]

if injury_features_used:
    print(f"\n  ✅ FOUND {len(injury_features_used)} injury features in training:")
    for f in injury_features_used:
        print(f"     • {f}")
else:
    print(f"\n  ❌ NO injury features were used in model training")

# Summary
print(f"\n{'='*120}")
print("SUMMARY")
print("=" * 120)

print(f"\n📊 FINDINGS:")
print(f"   • Phase 6 data has {len(injury_cols)} injury columns")
print(f"   • Pregame features has {len(injury_cols_pregame)} injury columns")
print(f"   • Model training used {len(injury_features_used)} injury features")

if len(injury_cols) > 0 and len(injury_features_used) == 0:
    print(f"\n⚠️ CRITICAL ISSUE:")
    print(f"   • Injury data EXISTS in the dataset")
    print(f"   • But was NOT USED in model training!")
    print(f"   • This is a feature selection issue, not a data issue")
elif len(injury_cols) == 0:
    print(f"\n⚠️ CRITICAL ISSUE:")
    print(f"   • NO injury data in the dataset")
    print(f"   • Need to add injury features from ESPN/nfl_data_py")
else:
    print(f"\n✅ Injury data is present and being used")

print(f"\n{'='*120}")

