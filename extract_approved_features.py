"""
Extract the list of 191 approved features (r >= 0.85) from validation results
"""
import pandas as pd
import json

# Set UTF-8 encoding for Windows console
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("="*120)
print("EXTRACTING APPROVED FEATURES (r >= 0.85)")
print("="*120)

# Load validation results
df = pd.read_csv('results/full_validation_results_2024.csv')

print(f"\n[1/3] Loading validation results...")
print(f"  Total features validated: {len(df)}")

# Filter to approved features (r >= 0.85)
approved = df[df['r'] >= 0.85].copy()
approved = approved.sort_values('r', ascending=False)

print(f"\n[2/3] Filtering to approved features (r >= 0.85)...")
print(f"  Approved features: {len(approved)}")
print(f"  Pass rate: {len(approved) / len(df) * 100:.1f}%")

# Breakdown by category
print(f"\n  Breakdown by category:")
for cat in approved['category'].unique():
    cat_df = approved[approved['category'] == cat]
    print(f"    {cat}: {len(cat_df)} features (mean r={cat_df['r'].mean():.4f})")

# Save approved features list
approved_list = approved['feature'].tolist()

# Save as JSON
with open('results/approved_features_r085.json', 'w') as f:
    json.dump({
        'threshold': 0.85,
        'count': len(approved_list),
        'features': approved_list,
        'metadata': {
            'total_validated': int(len(df)),
            'pass_rate': float(len(approved) / len(df)),
            'mean_r': float(approved['r'].mean()),
            'median_r': float(approved['r'].median()),
            'perfect_correlations': int((approved['r'] == 1.0).sum())
        }
    }, f, indent=2)

# Save as CSV
approved[['feature', 'category', 'r', 'mae', 'mape', 'status']].to_csv(
    'results/approved_features_r085.csv', 
    index=False
)

print(f"\n[3/3] Saved approved features...")
print(f"  JSON: results/approved_features_r085.json")
print(f"  CSV: results/approved_features_r085.csv")

# Display top 20 and bottom 20 of approved features
print(f"\n{'='*120}")
print(f"TOP 20 APPROVED FEATURES (highest correlation):")
print(f"{'='*120}")
for idx, row in approved.head(20).iterrows():
    print(f"  {row['feature']:<50} r={row['r']:>7.4f} [{row['category']}]")

print(f"\n{'='*120}")
print(f"BOTTOM 20 APPROVED FEATURES (lowest correlation, but still r >= 0.85):")
print(f"{'='*120}")
for idx, row in approved.tail(20).iterrows():
    print(f"  {row['feature']:<50} r={row['r']:>7.4f} [{row['category']}]")

print(f"\n{'='*120}")
print(f"✅ APPROVED FEATURES EXTRACTED: {len(approved_list)} features")
print(f"{'='*120}")

