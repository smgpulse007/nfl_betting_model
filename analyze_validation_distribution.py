"""
Analyze validation results distribution to recommend evidence-based thresholds
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set UTF-8 encoding for Windows console
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("="*120)
print("VALIDATION DISTRIBUTION ANALYSIS")
print("="*120)

# Load validation results
df = pd.read_csv('results/full_validation_results_2024.csv')

print(f"\n[1/5] Loading validation results...")
print(f"  Total features validated: {len(df)}")
print(f"  Categories: {df['category'].value_counts().to_dict()}")

# Analyze correlation distribution
print(f"\n[2/5] Correlation distribution...")
print(f"\n  Overall statistics:")
print(f"    Mean: {df['r'].mean():.4f}")
print(f"    Median: {df['r'].median():.4f}")
print(f"    Std: {df['r'].std():.4f}")
print(f"    Min: {df['r'].min():.4f}")
print(f"    Max: {df['r'].max():.4f}")

print(f"\n  Percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"    {p}th: {df['r'].quantile(p/100):.4f}")

# Analyze by category
print(f"\n[3/5] Correlation by category...")
for cat in df['category'].unique():
    cat_df = df[df['category'] == cat]
    print(f"\n  {cat}:")
    print(f"    Count: {len(cat_df)}")
    print(f"    Mean r: {cat_df['r'].mean():.4f}")
    print(f"    Median r: {cat_df['r'].median():.4f}")
    print(f"    Pass rate (r>0.95): {(cat_df['r'] > 0.95).sum() / len(cat_df) * 100:.1f}%")
    print(f"    Pass rate (r>0.85): {(cat_df['r'] > 0.85).sum() / len(cat_df) * 100:.1f}%")
    print(f"    Pass rate (r>0.70): {(cat_df['r'] > 0.70).sum() / len(cat_df) * 100:.1f}%")

# Recommend thresholds
print(f"\n[4/5] Threshold recommendations...")

thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
print(f"\n  Features retained at different thresholds:")
print(f"  {'Threshold':<12} {'Count':<8} {'% of Total':<12} {'EXACT':<8} {'PARTIAL':<10} {'CLOSE':<8}")
print(f"  {'-'*70}")

for thresh in thresholds:
    retained = df[df['r'] >= thresh]
    exact = retained[retained['category'] == 'EXACT MATCH']
    partial = retained[retained['category'] == 'PARTIAL MATCH']
    close = retained[retained['category'] == 'CLOSE APPROXIMATION']

    print(f"  r >= {thresh:<6.2f}  {len(retained):<8} {len(retained)/len(df)*100:<11.1f}% {len(exact):<8} {len(partial):<10} {len(close):<8}")

# Identify problematic features
print(f"\n[5/5] Problematic features (r < 0.50)...")
problematic = df[df['r'] < 0.50].sort_values('r')
print(f"\n  Count: {len(problematic)}")
print(f"\n  Features:")
for idx, row in problematic.iterrows():
    print(f"    {row['feature']:<50} r={row['r']:>7.4f} (MAPE={row['mape']:>6.1f}%) [{row['category']}]")

print(f"\n{'='*120}")
print(f"RECOMMENDATIONS")
print(f"{'='*120}")

print(f"\n1. **Recommended Threshold: r >= 0.85**")
print(f"   - Retains {len(df[df['r'] >= 0.85])} features ({len(df[df['r'] >= 0.85])/len(df)*100:.1f}%)")
print(f"   - Balances quality (high correlation) with quantity (enough features)")
print(f"   - Excludes {len(df[df['r'] < 0.85])} low-quality features")

print(f"\n2. **Alternative Threshold: r >= 0.70**")
print(f"   - Retains {len(df[df['r'] >= 0.70])} features ({len(df[df['r'] >= 0.70])/len(df)*100:.1f}%)")
print(f"   - More permissive, includes more features")
print(f"   - May include some noisy features")

print(f"\n3. **Conservative Threshold: r >= 0.95**")
print(f"   - Retains {len(df[df['r'] >= 0.95])} features ({len(df[df['r'] >= 0.95])/len(df)*100:.1f}%)")
print(f"   - Highest quality, near-perfect correlations")
print(f"   - May exclude some useful features")

print(f"\n4. **Features to investigate/fix:**")
print(f"   - {len(problematic)} features with r < 0.50")
print(f"   - Likely due to incorrect derivation logic or different ESPN methodology")
print(f"   - Consider excluding or fixing these features")

print(f"\n{'='*120}")

