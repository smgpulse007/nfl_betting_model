"""Check which features are missing from the derivation"""
import pandas as pd

# Load mapping
mapping = pd.read_csv('results/comprehensive_feature_mapping.csv')
to_derive = mapping[mapping['category'].isin(['EXACT MATCH', 'PARTIAL MATCH', 'CLOSE APPROXIMATION'])]

# Load derived features
derived = pd.read_parquet('data/derived_features/espn_derived_2024.parquet')

print(f"Expected features: {len(to_derive)}")
print(f"Derived features: {len(derived.columns)}")
print(f"Missing: {len(to_derive) - len(derived.columns)}")

# Find missing features
missing = set(to_derive['espn_feature']) - set(derived.columns)

print(f"\nMissing features ({len(missing)}):")

# Group by category
missing_df = to_derive[to_derive['espn_feature'].isin(missing)].copy()
print("\nBy category:")
print(missing_df['category'].value_counts())

print("\nFirst 50 missing features:")
for i, f in enumerate(sorted(missing)[:50], 1):
    cat = missing_df[missing_df['espn_feature'] == f]['category'].iloc[0] if len(missing_df[missing_df['espn_feature'] == f]) > 0 else 'UNKNOWN'
    print(f"  {i:2d}. {f:60s} [{cat}]")

