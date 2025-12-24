"""Check which ESPN features actually exist in the data"""
import pandas as pd

# Load ESPN data
espn_stats = pd.read_parquet('data/espn_raw/team_stats_2024.parquet')
espn_records = pd.read_parquet('data/espn_raw/team_records_2024.parquet')
espn = pd.merge(espn_stats, espn_records, on='team', how='outer')

print(f"ESPN total columns: {len(espn.columns)}")
print(f"ESPN teams: {len(espn)}")

# Load feature mapping
mapping = pd.read_csv('results/comprehensive_feature_mapping.csv')

# Check which mapped features exist in ESPN
espn_cols = set(espn.columns)
mapped_features = set(mapping['espn_feature'])

existing = mapped_features & espn_cols
missing = mapped_features - espn_cols

print(f"\nFeatures in mapping: {len(mapped_features)}")
print(f"Features that exist in ESPN: {len(existing)}")
print(f"Features missing from ESPN: {len(missing)}")

# Check by category
print("\n" + "="*80)
print("FEATURES BY CATEGORY")
print("="*80)

for category in ['EXACT MATCH', 'PARTIAL MATCH', 'CLOSE APPROXIMATION', 'CANNOT DERIVE']:
    cat_features = mapping[mapping['category'] == category]['espn_feature']
    cat_existing = set(cat_features) & espn_cols
    cat_missing = set(cat_features) - espn_cols
    
    print(f"\n{category}:")
    print(f"  Total: {len(cat_features)}")
    print(f"  Exist in ESPN: {len(cat_existing)} ({len(cat_existing)/len(cat_features)*100:.1f}%)")
    print(f"  Missing: {len(cat_missing)} ({len(cat_missing)/len(cat_features)*100:.1f}%)")
    
    if len(cat_missing) > 0 and len(cat_missing) <= 10:
        print(f"  Missing features: {sorted(cat_missing)}")

# Save list of features that exist
existing_df = mapping[mapping['espn_feature'].isin(espn_cols)].copy()
existing_df.to_csv('results/features_existing_in_espn.csv', index=False)
print(f"\n✅ Saved {len(existing_df)} existing features to: results/features_existing_in_espn.csv")

# Save list of features to derive
to_derive = existing_df[existing_df['category'].isin(['EXACT MATCH', 'PARTIAL MATCH', 'CLOSE APPROXIMATION'])].copy()
print(f"✅ Features to derive: {len(to_derive)}")
print(f"   - EXACT MATCH: {len(to_derive[to_derive['category']=='EXACT MATCH'])}")
print(f"   - PARTIAL MATCH: {len(to_derive[to_derive['category']=='PARTIAL MATCH'])}")
print(f"   - CLOSE APPROXIMATION: {len(to_derive[to_derive['category']=='CLOSE APPROXIMATION'])}")

