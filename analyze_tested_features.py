"""Analyze which features were tested and their categories"""
import pandas as pd

# Load mapping
df = pd.read_csv('results/comprehensive_feature_mapping.csv')

# Features I tested
tested = [
    'passing_passingYards',
    'passing_passingAttempts', 
    'passing_completions',
    'passing_passingTouchdowns',
    'passing_interceptions',
    'passing_completionPct',
    'rushing_rushingYards',
    'rushing_rushingAttempts',
    'rushing_rushingTouchdowns',
    'total_wins',
    'total_pointsFor'
]

print("=" * 100)
print("ANALYSIS: Which Features Were Tested?")
print("=" * 100)

print(f"\nTotal ESPN features: {len(df)}")
print(f"Features tested: {len(tested)}")
print(f"Coverage: {len(tested)/len(df)*100:.1f}%")

print("\n" + "=" * 100)
print("TESTED FEATURES BY CATEGORY")
print("=" * 100)

for f in tested:
    row = df[df['espn_feature'] == f]
    if len(row) > 0:
        cat = row.iloc[0]['category']
        conf = row.iloc[0]['confidence']
        print(f"{f:<35} {cat:<20} {conf} confidence")
    else:
        print(f"{f:<35} NOT IN MAPPING")

print("\n" + "=" * 100)
print("CATEGORY BREAKDOWN")
print("=" * 100)

tested_df = df[df['espn_feature'].isin(tested)]
print("\nTested features by category:")
print(tested_df['category'].value_counts())

print("\nAll features by category:")
print(df['category'].value_counts())

print("\n" + "=" * 100)
print("KEY QUESTION: Are the 11 tested features representative?")
print("=" * 100)

exact_match_total = len(df[df['category'] == 'EXACT MATCH'])
exact_match_tested = len(tested_df[tested_df['category'] == 'EXACT MATCH'])

partial_match_total = len(df[df['category'] == 'PARTIAL MATCH'])
partial_match_tested = len(tested_df[tested_df['category'] == 'PARTIAL MATCH'])

print(f"\nEXACT MATCH features:")
print(f"  Total: {exact_match_total}")
print(f"  Tested: {exact_match_tested}")
print(f"  Coverage: {exact_match_tested/exact_match_total*100:.1f}%")

print(f"\nPARTIAL MATCH features:")
print(f"  Total: {partial_match_total}")
print(f"  Tested: {partial_match_tested}")
print(f"  Coverage: {partial_match_tested/partial_match_total*100:.1f}%")

print("\n" + "=" * 100)
print("CONCLUSION")
print("=" * 100)

if exact_match_tested < 20:
    print("\n⚠️  WARNING: Only tested a small sample of EXACT MATCH features!")
    print(f"   Should test at least 20-30 EXACT MATCH features to be confident.")
    print(f"   Currently tested: {exact_match_tested}/{exact_match_total}")
else:
    print("\n✅ Good coverage of EXACT MATCH features")

print("\n" + "=" * 100)

