"""
Create comprehensive ESPN ↔ nfl-data-py feature mapping table
"""
import pandas as pd
import json
from pathlib import Path

# Define feature mappings
FEATURE_MAPPINGS = {
    # PASSING STATS - EXACT MATCH (r > 0.95)
    'passing_passingYards': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp[pbp['play_type']=='pass']['yards_gained'].sum()",
        'years_available': '1999-2024',
        'notes': 'Sum of yards_gained on all pass plays'
    },
    'passing_passingAttempts': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp['pass_attempt'].sum()",
        'years_available': '1999-2024',
        'notes': 'Count of all pass attempts'
    },
    'passing_completions': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp['complete_pass'].sum()",
        'years_available': '1999-2024',
        'notes': 'Count of completed passes'
    },
    'passing_passingTouchdowns': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp['pass_touchdown'].sum()",
        'years_available': '1999-2024',
        'notes': 'Count of passing TDs'
    },
    'passing_interceptions': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp['interception'].sum()",
        'years_available': '1999-2024',
        'notes': 'Count of interceptions thrown'
    },
    'passing_sacks': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp['sack'].sum()",
        'years_available': '1999-2024',
        'notes': 'Count of sacks taken'
    },
    'passing_sackYardsLost': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp[pbp['sack']==1]['yards_gained'].sum()",
        'years_available': '1999-2024',
        'notes': 'Yards lost on sacks (negative yards_gained)'
    },
    'passing_completionPct': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp['complete_pass'].sum() / pbp['pass_attempt'].sum() * 100",
        'years_available': '1999-2024',
        'notes': 'Completion percentage'
    },
    'passing_passingYardsAfterCatch': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp[pbp['play_type']=='pass']['yards_after_catch'].sum()",
        'years_available': '2006-2024',
        'notes': 'YAC available from 2006+'
    },
    'passing_passingYardsAtCatch': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp[pbp['play_type']=='pass']['air_yards'].sum()",
        'years_available': '2006-2024',
        'notes': 'Air yards available from 2006+'
    },
    
    # RUSHING STATS - EXACT MATCH
    'rushing_rushingYards': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp[pbp['play_type']=='run']['yards_gained'].sum()",
        'years_available': '1999-2024',
        'notes': 'Sum of yards_gained on all rush plays'
    },
    'rushing_rushingAttempts': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp['rush_attempt'].sum()",
        'years_available': '1999-2024',
        'notes': 'Count of all rush attempts'
    },
    'rushing_rushingTouchdowns': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp['rush_touchdown'].sum()",
        'years_available': '1999-2024',
        'notes': 'Count of rushing TDs'
    },
    
    # RECEIVING STATS - EXACT MATCH
    'receiving_receivingYards': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp[pbp['complete_pass']==1]['yards_gained'].sum()",
        'years_available': '1999-2024',
        'notes': 'Same as passing yards (team perspective)'
    },
    'receiving_receivingTouchdowns': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp['pass_touchdown'].sum()",
        'years_available': '1999-2024',
        'notes': 'Same as passing TDs (team perspective)'
    },
    
    # DEFENSIVE STATS - EXACT MATCH
    'defensive_totalSacks': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp[pbp['defteam']==team]['sack'].sum()",
        'years_available': '1999-2024',
        'notes': 'Sacks by defense'
    },
    'defensiveInterceptions_interceptions': {
        'category': 'EXACT MATCH',
        'confidence': 'High',
        'nfl_data_py_source': 'play-by-play',
        'derivation_formula': "pbp[pbp['defteam']==team]['interception'].sum()",
        'years_available': '1999-2024',
        'notes': 'Interceptions by defense'
    },
    
    # TRULY NEW ESPN FEATURES (CANNOT DERIVE)
    'defensive_hurries': {
        'category': 'CANNOT DERIVE',
        'confidence': 'N/A',
        'nfl_data_py_source': 'None',
        'derivation_formula': 'N/A - Not available in nfl-data-py',
        'years_available': '2024-2025 (ESPN only)',
        'notes': 'QB hurries not tracked in play-by-play'
    },
    'defensive_qbHits': {
        'category': 'CANNOT DERIVE',
        'confidence': 'N/A',
        'nfl_data_py_source': 'PFR (partial)',
        'derivation_formula': 'PFR has times_hit but only for 2018+',
        'years_available': '2018-2024 (PFR), 2024-2025 (ESPN)',
        'notes': 'PFR has QB hits from 2018+, but not team-level'
    },
}

# Add more mappings...
# (This would continue for all 327 ESPN features)

def create_mapping_table():
    """Create mapping table from dictionary"""
    df = pd.DataFrame.from_dict(FEATURE_MAPPINGS, orient='index')
    df.index.name = 'espn_feature'
    df = df.reset_index()
    
    # Add feature type
    df['feature_type'] = df['espn_feature'].apply(lambda x: x.split('_')[0] if '_' in x else 'general')
    
    # Reorder columns
    df = df[['espn_feature', 'feature_type', 'category', 'confidence', 
             'nfl_data_py_source', 'derivation_formula', 'years_available', 'notes']]
    
    return df

def save_mapping_table(df):
    """Save mapping table to CSV and Markdown"""
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Save to CSV
    csv_path = output_dir / 'espn_nfl_data_py_feature_mapping.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")
    
    # Save to Markdown
    md_path = output_dir / 'espn_nfl_data_py_feature_mapping_table.md'
    with open(md_path, 'w') as f:
        f.write("# ESPN ↔ nfl-data-py Feature Mapping Table\n\n")
        f.write(df.to_markdown(index=False))
    print(f"Saved Markdown to: {md_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("MAPPING SUMMARY")
    print("=" * 80)
    print(f"\nTotal Features Mapped: {len(df)}")
    print(f"\nBy Category:")
    print(df['category'].value_counts())
    print(f"\nBy Confidence:")
    print(df['confidence'].value_counts())

if __name__ == '__main__':
    df = create_mapping_table()
    save_mapping_table(df)

