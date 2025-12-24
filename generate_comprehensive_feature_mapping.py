"""
Generate comprehensive ESPN ↔ nfl-data-py feature mapping for ALL 283 ESPN features
"""
import pandas as pd
import json
from pathlib import Path

def load_espn_features():
    """Load ESPN features from collected data"""
    stats_df = pd.read_parquet('data/espn_raw/team_stats_2024.parquet')
    records_df = pd.read_parquet('data/espn_raw/team_records_2024.parquet')
    
    # Get all feature names
    stats_features = [col for col in stats_df.columns if col not in ['team_id', 'team', 'season', 'season_type']]
    records_features = [col for col in records_df.columns if col not in ['team_id', 'team', 'season', 'season_type']]
    
    return stats_features, records_features

def categorize_feature(feature_name):
    """Categorize ESPN feature and suggest nfl-data-py mapping"""
    
    # Extract category and metric
    if '_' in feature_name:
        category, metric = feature_name.split('_', 1)
    else:
        category = 'general'
        metric = feature_name
    
    # Define mapping rules
    mapping = {
        'category': 'UNKNOWN',
        'confidence': 'Unknown',
        'nfl_data_py_source': 'TBD',
        'derivation_formula': 'TBD',
        'years_available': 'TBD',
        'notes': ''
    }
    
    # PASSING FEATURES
    if category == 'passing':
        mapping['nfl_data_py_source'] = 'play-by-play'
        mapping['years_available'] = '1999-2024'
        
        if metric in ['passingYards', 'netPassingYards']:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp[pbp['play_type']=='pass']['yards_gained'].sum()"
            mapping['notes'] = 'Sum of yards_gained on pass plays'
        
        elif metric in ['passingAttempts', 'netPassingAttempts']:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['pass_attempt'].sum()"
            mapping['notes'] = 'Count of pass attempts'
        
        elif metric == 'completions':
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['complete_pass'].sum()"
            mapping['notes'] = 'Count of completed passes'
        
        elif metric in ['passingTouchdowns', 'totalTouchdowns']:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['pass_touchdown'].sum()"
            mapping['notes'] = 'Count of passing TDs'
        
        elif metric == 'interceptions':
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['interception'].sum()"
            mapping['notes'] = 'Count of interceptions thrown'
        
        elif metric == 'sacks':
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['sack'].sum()"
            mapping['notes'] = 'Count of sacks taken'
        
        elif metric == 'sackYardsLost':
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "abs(pbp[pbp['sack']==1]['yards_gained'].sum())"
            mapping['notes'] = 'Yards lost on sacks'
        
        elif metric == 'completionPct':
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['complete_pass'].sum() / pbp['pass_attempt'].sum() * 100"
            mapping['notes'] = 'Completion percentage'
        
        elif metric in ['passingYardsAfterCatch', 'yardsAfterCatch']:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp[pbp['complete_pass']==1]['yards_after_catch'].sum()"
            mapping['years_available'] = '2006-2024'
            mapping['notes'] = 'YAC available from 2006+'
        
        elif metric in ['passingYardsAtCatch', 'airYards']:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp[pbp['complete_pass']==1]['air_yards'].sum()"
            mapping['years_available'] = '2006-2024'
            mapping['notes'] = 'Air yards available from 2006+'
        
        elif metric in ['QBRating', 'passerRating']:
            mapping['category'] = 'CLOSE APPROXIMATION'
            mapping['confidence'] = 'Medium'
            mapping['derivation_formula'] = "Calculate from completions, attempts, yards, TDs, INTs using NFL formula"
            mapping['notes'] = 'Can calculate from basic stats'
        
        elif metric == 'ESPNQBRating':
            mapping['category'] = 'CANNOT DERIVE'
            mapping['confidence'] = 'N/A'
            mapping['derivation_formula'] = 'N/A - ESPN proprietary metric'
            mapping['years_available'] = '2024-2025 (ESPN only)'
            mapping['notes'] = 'ESPN QBR is proprietary'
        
        else:
            mapping['category'] = 'PARTIAL MATCH'
            mapping['confidence'] = 'Low'
            mapping['notes'] = f'Need to analyze: {metric}'
    
    # RUSHING FEATURES
    elif category == 'rushing':
        mapping['nfl_data_py_source'] = 'play-by-play'
        mapping['years_available'] = '1999-2024'
        
        if metric in ['rushingYards', 'netTotalYards']:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp[pbp['play_type']=='run']['yards_gained'].sum()"
            mapping['notes'] = 'Sum of yards_gained on rush plays'
        
        elif metric == 'rushingAttempts':
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['rush_attempt'].sum()"
            mapping['notes'] = 'Count of rush attempts'
        
        elif metric == 'rushingTouchdowns':
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['rush_touchdown'].sum()"
            mapping['notes'] = 'Count of rushing TDs'
        
        elif metric == 'ESPNRBRating':
            mapping['category'] = 'CANNOT DERIVE'
            mapping['confidence'] = 'N/A'
            mapping['derivation_formula'] = 'N/A - ESPN proprietary metric'
            mapping['years_available'] = '2024-2025 (ESPN only)'
            mapping['notes'] = 'ESPN RB Rating is proprietary'
        
        else:
            mapping['category'] = 'PARTIAL MATCH'
            mapping['confidence'] = 'Low'
            mapping['notes'] = f'Need to analyze: {metric}'
    
    # DEFENSIVE FEATURES
    elif category in ['defensive', 'defensiveInterceptions']:
        mapping['nfl_data_py_source'] = 'play-by-play'
        mapping['years_available'] = '1999-2024'
        
        if metric in ['totalSacks', 'sacks']:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp[pbp['defteam']==team]['sack'].sum()"
            mapping['notes'] = 'Sacks by defense'
        
        elif metric == 'interceptions':
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp[pbp['defteam']==team]['interception'].sum()"
            mapping['notes'] = 'Interceptions by defense'
        
        elif metric == 'hurries':
            mapping['category'] = 'CANNOT DERIVE'
            mapping['confidence'] = 'N/A'
            mapping['derivation_formula'] = 'N/A - Not in play-by-play'
            mapping['years_available'] = '2024-2025 (ESPN only)'
            mapping['notes'] = 'QB hurries not tracked in nfl-data-py'
        
        else:
            mapping['category'] = 'PARTIAL MATCH'
            mapping['confidence'] = 'Low'
            mapping['notes'] = f'Need to analyze: {metric}'
    
    # RECEIVING FEATURES
    elif category == 'receiving':
        mapping['nfl_data_py_source'] = 'play-by-play'
        mapping['years_available'] = '1999-2024'

        if metric in ['receivingYards', 'netTotalYards']:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp[pbp['complete_pass']==1]['yards_gained'].sum()"
            mapping['notes'] = 'Same as passing yards (team perspective)'

        elif metric in ['receptions', 'receivingReceptions']:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['complete_pass'].sum()"
            mapping['notes'] = 'Same as completions'

        elif metric in ['receivingTouchdowns', 'totalTouchdowns']:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['pass_touchdown'].sum()"
            mapping['notes'] = 'Same as passing TDs'

        elif metric == 'ESPNWRRating':
            mapping['category'] = 'CANNOT DERIVE'
            mapping['confidence'] = 'N/A'
            mapping['derivation_formula'] = 'N/A - ESPN proprietary metric'
            mapping['years_available'] = '2024-2025 (ESPN only)'
            mapping['notes'] = 'ESPN WR Rating is proprietary'

        else:
            mapping['category'] = 'PARTIAL MATCH'
            mapping['confidence'] = 'Low'
            mapping['notes'] = f'Need to analyze: {metric}'

    # KICKING/PUNTING/RETURNING FEATURES
    elif category in ['kicking', 'punting', 'returning']:
        mapping['nfl_data_py_source'] = 'play-by-play'
        mapping['years_available'] = '1999-2024'

        if 'fieldGoal' in metric or 'extraPoint' in metric:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp[pbp['field_goal_attempt']==1 or pbp['extra_point_attempt']==1]"
            mapping['notes'] = 'Can derive from play-by-play'

        elif 'punt' in metric.lower():
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp[pbp['punt_attempt']==1]"
            mapping['notes'] = 'Can derive from play-by-play'

        elif 'kickoff' in metric.lower() or 'return' in metric.lower():
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp[pbp['kickoff_attempt']==1 or pbp['return']==1]"
            mapping['notes'] = 'Can derive from play-by-play'

        else:
            mapping['category'] = 'PARTIAL MATCH'
            mapping['confidence'] = 'Low'
            mapping['notes'] = f'Need to analyze: {metric}'

    # SCORING FEATURES
    elif category == 'scoring':
        mapping['nfl_data_py_source'] = 'play-by-play'
        mapping['years_available'] = '1999-2024'
        mapping['category'] = 'EXACT MATCH'
        mapping['confidence'] = 'High'
        mapping['derivation_formula'] = "pbp['touchdown'].sum() or pbp['field_goal_result']=='made'"
        mapping['notes'] = 'Can derive from play-by-play scoring plays'

    # MISCELLANEOUS FEATURES (downs, possession, etc.)
    elif category == 'miscellaneous':
        mapping['nfl_data_py_source'] = 'play-by-play'
        mapping['years_available'] = '1999-2024'

        if 'firstDown' in metric:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['first_down'].sum()"
            mapping['notes'] = 'Can derive from play-by-play'

        elif 'thirdDown' in metric or 'fourthDown' in metric:
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp[pbp['down']==3 or pbp['down']==4]"
            mapping['notes'] = 'Can derive from down/distance data'

        elif 'possession' in metric.lower():
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp.groupby('posteam')['game_seconds_remaining'].diff()"
            mapping['notes'] = 'Can calculate from play timestamps'

        else:
            mapping['category'] = 'PARTIAL MATCH'
            mapping['confidence'] = 'Low'
            mapping['notes'] = f'Need to analyze: {metric}'

    # GENERAL FEATURES (fumbles, penalties, etc.)
    elif category == 'general':
        mapping['nfl_data_py_source'] = 'play-by-play'
        mapping['years_available'] = '1999-2024'

        if 'fumble' in metric.lower():
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['fumble'].sum() or pbp['fumble_lost'].sum()"
            mapping['notes'] = 'Can derive from play-by-play'

        elif 'penalt' in metric.lower():
            mapping['category'] = 'EXACT MATCH'
            mapping['confidence'] = 'High'
            mapping['derivation_formula'] = "pbp['penalty'].sum()"
            mapping['notes'] = 'Can derive from play-by-play'

        else:
            mapping['category'] = 'PARTIAL MATCH'
            mapping['confidence'] = 'Low'
            mapping['notes'] = f'Need to analyze: {metric}'

    # RECORD FEATURES
    elif category in ['total', 'home', 'road', 'vsdiv', 'vsconf']:
        mapping['nfl_data_py_source'] = 'schedules'
        mapping['years_available'] = '1999-2024'
        mapping['category'] = 'EXACT MATCH'
        mapping['confidence'] = 'High'
        mapping['derivation_formula'] = 'Calculate from schedule results'
        mapping['notes'] = 'Can derive from game results'

    return mapping

def generate_all_mappings():
    """Generate mappings for all ESPN features"""
    stats_features, records_features = load_espn_features()
    
    all_mappings = {}
    
    # Map stats features
    for feature in stats_features:
        all_mappings[feature] = categorize_feature(feature)
    
    # Map records features
    for feature in records_features:
        all_mappings[feature] = categorize_feature(feature)
    
    return all_mappings

def save_mappings(mappings):
    """Save mappings to CSV and JSON"""
    df = pd.DataFrame.from_dict(mappings, orient='index')
    df.index.name = 'espn_feature'
    df = df.reset_index()
    
    # Add feature type
    df['feature_type'] = df['espn_feature'].apply(lambda x: x.split('_')[0] if '_' in x else 'general')
    
    # Reorder columns
    df = df[['espn_feature', 'feature_type', 'category', 'confidence', 
             'nfl_data_py_source', 'derivation_formula', 'years_available', 'notes']]
    
    # Save to CSV
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / 'comprehensive_feature_mapping.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV to: {csv_path}")
    
    # Save to JSON
    json_path = output_dir / 'comprehensive_feature_mapping.json'
    with open(json_path, 'w') as f:
        json.dump(mappings, f, indent=2)
    print(f"✅ Saved JSON to: {json_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE FEATURE MAPPING SUMMARY")
    print("=" * 80)
    print(f"\nTotal Features Mapped: {len(df)}")
    print(f"\nBy Category:")
    print(df['category'].value_counts())
    print(f"\nBy Confidence:")
    print(df['confidence'].value_counts())
    print(f"\nBy Feature Type:")
    print(df['feature_type'].value_counts())
    
    return df

if __name__ == '__main__':
    print("Generating comprehensive feature mapping...")
    mappings = generate_all_mappings()
    df = save_mappings(mappings)

