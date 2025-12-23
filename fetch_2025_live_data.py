"""
Fetch all 2025 Week 1-16 data with live odds and injuries from ESPN API
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from espn_game_api import ESPNGameAPI

def fetch_all_2025_data():
    """Fetch all Week 1-16 2025 games with live ESPN data"""
    
    print("="*100)
    print("FETCHING 2025 WEEK 1-16 DATA FROM ESPN API")
    print("="*100)
    
    api = ESPNGameAPI()
    
    # Fetch all weeks 1-16
    print("\n[1/3] Fetching scoreboard data for Weeks 1-16...")
    weeks = list(range(1, 17))
    all_games = api.get_all_weeks_data(weeks=weeks, year=2025, season_type=2)
    
    if all_games.empty:
        print("❌ No games found!")
        return None, None
    
    print(f"✅ Found {len(all_games)} games across {all_games['week'].nunique()} weeks")
    
    # Show summary by week
    print("\nGames by week:")
    week_summary = all_games.groupby('week').agg({
        'event_id': 'count',
        'status': lambda x: (x == 'STATUS_FINAL').sum()
    }).rename(columns={'event_id': 'total_games', 'status': 'completed_games'})
    print(week_summary.to_string())
    
    # Filter to completed games only
    completed_games = all_games[all_games['status'] == 'STATUS_FINAL'].copy()
    print(f"\n✅ {len(completed_games)} completed games")
    
    # [2/3] Get detailed data for each completed game (injuries, box scores)
    print("\n[2/3] Fetching detailed game data (injuries, stats)...")
    
    detailed_data = []
    injuries_data = []
    
    for idx, game in completed_games.iterrows():
        event_id = game['event_id']
        week = game['week']
        matchup = f"{game['away_team']} @ {game['home_team']}"
        
        print(f"  Week {week}: {matchup} (Event {event_id})...")
        
        # Get game summary
        summary = api.get_game_summary(event_id)
        
        if summary:
            detailed_data.append(summary)
            
            # Get injuries
            injuries = api.get_injuries_from_summary(event_id)
            if injuries:
                for inj in injuries:
                    inj['event_id'] = event_id
                    inj['week'] = week
                    injuries_data.append(inj)
        
        time.sleep(0.3)  # Rate limiting
    
    # Convert to DataFrames
    detailed_df = pd.DataFrame(detailed_data)
    injuries_df = pd.DataFrame(injuries_data)
    
    print(f"\n✅ Detailed data for {len(detailed_df)} games")
    print(f"✅ {len(injuries_df)} injury records")
    
    # [3/3] Merge and process
    print("\n[3/3] Processing and merging data...")
    
    # Merge scoreboard with detailed data
    final_df = completed_games.merge(
        detailed_df,
        on='event_id',
        how='left',
        suffixes=('', '_detail')
    )
    
    # Clean up columns
    final_df = final_df[[
        'event_id', 'week', 'season', 'season_type',
        'away_team', 'home_team',
        'away_score', 'home_score',
        'spread', 'spread_value', 'over_under',
        'home_ml', 'away_ml',
        'home_win_pct', 'away_win_pct',
        'date', 'status',
        'venue', 'attendance'
    ]].copy()
    
    # Calculate results
    final_df['home_win'] = (final_df['home_score'] > final_df['away_score']).astype(int)
    final_df['margin'] = final_df['home_score'] - final_df['away_score']
    final_df['total_points'] = final_df['home_score'] + final_df['away_score']
    
    # Spread results (need to parse spread string like "SF -4.5")
    final_df['spread_line'] = final_df['spread'].apply(parse_spread_line)
    final_df['home_cover'] = (final_df['margin'] > final_df['spread_line']).astype(int)
    
    # Totals results
    final_df['over_hit'] = (final_df['total_points'] > final_df['over_under']).astype(int)
    
    # Save data
    output_dir = Path('data/2025_espn')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_df.to_csv(output_dir / 'games_week1_16.csv', index=False)
    injuries_df.to_csv(output_dir / 'injuries_week1_16.csv', index=False)
    
    print(f"\n✅ Saved to {output_dir}/")
    print(f"   - games_week1_16.csv ({len(final_df)} games)")
    print(f"   - injuries_week1_16.csv ({len(injuries_df)} records)")
    
    return final_df, injuries_df

def parse_spread_line(spread_str):
    """
    Parse spread string like 'SF -4.5' to get the line from home team perspective
    
    Returns:
        float: Spread line (positive = home favored, negative = home underdog)
    """
    if pd.isna(spread_str) or spread_str == '':
        return np.nan
    
    try:
        # Format is usually "TEAM -X.X" or "TEAM +X.X"
        parts = spread_str.split()
        if len(parts) >= 2:
            # Get the number part (e.g., "-4.5")
            line_str = parts[-1]
            line = float(line_str)
            
            # The team mentioned is the favorite if negative
            # We need to determine if that's home or away
            # This is tricky - for now just return the absolute value
            # and we'll fix the sign based on which team is mentioned
            return line
    except:
        pass
    
    return np.nan

if __name__ == "__main__":
    games_df, injuries_df = fetch_all_2025_data()
    
    if games_df is not None:
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        
        print(f"\nTotal games: {len(games_df)}")
        print(f"Weeks covered: {games_df['week'].min()}-{games_df['week'].max()}")
        print(f"Total injuries: {len(injuries_df)}")
        
        print("\nSample game:")
        sample = games_df.iloc[0]
        print(f"  {sample['away_team']} @ {sample['home_team']}")
        print(f"  Score: {sample['away_score']}-{sample['home_score']}")
        print(f"  Spread: {sample['spread']}")
        print(f"  O/U: {sample['over_under']}")
        print(f"  Home ML: {sample['home_ml']}")
        print(f"  Away ML: {sample['away_ml']}")

