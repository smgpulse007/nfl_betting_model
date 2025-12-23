"""
ESPN API Data Fetcher for Independent Features (Non-Vegas)

Fetches team stats, player stats, injuries, and context data
that are independent of Vegas betting lines.
"""
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Optional
import json

class ESPNIndependentData:
    """Fetch independent NFL data from ESPN API"""
    
    BASE_SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
    BASE_CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
    BASE_WEB = "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _get(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        """Make GET request with retries"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {url}")
                    print(f"Error: {e}")
                    return None
                time.sleep(0.5 * (attempt + 1))
        return None
    
    # ========================================================================
    # TEAM STATISTICS (Independent of Vegas)
    # ========================================================================
    
    def get_team_stats(self, team_id: int, season: int = 2024, season_type: int = 2) -> Optional[Dict]:
        """
        Get comprehensive team statistics
        
        Returns:
            - Offensive stats: points/game, yards/game, passing/rushing
            - Defensive stats: points allowed, yards allowed, sacks
            - Efficiency: red zone %, third down %, turnovers
        """
        url = f"{self.BASE_CORE}/seasons/{season}/types/{season_type}/teams/{team_id}/statistics"
        
        data = self._get(url)
        if not data:
            return None
        
        # Parse statistics
        stats = {}
        if 'splits' in data and 'categories' in data['splits']:
            for category in data['splits']['categories']:
                cat_name = category.get('name', '')
                for stat in category.get('stats', []):
                    stat_name = stat.get('name', '')
                    stat_value = stat.get('value', 0)
                    stats[f"{cat_name}_{stat_name}"] = stat_value
        
        return stats
    
    def get_team_record(self, team_id: int, season: int = 2024, season_type: int = 2) -> Optional[Dict]:
        """
        Get team record with splits
        
        Returns:
            - Overall record
            - Home/away splits
            - Division/conference record
            - Streak
        """
        url = f"{self.BASE_CORE}/seasons/{season}/types/{season_type}/teams/{team_id}/record"
        
        data = self._get(url)
        if not data:
            return None
        
        record = {}
        if 'items' in data:
            for item in data['items']:
                record_type = item.get('type', '')
                stats = item.get('stats', [])
                for stat in stats:
                    stat_name = stat.get('name', '')
                    stat_value = stat.get('value', 0)
                    record[f"{record_type}_{stat_name}"] = stat_value
        
        return record
    
    # ========================================================================
    # PLAYER STATISTICS (QB, RB, WR)
    # ========================================================================
    
    def get_team_roster(self, team_id: int) -> Optional[List[Dict]]:
        """
        Get team roster with player details
        
        Returns list of players with:
            - Position
            - Jersey number
            - Status (active/injured)
            - Experience
        """
        url = f"{self.BASE_SITE}/teams/{team_id}/roster"
        
        data = self._get(url)
        if not data or 'athletes' not in data:
            return None
        
        roster = []
        for athlete in data['athletes']:
            # Handle position - can be string or dict
            position = athlete.get('position')
            if isinstance(position, dict):
                position = position.get('abbreviation')

            # Handle status - can be string or dict
            status = athlete.get('status')
            if isinstance(status, dict):
                status = status.get('type')

            # Handle experience - can be int or dict
            experience = athlete.get('experience', 0)
            if isinstance(experience, dict):
                experience = experience.get('years', 0)

            player = {
                'id': athlete.get('id'),
                'name': athlete.get('displayName'),
                'position': position,
                'jersey': athlete.get('jersey'),
                'status': status,
                'experience': experience
            }
            roster.append(player)
        
        return roster
    
    def get_player_stats(self, player_id: int, season: int = 2024) -> Optional[Dict]:
        """
        Get player statistics for the season
        
        For QB: passing yards, TDs, INTs, completion %, rating
        For RB: rushing yards, yards/carry, TDs, receptions
        For WR: receptions, yards, TDs, targets
        """
        url = f"{self.BASE_WEB}/athletes/{player_id}/stats"
        params = {'season': season}
        
        data = self._get(url, params=params)
        if not data:
            return None
        
        stats = {}
        # Parse stats from response
        # (ESPN's structure varies, need to handle different formats)
        
        return stats
    
    # ========================================================================
    # GAME CONTEXT (Injuries, Weather, etc.)
    # ========================================================================
    
    def get_game_injuries(self, event_id: int) -> Optional[List[Dict]]:
        """
        Get injuries for a specific game
        
        Returns list of injured players with:
            - Team
            - Player name
            - Position
            - Status (out, questionable, doubtful)
        """
        url = f"{self.BASE_SITE}/summary"
        params = {'event': event_id}
        
        data = self._get(url, params=params)
        if not data or 'injuries' not in data:
            return None
        
        injuries = []
        for team_injuries in data['injuries']:
            team_id = team_injuries.get('team', {}).get('id')
            team_name = team_injuries.get('team', {}).get('abbreviation')
            
            for injury in team_injuries.get('injuries', []):
                inj_data = {
                    'team_id': team_id,
                    'team': team_name,
                    'player_id': injury.get('athlete', {}).get('id'),
                    'player_name': injury.get('athlete', {}).get('displayName'),
                    'position': injury.get('athlete', {}).get('position', {}).get('abbreviation'),
                    'status': injury.get('status'),
                    'details': injury.get('details', {}).get('detail')
                }
                injuries.append(inj_data)
        
        return injuries
    
    # ========================================================================
    # BATCH OPERATIONS
    # ========================================================================
    
    def get_all_team_stats_for_season(self, season: int = 2024, season_type: int = 2) -> pd.DataFrame:
        """
        Get stats for all 32 NFL teams
        
        Returns DataFrame with team stats
        """
        # Get all teams first
        teams_url = f"{self.BASE_SITE}/teams"
        teams_data = self._get(teams_url)
        
        if not teams_data or 'sports' not in teams_data:
            return pd.DataFrame()
        
        all_stats = []
        
        # Extract team IDs
        for sport in teams_data['sports']:
            for league in sport.get('leagues', []):
                for team in league.get('teams', []):
                    team_info = team.get('team', {})
                    team_id = team_info.get('id')
                    team_name = team_info.get('abbreviation')
                    
                    print(f"Fetching stats for {team_name}...")
                    
                    # Get team stats
                    stats = self.get_team_stats(team_id, season, season_type)
                    if stats:
                        stats['team_id'] = team_id
                        stats['team'] = team_name
                        stats['season'] = season
                        all_stats.append(stats)
                    
                    time.sleep(0.3)  # Rate limiting
        
        return pd.DataFrame(all_stats)


if __name__ == "__main__":
    # Test the API
    api = ESPNIndependentData()
    
    print("Testing ESPN Independent Data API...")
    print("="*80)
    
    # Test 1: Get team stats (Kansas City Chiefs = 12)
    print("\n[1] Testing team stats (KC Chiefs)...")
    stats = api.get_team_stats(team_id=12, season=2024)
    if stats:
        print(f"✅ Got {len(stats)} stats")
        print(f"Sample: {list(stats.keys())[:5]}")
    else:
        print("❌ Failed to get team stats")
    
    # Test 2: Get team record
    print("\n[2] Testing team record (KC Chiefs)...")
    record = api.get_team_record(team_id=12, season=2024)
    if record:
        print(f"✅ Got {len(record)} record stats")
        print(f"Sample: {list(record.keys())[:5]}")
    else:
        print("❌ Failed to get team record")
    
    # Test 3: Get roster
    print("\n[3] Testing roster (KC Chiefs)...")
    roster = api.get_team_roster(team_id=12)
    if roster:
        print(f"✅ Got {len(roster)} players")
        print(f"Sample: {roster[0]}")
    else:
        print("❌ Failed to get roster")
    
    print("\n" + "="*80)
    print("API test complete!")

