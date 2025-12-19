"""
ESPN Data Fetcher - Build independent features from ESPN API
"""
import requests
import pandas as pd
from datetime import datetime
import time

class ESPNDataFetcher:
    """Fetch data from ESPN's unofficial API for feature engineering"""
    
    CORE_API = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
    SITE_API = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
    
    # All 32 NFL teams
    TEAMS = {
        'ARI': 22, 'ATL': 1, 'BAL': 33, 'BUF': 2, 'CAR': 29, 'CHI': 3,
        'CIN': 4, 'CLE': 5, 'DAL': 6, 'DEN': 7, 'DET': 8, 'GB': 9,
        'HOU': 34, 'IND': 11, 'JAX': 30, 'KC': 12, 'LV': 13, 'LAC': 24,
        'LA': 14, 'MIA': 15, 'MIN': 16, 'NE': 17, 'NO': 18, 'NYG': 19,
        'NYJ': 20, 'PHI': 21, 'PIT': 23, 'SF': 25, 'SEA': 26, 'TB': 27,
        'TEN': 10, 'WAS': 28
    }
    
    def __init__(self):
        self.session = requests.Session()
        
    def _get(self, url):
        """Make GET request with retry logic"""
        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt == 2:
                    print(f"Failed after 3 attempts: {url[:80]}...")
                    return None
                time.sleep(1)
    
    def get_team_injuries(self, team_abbr):
        """Get current injuries for a team"""
        team_id = self.TEAMS.get(team_abbr)
        if not team_id:
            return []
        
        url = f"{self.CORE_API}/teams/{team_id}/injuries"
        data = self._get(url)
        
        injuries = []
        if data and 'items' in data:
            for item in data['items']:
                if '$ref' in item:
                    detail = self._get(item['$ref'])
                    if detail:
                        # Fetch athlete name from reference if needed
                        athlete_info = detail.get('athlete', {})
                        player_name = athlete_info.get('displayName')
                        if not player_name and '$ref' in athlete_info:
                            athlete_data = self._get(athlete_info['$ref'])
                            if athlete_data:
                                player_name = athlete_data.get('displayName')

                        injuries.append({
                            'team': team_abbr,
                            'player_id': athlete_info.get('id'),
                            'player_name': player_name,
                            'status': detail.get('status'),
                            'type': detail.get('type', {}).get('description'),
                            'detail': detail.get('details', {}).get('detail')
                        })
        return injuries
    
    def get_all_injuries(self):
        """Get injuries for all teams"""
        all_injuries = []
        for team in self.TEAMS:
            print(f"Fetching injuries for {team}...")
            injuries = self.get_team_injuries(team)
            all_injuries.extend(injuries)
            time.sleep(0.2)  # Rate limiting
        return pd.DataFrame(all_injuries)
    
    def get_current_odds(self):
        """Get current week's odds for all games"""
        url = f"{self.SITE_API}/scoreboard"
        data = self._get(url)
        
        games = []
        if data and 'events' in data:
            for event in data['events']:
                game = {
                    'event_id': event['id'],
                    'name': event['name'],
                    'date': event['date'],
                    'status': event['status']['type']['name']
                }
                
                # Get odds
                comp = event.get('competitions', [{}])[0]
                odds = comp.get('odds', [{}])
                if odds:
                    game['spread'] = odds[0].get('details')
                    game['over_under'] = odds[0].get('overUnder')
                    game['home_ml'] = odds[0].get('homeTeamOdds', {}).get('moneyLine')
                    game['away_ml'] = odds[0].get('awayTeamOdds', {}).get('moneyLine')
                
                # Get teams
                for comp_team in comp.get('competitors', []):
                    if comp_team.get('homeAway') == 'home':
                        game['home_team'] = comp_team.get('team', {}).get('abbreviation')
                    else:
                        game['away_team'] = comp_team.get('team', {}).get('abbreviation')
                
                games.append(game)
        
        return pd.DataFrame(games)
    
    def get_team_ats_record(self, team_abbr, year=2025):
        """Get team's against-the-spread record"""
        team_id = self.TEAMS.get(team_abbr)
        if not team_id:
            return None
        
        url = f"{self.CORE_API}/seasons/{year}/types/2/teams/{team_id}/ats"
        data = self._get(url)
        
        if data:
            return {
                'team': team_abbr,
                'ats_wins': data.get('wins'),
                'ats_losses': data.get('losses'),
                'ats_pushes': data.get('pushes'),
                'ats_pct': data.get('wins', 0) / max(1, data.get('wins', 0) + data.get('losses', 0))
            }
        return None
    
    def get_all_ats_records(self, year=2025):
        """Get ATS records for all teams"""
        records = []
        for team in self.TEAMS:
            print(f"Fetching ATS for {team}...")
            record = self.get_team_ats_record(team, year)
            if record:
                records.append(record)
            time.sleep(0.1)
        return pd.DataFrame(records)


if __name__ == "__main__":
    fetcher = ESPNDataFetcher()
    
    print("="*60)
    print("ESPN DATA FETCHER - Current Week Data")
    print("="*60)
    
    # Get current odds
    print("\n📊 Current Week Odds:")
    odds_df = fetcher.get_current_odds()
    print(odds_df[['away_team', 'home_team', 'spread', 'over_under', 'status']].to_string())
    
    # Get sample injuries
    print("\n🏥 Sample Injuries (KC):")
    injuries = fetcher.get_team_injuries('KC')
    for inj in injuries[:5]:
        print(f"  {inj['player_name']}: {inj['status']} - {inj['type']}")

