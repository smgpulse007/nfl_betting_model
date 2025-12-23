"""
ESPN Game API - Direct access to ESPN's hidden API for game-specific data
Based on: https://scrapecreators.com/blog/espn-api-free-sports-data

This uses ESPN's site.web.api.espn.com endpoint which provides:
- Live odds (spread, moneyline, over/under)
- Box scores and stats
- Play-by-play data
- Injury reports
- Team records
"""
import requests
import pandas as pd
from datetime import datetime
import time

class ESPNGameAPI:
    """Access ESPN's hidden game API for detailed game data"""
    
    # The hidden API endpoint
    BASE_URL = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _get(self, url, params=None):
        """Make GET request with retry logic"""
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt == 2:
                    print(f"Failed after 3 attempts: {e}")
                    return None
                time.sleep(1)
        return None
    
    def get_game_summary(self, event_id):
        """
        Get complete game summary including odds, stats, injuries
        
        Args:
            event_id: ESPN event ID (e.g., '401671733')
        
        Returns:
            dict with game data
        """
        url = f"{self.BASE_URL}/summary"
        params = {
            'region': 'us',
            'lang': 'en',
            'contentorigin': 'espn',
            'event': event_id
        }
        
        data = self._get(url, params)
        if not data:
            return None
        
        # Extract key information
        game_info = {}
        
        # Basic game info
        header = data.get('header', {})
        game_info['event_id'] = event_id
        game_info['game_date'] = header.get('competitions', [{}])[0].get('date')
        game_info['status'] = header.get('competitions', [{}])[0].get('status', {}).get('type', {}).get('name')
        
        # Teams
        competitions = header.get('competitions', [{}])[0]
        for team in competitions.get('competitors', []):
            prefix = 'home' if team.get('homeAway') == 'home' else 'away'
            game_info[f'{prefix}_team'] = team.get('team', {}).get('abbreviation')
            game_info[f'{prefix}_score'] = team.get('score')
            game_info[f'{prefix}_record'] = team.get('records', [{}])[0].get('summary')
        
        # Odds from pickcenter
        pickcenter = data.get('pickcenter', [])
        if pickcenter:
            # Get consensus odds (first provider)
            consensus = pickcenter[0]
            game_info['spread'] = consensus.get('details', '')
            game_info['spread_value'] = consensus.get('spread')
            game_info['over_under'] = consensus.get('overUnder')
            
            # Get moneylines
            home_odds = consensus.get('homeTeamOdds', {})
            away_odds = consensus.get('awayTeamOdds', {})
            
            game_info['home_ml'] = home_odds.get('moneyLine')
            game_info['away_ml'] = away_odds.get('moneyLine')
            game_info['home_win_pct'] = home_odds.get('winPercentage')
            game_info['away_win_pct'] = away_odds.get('winPercentage')
            
            # Get spread odds
            home_spread = home_odds.get('current', {}).get('pointSpread', {})
            away_spread = away_odds.get('current', {}).get('pointSpread', {})
            game_info['home_spread'] = home_spread.get('american')
            game_info['away_spread'] = away_spread.get('american')
        
        # Box score stats
        boxscore = data.get('boxscore', {})
        if boxscore and 'teams' in boxscore:
            for team_data in boxscore['teams']:
                prefix = 'home' if team_data.get('homeAway') == 'home' else 'away'
                stats = {s['name']: s.get('displayValue') for s in team_data.get('statistics', [])}
                game_info[f'{prefix}_stats'] = stats
        
        # Game info (venue, attendance, etc.)
        game_details = data.get('gameInfo', {})
        venue = game_details.get('venue', {})
        game_info['venue'] = venue.get('fullName')
        game_info['attendance'] = game_details.get('attendance')
        
        return game_info
    
    def get_scoreboard(self, week=None, year=2025, season_type=2):
        """
        Get scoreboard for a specific week
        
        Args:
            week: Week number (None for current week)
            year: Season year
            season_type: 1=preseason, 2=regular, 3=postseason
        
        Returns:
            DataFrame with all games
        """
        url = f"{self.BASE_URL}/scoreboard"
        params = {
            'region': 'us',
            'lang': 'en',
            'contentorigin': 'espn',
            'seasontype': season_type,
            'season': year
        }
        
        if week:
            params['week'] = week
        
        data = self._get(url, params)
        if not data or 'events' not in data:
            return pd.DataFrame()
        
        games = []
        for event in data['events']:
            game = {
                'event_id': event['id'],
                'name': event['name'],
                'short_name': event['shortName'],
                'date': event['date'],
                'week': data.get('week', {}).get('number'),
                'season': year,
                'season_type': season_type
            }
            
            comp = event.get('competitions', [{}])[0]
            
            # Teams and scores
            for team in comp.get('competitors', []):
                prefix = 'home' if team.get('homeAway') == 'home' else 'away'
                game[f'{prefix}_team'] = team.get('team', {}).get('abbreviation')
                game[f'{prefix}_score'] = team.get('score')
                game[f'{prefix}_record'] = team.get('records', [{}])[0].get('summary') if team.get('records') else None
            
            # Odds
            odds = comp.get('odds', [])
            if odds:
                game['spread'] = odds[0].get('details')
                game['over_under'] = odds[0].get('overUnder')
                
                home_odds = odds[0].get('homeTeamOdds', {})
                away_odds = odds[0].get('awayTeamOdds', {})
                
                game['home_ml'] = home_odds.get('moneyLine')
                game['away_ml'] = away_odds.get('moneyLine')
            
            # Status
            status = comp.get('status', {})
            game['status'] = status.get('type', {}).get('name')
            game['status_detail'] = status.get('type', {}).get('detail')
            
            games.append(game)
        
        return pd.DataFrame(games)
    
    def get_event_id_from_teams(self, home_team, away_team, week, year=2025):
        """
        Find event ID for a specific matchup
        
        Args:
            home_team: Home team abbreviation (e.g., 'IND')
            away_team: Away team abbreviation (e.g., 'SF')
            week: Week number
            year: Season year
        
        Returns:
            event_id or None
        """
        scoreboard = self.get_scoreboard(week=week, year=year)
        if scoreboard.empty:
            return None
        
        match = scoreboard[
            (scoreboard['home_team'] == home_team) & 
            (scoreboard['away_team'] == away_team)
        ]
        
        if not match.empty:
            return match.iloc[0]['event_id']
        
        return None

    def get_all_weeks_data(self, weeks, year=2025, season_type=2):
        """
        Get scoreboard data for multiple weeks

        Args:
            weeks: List of week numbers
            year: Season year
            season_type: 1=preseason, 2=regular, 3=postseason

        Returns:
            DataFrame with all games from all weeks
        """
        all_games = []

        for week in weeks:
            print(f"Fetching Week {week}...")
            scoreboard = self.get_scoreboard(week=week, year=year, season_type=season_type)
            if not scoreboard.empty:
                all_games.append(scoreboard)
            time.sleep(0.5)  # Rate limiting

        if all_games:
            return pd.concat(all_games, ignore_index=True)
        return pd.DataFrame()

    def get_injuries_from_summary(self, event_id):
        """
        Extract injury data from game summary

        Args:
            event_id: ESPN event ID

        Returns:
            List of injury dicts
        """
        url = f"{self.BASE_URL}/summary"
        params = {
            'region': 'us',
            'lang': 'en',
            'contentorigin': 'espn',
            'event': event_id
        }

        data = self._get(url, params)
        if not data:
            return []

        injuries = []

        # Injuries are in the 'injuries' key
        injury_data = data.get('injuries', [])

        for team_injuries in injury_data:
            team_abbr = team_injuries.get('team', {}).get('abbreviation')

            for injury in team_injuries.get('injuries', []):
                athlete = injury.get('athlete', {})
                injuries.append({
                    'team': team_abbr,
                    'player_name': athlete.get('displayName'),
                    'player_id': athlete.get('id'),
                    'position': athlete.get('position', {}).get('abbreviation'),
                    'status': injury.get('status'),
                    'injury_type': injury.get('type'),
                    'details': injury.get('details', {}).get('detail')
                })

        return injuries

if __name__ == "__main__":
    api = ESPNGameAPI()

    print("="*100)
    print("ESPN GAME API - Testing Direct API Access")
    print("="*100)

    # Test 1: Get current week scoreboard
    print("\n[1/3] Current Week Scoreboard:")
    scoreboard = api.get_scoreboard()
    if not scoreboard.empty:
        print(scoreboard[['away_team', 'home_team', 'spread', 'over_under', 'status']].head(10).to_string())

    # Test 2: Get SF @ IND game details
    print("\n[2/3] Finding SF @ IND game...")
    event_id = api.get_event_id_from_teams('IND', 'SF', week=16, year=2025)

    if event_id:
        print(f"  Event ID: {event_id}")
        print(f"\n[3/3] Getting detailed game data...")
        game_data = api.get_game_summary(event_id)

        if game_data:
            print(f"\n  Game: {game_data.get('away_team')} @ {game_data.get('home_team')}")
            print(f"  Date: {game_data.get('game_date')}")
            print(f"  Status: {game_data.get('status')}")
            print(f"  Spread: {game_data.get('spread')}")
            print(f"  O/U: {game_data.get('over_under')}")
            print(f"  Home ML: {game_data.get('home_ml')}")
            print(f"  Away ML: {game_data.get('away_ml')}")
            print(f"  Venue: {game_data.get('venue')}")
    else:
        print("  Game not found")

