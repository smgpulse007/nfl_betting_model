"""
ESPN API Explorer - Test endpoints for alternative data sources
"""
import requests
import json
from datetime import datetime

# Base URLs
SITE_API = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CORE_API = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
WEB_API = "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl"
CDN_API = "https://cdn.espn.com/core/nfl"

# Team IDs (sample)
TEAM_IDS = {
    'KC': 12, 'BUF': 2, 'PHI': 21, 'SF': 25, 'DAL': 6, 'DET': 8,
    'BAL': 33, 'MIA': 15, 'CLE': 5, 'JAX': 30, 'CIN': 4, 'HOU': 34
}

def fetch_json(url, name=""):
    """Fetch JSON from URL with error handling"""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"✅ {name}: {len(str(data))} bytes")
        return data
    except Exception as e:
        print(f"❌ {name}: {e}")
        return None

def explore_injuries():
    """Get team injuries - potential edge source"""
    print("\n" + "="*60)
    print("🏥 TEAM INJURIES (Potential Edge)")
    print("="*60)
    
    for team, team_id in list(TEAM_IDS.items())[:3]:
        url = f"{CORE_API}/teams/{team_id}/injuries"
        data = fetch_json(url, f"{team} injuries")
        if data and 'items' in data:
            print(f"   {team}: {len(data['items'])} injury entries")
            for item in data['items'][:2]:
                if '$ref' in item:
                    # Fetch actual injury data
                    injury_data = fetch_json(item['$ref'], f"  └─ injury detail")
                    if injury_data:
                        athlete = injury_data.get('athlete', {})
                        status = injury_data.get('status', 'Unknown')
                        print(f"      Player: {athlete.get('displayName', 'N/A')} | Status: {status}")

def explore_depth_charts():
    """Get depth charts - roster changes"""
    print("\n" + "="*60)
    print("📋 DEPTH CHARTS (Roster Changes)")
    print("="*60)
    
    url = f"{CORE_API}/seasons/2025/teams/12/depthcharts"  # KC
    data = fetch_json(url, "KC depth chart")
    if data and 'items' in data:
        print(f"   Found {len(data['items'])} position groups")

def explore_odds_movement():
    """Get odds movement - sharp money indicator"""
    print("\n" + "="*60)
    print("📈 ODDS MOVEMENT (Sharp Money)")
    print("="*60)
    
    # First get current week's events
    scoreboard_url = f"{SITE_API}/scoreboard"
    scoreboard = fetch_json(scoreboard_url, "Scoreboard")
    
    if scoreboard and 'events' in scoreboard:
        event = scoreboard['events'][0]
        event_id = event['id']
        print(f"   Checking: {event['name']}")
        
        # Get odds
        odds_url = f"{CORE_API}/events/{event_id}/competitions/{event_id}/odds"
        odds = fetch_json(odds_url, "Game odds")
        if odds and 'items' in odds:
            for item in odds['items'][:2]:
                if '$ref' in item:
                    odds_detail = fetch_json(item['$ref'], "  └─ odds detail")
                    if odds_detail:
                        provider = odds_detail.get('provider', {}).get('name', 'Unknown')
                        spread = odds_detail.get('spread', 'N/A')
                        total = odds_detail.get('overUnder', 'N/A')
                        print(f"      {provider}: Spread={spread}, Total={total}")

def explore_player_stats():
    """Get player game logs - for feature engineering"""
    print("\n" + "="*60)
    print("🏈 PLAYER GAME LOGS (Feature Engineering)")
    print("="*60)
    
    # Patrick Mahomes
    athlete_id = 3139477
    url = f"{WEB_API}/athletes/{athlete_id}/gamelog"
    data = fetch_json(url, "Mahomes gamelog")
    if data:
        print(f"   Keys: {list(data.keys())}")

def explore_team_stats():
    """Get team statistics"""
    print("\n" + "="*60)
    print("📊 TEAM STATISTICS")
    print("="*60)
    
    url = f"{CORE_API}/seasons/2025/types/2/teams/12/statistics"  # KC regular season
    data = fetch_json(url, "KC team stats")
    if data and 'splits' in data:
        print(f"   Found {len(data['splits'].get('categories', []))} stat categories")

def explore_past_performances():
    """Get team betting history"""
    print("\n" + "="*60)
    print("🎰 TEAM BETTING HISTORY (ATS Performance)")
    print("="*60)
    
    # Provider 1002 = consensus
    url = f"{CORE_API}/teams/12/odds/1002/past-performances?limit=20"
    data = fetch_json(url, "KC past performances")
    if data and 'items' in data:
        print(f"   Found {len(data['items'])} past games with betting data")

def main():
    print("="*60)
    print("ESPN API EXPLORER - Finding Independent Edge")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    explore_injuries()
    explore_depth_charts()
    explore_odds_movement()
    explore_player_stats()
    explore_team_stats()
    explore_past_performances()
    
    print("\n" + "="*60)
    print("📝 SUMMARY: Key Data Sources for Edge")
    print("="*60)
    print("""
    1. INJURIES - Real-time injury status (Vegas may lag)
    2. DEPTH CHARTS - Roster changes, starter updates
    3. ODDS MOVEMENT - Track line moves for sharp money
    4. PLAYER STATS - Build custom metrics
    5. TEAM STATS - Advanced team metrics
    6. PAST PERFORMANCES - Historical ATS/O-U records
    """)

if __name__ == "__main__":
    main()

