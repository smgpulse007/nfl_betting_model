"""
Test ESPN API with current week data
"""
import requests
import json

def test_espn_api():
    print("="*100)
    print("TESTING ESPN HIDDEN API")
    print("="*100)
    
    # Test 1: Current scoreboard (no year specified)
    print("\n[1/4] Testing current scoreboard...")
    url1 = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    
    try:
        resp = requests.get(url1, timeout=10)
        print(f"  Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Week: {data.get('week', {}).get('number')}")
            print(f"  Season: {data.get('season', {}).get('year')}")
            print(f"  Events: {len(data.get('events', []))}")
            
            # Show first game
            if data.get('events'):
                event = data['events'][0]
                comp = event.get('competitions', [{}])[0]
                teams = comp.get('competitors', [])
                
                home = next((t for t in teams if t.get('homeAway') == 'home'), {})
                away = next((t for t in teams if t.get('homeAway') == 'away'), {})
                
                print(f"\n  Sample Game:")
                print(f"    {away.get('team', {}).get('abbreviation')} @ {home.get('team', {}).get('abbreviation')}")
                print(f"    Event ID: {event.get('id')}")
                print(f"    Date: {event.get('date')}")
                
                # Check for odds
                odds = comp.get('odds', [])
                if odds:
                    print(f"    Spread: {odds[0].get('details')}")
                    print(f"    O/U: {odds[0].get('overUnder')}")
        else:
            print(f"  Error: {resp.text[:200]}")
    except Exception as e:
        print(f"  Exception: {e}")
    
    # Test 2: Try 2024 season
    print("\n[2/4] Testing 2024 season, week 18...")
    url2 = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    params = {'season': 2024, 'seasontype': 2, 'week': 18}
    
    try:
        resp = requests.get(url2, params=params, timeout=10)
        print(f"  Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Events: {len(data.get('events', []))}")
            
            if data.get('events'):
                event_id = data['events'][0]['id']
                print(f"  First Event ID: {event_id}")
                
                # Test 3: Get game summary
                print(f"\n[3/4] Testing game summary for event {event_id}...")
                url3 = f"https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/summary"
                params3 = {'event': event_id}
                
                resp3 = requests.get(url3, params=params3, timeout=10)
                print(f"  Status: {resp3.status_code}")
                
                if resp3.status_code == 200:
                    summary = resp3.json()
                    
                    # Check for pickcenter (odds)
                    pickcenter = summary.get('pickcenter', [])
                    if pickcenter:
                        print(f"  Pickcenter providers: {len(pickcenter)}")
                        consensus = pickcenter[0]
                        print(f"  Spread: {consensus.get('details')}")
                        print(f"  O/U: {consensus.get('overUnder')}")
                        print(f"  Home ML: {consensus.get('homeTeamOdds', {}).get('moneyLine')}")
                        print(f"  Away ML: {consensus.get('awayTeamOdds', {}).get('moneyLine')}")
                    
                    # Check header
                    header = summary.get('header', {})
                    comp = header.get('competitions', [{}])[0]
                    teams = comp.get('competitors', [])
                    
                    for team in teams:
                        prefix = team.get('homeAway')
                        abbr = team.get('team', {}).get('abbreviation')
                        print(f"  {prefix}: {abbr}")
        else:
            print(f"  Error: {resp.text[:200]}")
    except Exception as e:
        print(f"  Exception: {e}")
    
    # Test 4: Try to find SF @ IND in Week 16 2024
    print("\n[4/4] Searching for SF @ IND in Week 16 2024...")
    url4 = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    params4 = {'season': 2024, 'seasontype': 2, 'week': 16}
    
    try:
        resp = requests.get(url4, params=params4, timeout=10)
        print(f"  Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            events = data.get('events', [])
            print(f"  Total games in Week 16 2024: {len(events)}")
            
            # Find SF @ IND
            for event in events:
                comp = event.get('competitions', [{}])[0]
                teams = comp.get('competitors', [])
                
                home = next((t for t in teams if t.get('homeAway') == 'home'), {})
                away = next((t for t in teams if t.get('homeAway') == 'away'), {})
                
                home_abbr = home.get('team', {}).get('abbreviation')
                away_abbr = away.get('team', {}).get('abbreviation')
                
                if (home_abbr == 'IND' and away_abbr == 'SF') or (home_abbr == 'SF' and away_abbr == 'IND'):
                    print(f"\n  ✅ FOUND: {away_abbr} @ {home_abbr}")
                    print(f"  Event ID: {event.get('id')}")
                    print(f"  Date: {event.get('date')}")
                    print(f"  Status: {comp.get('status', {}).get('type', {}).get('name')}")
                    
                    # Get odds
                    odds = comp.get('odds', [])
                    if odds:
                        print(f"  Spread: {odds[0].get('details')}")
                        print(f"  O/U: {odds[0].get('overUnder')}")
                        print(f"  Home ML: {odds[0].get('homeTeamOdds', {}).get('moneyLine')}")
                        print(f"  Away ML: {odds[0].get('awayTeamOdds', {}).get('moneyLine')}")
                    
                    # Now get full summary
                    event_id = event.get('id')
                    print(f"\n  Getting full summary...")
                    url_summary = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/summary"
                    resp_summary = requests.get(url_summary, params={'event': event_id}, timeout=10)
                    
                    if resp_summary.status_code == 200:
                        summary = resp_summary.json()
                        pickcenter = summary.get('pickcenter', [])
                        
                        if pickcenter:
                            print(f"\n  📊 PICKCENTER ODDS:")
                            for provider in pickcenter[:3]:  # Show first 3 providers
                                provider_name = provider.get('provider', {}).get('name', 'Unknown')
                                print(f"\n    Provider: {provider_name}")
                                print(f"    Spread: {provider.get('details')}")
                                print(f"    O/U: {provider.get('overUnder')}")
                                
                                home_odds = provider.get('homeTeamOdds', {})
                                away_odds = provider.get('awayTeamOdds', {})
                                
                                print(f"    Home ML: {home_odds.get('moneyLine')}")
                                print(f"    Away ML: {away_odds.get('moneyLine')}")
                                print(f"    Home Win %: {home_odds.get('winPercentage')}")
                                print(f"    Away Win %: {away_odds.get('winPercentage')}")
                    
                    break
        else:
            print(f"  Error: {resp.text[:200]}")
    except Exception as e:
        print(f"  Exception: {e}")

if __name__ == "__main__":
    test_espn_api()

