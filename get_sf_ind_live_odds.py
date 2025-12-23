"""
Get live odds for SF @ IND using ESPN's hidden API
"""
import requests
import json

def get_live_sf_ind_odds():
    print("="*100)
    print("SF @ IND LIVE ODDS FROM ESPN API")
    print("="*100)
    
    event_id = "401772824"  # From scoreboard
    
    print(f"\nEvent ID: {event_id}")
    print(f"Fetching data from ESPN's hidden API...")
    
    url = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/summary"
    params = {'event': event_id}
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        print(f"Status: {resp.status_code}\n")
        
        if resp.status_code == 200:
            data = resp.json()
            
            # Header info
            header = data.get('header', {})
            comp = header.get('competitions', [{}])[0]
            
            # Teams
            teams = comp.get('competitors', [])
            home = next((t for t in teams if t.get('homeAway') == 'home'), {})
            away = next((t for t in teams if t.get('homeAway') == 'away'), {})
            
            home_team = home.get('team', {}).get('abbreviation')
            away_team = away.get('team', {}).get('abbreviation')
            
            print(f"📅 GAME INFO:")
            print(f"  Matchup: {away_team} @ {home_team}")
            print(f"  Date: {comp.get('date')}")
            print(f"  Status: {comp.get('status', {}).get('type', {}).get('name')}")
            print(f"  Venue: {data.get('gameInfo', {}).get('venue', {}).get('fullName')}")
            
            # Pickcenter (odds)
            pickcenter = data.get('pickcenter', [])
            
            if pickcenter:
                print(f"\n📊 LIVE ODDS (from {len(pickcenter)} providers):")
                
                for i, provider_data in enumerate(pickcenter):
                    provider = provider_data.get('provider', {})
                    provider_name = provider.get('name', 'Unknown')
                    
                    print(f"\n  [{i+1}] {provider_name.upper()}:")
                    print(f"      Spread: {provider_data.get('details', 'N/A')}")
                    print(f"      O/U: {provider_data.get('overUnder', 'N/A')}")
                    
                    home_odds = provider_data.get('homeTeamOdds', {})
                    away_odds = provider_data.get('awayTeamOdds', {})
                    
                    home_ml = home_odds.get('moneyLine')
                    away_ml = away_odds.get('moneyLine')
                    
                    print(f"      {home_team} ML: {home_ml}")
                    print(f"      {away_team} ML: {away_ml}")
                    
                    # Convert moneyline to implied probability
                    if home_ml and away_ml:
                        if home_ml < 0:
                            home_prob = abs(home_ml) / (abs(home_ml) + 100)
                        else:
                            home_prob = 100 / (home_ml + 100)
                        
                        if away_ml < 0:
                            away_prob = abs(away_ml) / (abs(away_ml) + 100)
                        else:
                            away_prob = 100 / (away_ml + 100)
                        
                        print(f"      {home_team} Implied Prob: {home_prob:.1%}")
                        print(f"      {away_team} Implied Prob: {away_prob:.1%}")
                    
                    # Win percentages
                    home_win_pct = home_odds.get('winPercentage')
                    away_win_pct = away_odds.get('winPercentage')
                    
                    if home_win_pct:
                        print(f"      {home_team} Win %: {home_win_pct}%")
                    if away_win_pct:
                        print(f"      {away_team} Win %: {away_win_pct}%")
                    
                    # Spread details
                    home_spread = home_odds.get('current', {}).get('pointSpread', {})
                    away_spread = away_odds.get('current', {}).get('pointSpread', {})
                    
                    if home_spread:
                        print(f"      {home_team} Spread: {home_spread.get('american', 'N/A')}")
                    if away_spread:
                        print(f"      {away_team} Spread: {away_spread.get('american', 'N/A')}")
                
                # Summary
                print(f"\n" + "="*100)
                print("CONSENSUS ODDS (Provider #1):")
                print("="*100)
                
                consensus = pickcenter[0]
                spread_detail = consensus.get('details', '')
                over_under = consensus.get('overUnder', 'N/A')
                
                home_odds = consensus.get('homeTeamOdds', {})
                away_odds = consensus.get('awayTeamOdds', {})
                
                home_ml = home_odds.get('moneyLine')
                away_ml = away_odds.get('moneyLine')
                
                print(f"\nSpread: {spread_detail}")
                print(f"O/U: {over_under}")
                print(f"{home_team} ML: {home_ml}")
                print(f"{away_team} ML: {away_ml}")
                
                # Determine favorite
                if home_ml and away_ml:
                    if home_ml < away_ml:
                        favorite = home_team
                        underdog = away_team
                        fav_ml = home_ml
                        dog_ml = away_ml
                    else:
                        favorite = away_team
                        underdog = home_team
                        fav_ml = away_ml
                        dog_ml = home_ml
                    
                    print(f"\nFavorite: {favorite} ({fav_ml})")
                    print(f"Underdog: {underdog} (+{abs(dog_ml)})")
                
                # Compare to user's data
                print(f"\n" + "="*100)
                print("COMPARISON TO USER'S DATA:")
                print("="*100)
                
                print(f"\nUser said:")
                print(f"  SF ML: 0.69 (decimal odds)")
                print(f"  Spread: SF -3.5")
                print(f"  O/U: 45.5")
                
                print(f"\nESPN API shows:")
                print(f"  Spread: {spread_detail}")
                print(f"  O/U: {over_under}")
                print(f"  {home_team} ML: {home_ml}")
                print(f"  {away_team} ML: {away_ml}")
                
                # Convert decimal 0.69 to American
                # Decimal odds < 2.0 means favorite
                # American = -(100 / (decimal - 1))
                if 0.69 < 2.0:
                    american_from_decimal = -(100 / (0.69 - 1))
                    print(f"\nUser's SF ML 0.69 converts to: {american_from_decimal:.0f} (American)")
                
            else:
                print("\n❌ No odds data available in pickcenter")
            
            # Save full JSON for inspection
            with open('results/sf_ind_full_api_response.json', 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n✅ Full API response saved to: results/sf_ind_full_api_response.json")
            
        else:
            print(f"❌ Error: {resp.status_code}")
            print(resp.text[:500])
    
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    get_live_sf_ind_odds()

