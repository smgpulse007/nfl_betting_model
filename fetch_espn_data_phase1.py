"""
Phase 1 Data Collection Script
Fetch all ESPN data needed for model enhancement
"""
import requests
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Optional
import json

class ESPNDataFetcher:
    """Simplified ESPN data fetcher for Phase 1"""
    
    BASE_SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
    BASE_CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make GET request with retries"""
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == 2:
                    print(f"  ❌ Failed: {e}")
                    return None
                time.sleep(0.5)
        return None
    
    def get_all_teams(self) -> List[Dict]:
        """Get all 32 NFL teams"""
        url = f"{self.BASE_SITE}/teams"
        data = self._get(url)
        
        if not data or 'sports' not in data:
            return []
        
        teams = []
        for sport in data['sports']:
            for league in sport.get('leagues', []):
                for team in league.get('teams', []):
                    team_info = team.get('team', {})
                    teams.append({
                        'team_id': team_info.get('id'),
                        'abbreviation': team_info.get('abbreviation'),
                        'display_name': team_info.get('displayName')
                    })
        return teams
    
    def get_team_stats(self, team_id: int, season: int = 2024) -> Optional[Dict]:
        """Get team statistics"""
        url = f"{self.BASE_CORE}/seasons/{season}/types/2/teams/{team_id}/statistics"
        data = self._get(url)
        
        if not data:
            return None
        
        stats = {}
        if 'splits' in data and 'categories' in data['splits']:
            for category in data['splits']['categories']:
                cat_name = category.get('name', '')
                for stat in category.get('stats', []):
                    stat_name = stat.get('name', '')
                    stat_value = stat.get('value', 0)
                    stats[f"{cat_name}_{stat_name}"] = stat_value
        
        return stats
    
    def get_team_record(self, team_id: int, season: int = 2024) -> Optional[Dict]:
        """Get team record"""
        url = f"{self.BASE_CORE}/seasons/{season}/types/2/teams/{team_id}/record"
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
    
    def fetch_all_team_stats(self, season: int, output_dir: Path):
        """Fetch stats for all teams"""
        teams = self.get_all_teams()
        print(f"\n📊 Fetching stats for {len(teams)} teams (Season {season})...")
        
        all_stats = []
        for i, team in enumerate(teams, 1):
            team_id = team['team_id']
            team_name = team['abbreviation']
            
            print(f"  [{i}/{len(teams)}] {team_name}...", end=' ')
            
            stats = self.get_team_stats(team_id, season)
            if stats:
                stats['team_id'] = team_id
                stats['team'] = team_name
                stats['season'] = season
                all_stats.append(stats)
                print(f"✅ {len(stats)} stats")
            else:
                print("❌ Failed")
            
            time.sleep(0.3)
        
        df = pd.DataFrame(all_stats)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"team_stats_{season}.parquet"
        df.to_parquet(save_path, index=False)
        print(f"\n💾 Saved to {save_path}")
        return df
    
    def fetch_all_team_records(self, season: int, output_dir: Path):
        """Fetch records for all teams"""
        teams = self.get_all_teams()
        print(f"\n📊 Fetching records for {len(teams)} teams (Season {season})...")
        
        all_records = []
        for i, team in enumerate(teams, 1):
            team_id = team['team_id']
            team_name = team['abbreviation']
            
            print(f"  [{i}/{len(teams)}] {team_name}...", end=' ')
            
            record = self.get_team_record(team_id, season)
            if record:
                record['team_id'] = team_id
                record['team'] = team_name
                record['season'] = season
                all_records.append(record)
                print(f"✅ {len(record)} stats")
            else:
                print("❌ Failed")
            
            time.sleep(0.3)
        
        df = pd.DataFrame(all_records)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"team_records_{season}.parquet"
        df.to_parquet(save_path, index=False)
        print(f"\n💾 Saved to {save_path}")
        return df


if __name__ == "__main__":
    print("="*80)
    print("🚀 Phase 1 Data Collection - ESPN API")
    print("="*80)
    
    fetcher = ESPNDataFetcher()
    output_dir = Path('data/espn_raw')
    
    # Fetch 2024 data
    print("\n[1/4] Fetching 2024 team stats...")
    stats_2024 = fetcher.fetch_all_team_stats(2024, output_dir)
    print(f"✅ Got {len(stats_2024)} teams, {len(stats_2024.columns)} columns")
    
    print("\n[2/4] Fetching 2024 team records...")
    records_2024 = fetcher.fetch_all_team_records(2024, output_dir)
    print(f"✅ Got {len(records_2024)} teams, {len(records_2024.columns)} columns")
    
    # Fetch 2025 data
    print("\n[3/4] Fetching 2025 team stats...")
    stats_2025 = fetcher.fetch_all_team_stats(2025, output_dir)
    print(f"✅ Got {len(stats_2025)} teams, {len(stats_2025.columns)} columns")
    
    print("\n[4/4] Fetching 2025 team records...")
    records_2025 = fetcher.fetch_all_team_records(2025, output_dir)
    print(f"✅ Got {len(records_2025)} teams, {len(records_2025.columns)} columns")
    
    print("\n" + "="*80)
    print("✅ Phase 1 Data Collection Complete!")
    print(f"📁 Data saved to: {output_dir}")
    print("="*80)

