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

    def get_all_teams(self) -> List[Dict]:
        """
        Get all 32 NFL teams with IDs and names

        Returns:
            List of dicts with team_id, abbreviation, display_name, location
        """
        teams_url = f"{self.BASE_SITE}/teams"
        teams_data = self._get(teams_url)

        if not teams_data or 'sports' not in teams_data:
            return []

        teams = []
        for sport in teams_data['sports']:
            for league in sport.get('leagues', []):
                for team in league.get('teams', []):
                    team_info = team.get('team', {})
                    teams.append({
                        'team_id': team_info.get('id'),
                        'abbreviation': team_info.get('abbreviation'),
                        'display_name': team_info.get('displayName'),
                        'location': team_info.get('location')
                    })

        return teams

    def get_all_team_stats_for_season(self, season: int = 2024, season_type: int = 2,
                                      save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Get stats for all 32 NFL teams

        Args:
            season: Year (e.g., 2024, 2025)
            season_type: 2 = regular season, 3 = playoffs
            save_path: Optional path to save parquet file

        Returns:
            DataFrame with team stats (279 stats per team)
        """
        teams = self.get_all_teams()
        if not teams:
            print("❌ Failed to get team list")
            return pd.DataFrame()

        print(f"\n📊 Fetching stats for {len(teams)} teams (Season {season})...")
        all_stats = []

        for i, team in enumerate(teams, 1):
            team_id = team['team_id']
            team_name = team['abbreviation']

            print(f"  [{i}/{len(teams)}] {team_name}...", end=' ')

            # Get team stats
            stats = self.get_team_stats(team_id, season, season_type)
            if stats:
                stats['team_id'] = team_id
                stats['team'] = team_name
                stats['season'] = season
                stats['season_type'] = season_type
                all_stats.append(stats)
                print(f"✅ {len(stats)} stats")
            else:
                print("❌ Failed")

            time.sleep(0.3)  # Rate limiting

        df = pd.DataFrame(all_stats)

        # Save to parquet if path provided
        if save_path and not df.empty:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_path, index=False)
            print(f"\n💾 Saved to {save_path}")

        return df

    def get_all_team_records_for_season(self, season: int = 2024, season_type: int = 2,
                                       save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Get records for all 32 NFL teams

        Args:
            season: Year (e.g., 2024, 2025)
            season_type: 2 = regular season, 3 = playoffs
            save_path: Optional path to save parquet file

        Returns:
            DataFrame with team records (44 stats per team)
        """
        teams = self.get_all_teams()
        if not teams:
            print("❌ Failed to get team list")
            return pd.DataFrame()

        print(f"\n📊 Fetching records for {len(teams)} teams (Season {season})...")
        all_records = []

        for i, team in enumerate(teams, 1):
            team_id = team['team_id']
            team_name = team['abbreviation']

            print(f"  [{i}/{len(teams)}] {team_name}...", end=' ')

            # Get team record
            record = self.get_team_record(team_id, season, season_type)
            if record:
                record['team_id'] = team_id
                record['team'] = team_name
                record['season'] = season
                record['season_type'] = season_type
                all_records.append(record)
                print(f"✅ {len(record)} stats")
            else:
                print("❌ Failed")

            time.sleep(0.3)  # Rate limiting

        df = pd.DataFrame(all_records)

        # Save to parquet if path provided
        if save_path and not df.empty:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_path, index=False)
            print(f"\n💾 Saved to {save_path}")

        return df

    def get_injuries_for_week(self, season: int, week: int,
                             save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Get injuries for all games in a specific week

        Args:
            season: Year (e.g., 2025)
            week: Week number (1-18)
            save_path: Optional path to save parquet file

        Returns:
            DataFrame with injuries for all games in the week
        """
        # Get scoreboard for the week
        scoreboard_url = f"{self.BASE_SITE}/scoreboard"
        params = {'seasontype': 2, 'week': week, 'dates': season}

        scoreboard = self._get(scoreboard_url, params=params)
        if not scoreboard or 'events' not in scoreboard:
            print(f"❌ Failed to get scoreboard for Week {week}")
            return pd.DataFrame()

        events = scoreboard['events']
        print(f"\n🏥 Fetching injuries for Week {week} ({len(events)} games)...")

        all_injuries = []

        for i, event in enumerate(events, 1):
            event_id = event['id']
            event_name = event.get('name', 'Unknown')

            print(f"  [{i}/{len(events)}] {event_name}...", end=' ')

            injuries = self.get_game_injuries(event_id)
            if injuries:
                for injury in injuries:
                    injury['event_id'] = event_id
                    injury['season'] = season
                    injury['week'] = week
                    all_injuries.append(injury)
                print(f"✅ {len(injuries)} injuries")
            else:
                print("✅ No injuries")

            time.sleep(0.3)  # Rate limiting

        df = pd.DataFrame(all_injuries)

        # Save to parquet if path provided
        if save_path and not df.empty:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_path, index=False)
            print(f"\n💾 Saved to {save_path}")

        return df


    # ========================================================================
    # DATA VALIDATION
    # ========================================================================

    def validate_team_stats(self, df: pd.DataFrame) -> Dict:
        """
        Validate team stats data quality

        Returns dict with validation results
        """
        validation = {
            'total_teams': len(df),
            'expected_teams': 32,
            'missing_teams': 32 - len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            'duplicate_teams': df['team'].duplicated().sum() if 'team' in df.columns else 0
        }

        return validation

    def print_validation_report(self, validation: Dict, data_type: str = "Data"):
        """Print formatted validation report"""
        print(f"\n{'='*80}")
        print(f"📋 {data_type} Validation Report")
        print(f"{'='*80}")
        print(f"Total Records: {validation.get('total_teams', 0)}")
        print(f"Expected: {validation.get('expected_teams', 'N/A')}")
        print(f"Missing: {validation.get('missing_teams', 0)}")
        print(f"Total Columns: {validation.get('total_columns', 0)}")
        print(f"Missing Values: {validation.get('missing_values', 0)} ({validation.get('missing_pct', 0):.2f}%)")
        print(f"Duplicates: {validation.get('duplicate_teams', 0)}")

        if validation.get('missing_teams', 0) == 0 and validation.get('duplicate_teams', 0) == 0:
            print("\n✅ Data quality: EXCELLENT")
        elif validation.get('missing_teams', 0) <= 2:
            print("\n⚠️ Data quality: GOOD (minor issues)")
        else:
            print("\n❌ Data quality: POOR (significant issues)")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fetch ESPN Independent Data for NFL Betting Model')
    parser.add_argument('--mode', choices=['test', 'fetch-stats', 'fetch-records', 'fetch-injuries', 'fetch-all'],
                       default='test', help='Operation mode')
    parser.add_argument('--season', type=int, default=2024, help='Season year (e.g., 2024, 2025)')
    parser.add_argument('--week', type=int, help='Week number for injury fetching (1-18)')
    parser.add_argument('--week-range', type=str, help='Week range for injuries (e.g., "1-16")')
    parser.add_argument('--output-dir', type=str, default='data/espn_raw', help='Output directory for data files')

    args = parser.parse_args()

    api = ESPNIndependentData()
    output_dir = Path(args.output_dir)

    if args.mode == 'test':
        # Test the API
        print("Testing ESPN Independent Data API...")
        print("="*80)

        # Test 1: Get all teams
        print("\n[1] Testing get all teams...")
        teams = api.get_all_teams()
        if teams:
            print(f"✅ Got {len(teams)} teams")
            print(f"Sample: {teams[0]}")
        else:
            print("❌ Failed to get teams")

        # Test 2: Get team stats (Kansas City Chiefs = 12)
        print("\n[2] Testing team stats (KC Chiefs)...")
        stats = api.get_team_stats(team_id=12, season=2024)
        if stats:
            print(f"✅ Got {len(stats)} stats")
            print(f"Sample: {list(stats.keys())[:5]}")
        else:
            print("❌ Failed to get team stats")

        # Test 3: Get team record
        print("\n[3] Testing team record (KC Chiefs)...")
        record = api.get_team_record(team_id=12, season=2024)
        if record:
            print(f"✅ Got {len(record)} record stats")
            print(f"Sample: {list(record.keys())[:5]}")
        else:
            print("❌ Failed to get team record")

        print("\n" + "="*80)
        print("API test complete!")

    elif args.mode == 'fetch-stats':
        # Fetch team stats for all teams
        save_path = output_dir / f"team_stats_{args.season}.parquet"
        df = api.get_all_team_stats_for_season(season=args.season, save_path=save_path)

        if not df.empty:
            validation = api.validate_team_stats(df)
            api.print_validation_report(validation, f"Team Stats {args.season}")

    elif args.mode == 'fetch-records':
        # Fetch team records for all teams
        save_path = output_dir / f"team_records_{args.season}.parquet"
        df = api.get_all_team_records_for_season(season=args.season, save_path=save_path)

        if not df.empty:
            validation = api.validate_team_stats(df)
            api.print_validation_report(validation, f"Team Records {args.season}")

    elif args.mode == 'fetch-injuries':
        # Fetch injuries for specific week(s)
        if args.week_range:
            # Parse week range (e.g., "1-16")
            start_week, end_week = map(int, args.week_range.split('-'))
            weeks = range(start_week, end_week + 1)
        elif args.week:
            weeks = [args.week]
        else:
            print("❌ Please specify --week or --week-range")
            exit(1)

        all_injuries = []
        for week in weeks:
            save_path = output_dir / f"injuries_{args.season}_week{week}.parquet"
            df = api.get_injuries_for_week(season=args.season, week=week, save_path=save_path)
            if not df.empty:
                all_injuries.append(df)

        if all_injuries:
            combined = pd.concat(all_injuries, ignore_index=True)
            combined_path = output_dir / f"injuries_{args.season}_weeks_{weeks[0]}-{weeks[-1]}.parquet"
            combined.to_parquet(combined_path, index=False)
            print(f"\n💾 Combined injuries saved to {combined_path}")
            print(f"Total injuries: {len(combined)}")

    elif args.mode == 'fetch-all':
        # Fetch everything for a season
        print(f"\n🚀 Fetching ALL data for {args.season} season...")
        print("="*80)

        # 1. Team stats
        print("\n[1/3] Fetching team stats...")
        stats_path = output_dir / f"team_stats_{args.season}.parquet"
        stats_df = api.get_all_team_stats_for_season(season=args.season, save_path=stats_path)
        if not stats_df.empty:
            validation = api.validate_team_stats(stats_df)
            api.print_validation_report(validation, f"Team Stats {args.season}")

        # 2. Team records
        print("\n[2/3] Fetching team records...")
        records_path = output_dir / f"team_records_{args.season}.parquet"
        records_df = api.get_all_team_records_for_season(season=args.season, save_path=records_path)
        if not records_df.empty:
            validation = api.validate_team_stats(records_df)
            api.print_validation_report(validation, f"Team Records {args.season}")

        # 3. Injuries (if 2025, fetch weeks 1-16)
        if args.season == 2025:
            print("\n[3/3] Fetching injuries for weeks 1-16...")
            all_injuries = []
            for week in range(1, 17):
                save_path = output_dir / f"injuries_{args.season}_week{week}.parquet"
                df = api.get_injuries_for_week(season=args.season, week=week, save_path=save_path)
                if not df.empty:
                    all_injuries.append(df)

            if all_injuries:
                combined = pd.concat(all_injuries, ignore_index=True)
                combined_path = output_dir / f"injuries_{args.season}_weeks_1-16.parquet"
                combined.to_parquet(combined_path, index=False)
                print(f"\n💾 Combined injuries saved to {combined_path}")
                print(f"Total injuries: {len(combined)}")

        print("\n" + "="*80)
        print("✅ All data fetched successfully!")
        print(f"📁 Data saved to: {output_dir}")
        print("="*80)

