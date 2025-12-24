"""
Derive ESPN-like features from nfl-data-py for validation
Phase 2: Feature Derivation & Validation
"""
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class ESPNFeatureDerivation:
    """Derive ESPN-like features from nfl-data-py"""
    
    def __init__(self, years: List[int]):
        """Initialize with years to process"""
        self.years = years
        self.pbp = None
        self.schedules = None

    def load_data(self):
        """Load all required nfl-data-py data"""
        print(f"Loading nfl-data-py data for {self.years}...")

        # Load play-by-play data
        print("  - Loading play-by-play data...")
        self.pbp = nfl.import_pbp_data(self.years)
        print(f"    Loaded {len(self.pbp):,} plays")

        # Load schedules
        print("  - Loading schedules...")
        self.schedules = nfl.import_schedules(self.years)
        print(f"    Loaded {len(self.schedules):,} games")

        print("✅ Data loading complete!\n")
        
    def derive_passing_features(self, team: str, season: int) -> Dict:
        """Derive passing features for a team-season"""
        # Filter to team's offensive plays
        team_pbp = self.pbp[
            (self.pbp['posteam'] == team) & 
            (self.pbp['season'] == season)
        ]
        
        # Filter to passing plays
        pass_plays = team_pbp[team_pbp['play_type'] == 'pass']
        
        features = {}
        
        # Basic passing stats
        features['passing_passingYards'] = pass_plays['yards_gained'].sum()
        features['passing_passingAttempts'] = pass_plays['pass_attempt'].sum()
        features['passing_completions'] = pass_plays['complete_pass'].sum()
        features['passing_passingTouchdowns'] = pass_plays['pass_touchdown'].sum()
        features['passing_interceptions'] = pass_plays['interception'].sum()
        features['passing_sacks'] = team_pbp['sack'].sum()
        features['passing_sackYardsLost'] = abs(team_pbp[team_pbp['sack'] == 1]['yards_gained'].sum())
        
        # Calculated stats
        if features['passing_passingAttempts'] > 0:
            features['passing_completionPct'] = (
                features['passing_completions'] / features['passing_passingAttempts'] * 100
            )
            features['passing_interceptionPct'] = (
                features['passing_interceptions'] / features['passing_passingAttempts'] * 100
            )
            features['passing_passingTouchdownPct'] = (
                features['passing_passingTouchdowns'] / features['passing_passingAttempts'] * 100
            )
            features['passing_avgGain'] = (
                features['passing_passingYards'] / features['passing_passingAttempts']
            )
        else:
            features['passing_completionPct'] = 0
            features['passing_interceptionPct'] = 0
            features['passing_passingTouchdownPct'] = 0
            features['passing_avgGain'] = 0
        
        # Air yards and YAC (available from 2006+)
        if season >= 2006:
            features['passing_passingYardsAfterCatch'] = pass_plays[
                pass_plays['complete_pass'] == 1
            ]['yards_after_catch'].sum()
            features['passing_passingYardsAtCatch'] = pass_plays[
                pass_plays['complete_pass'] == 1
            ]['air_yards'].sum()
        else:
            features['passing_passingYardsAfterCatch'] = np.nan
            features['passing_passingYardsAtCatch'] = np.nan
        
        # Net passing yards (passing yards - sack yards)
        features['passing_netPassingYards'] = (
            features['passing_passingYards'] - features['passing_sackYardsLost']
        )
        
        # Net passing attempts (attempts + sacks)
        features['passing_netPassingAttempts'] = (
            features['passing_passingAttempts'] + features['passing_sacks']
        )
        
        # First downs
        features['passing_passingFirstDowns'] = pass_plays['first_down_pass'].sum()
        
        # Big plays (20+ yards)
        features['passing_passingBigPlays'] = len(pass_plays[pass_plays['yards_gained'] >= 20])
        
        # Long passing (longest completion)
        if len(pass_plays[pass_plays['complete_pass'] == 1]) > 0:
            features['passing_longPassing'] = pass_plays[
                pass_plays['complete_pass'] == 1
            ]['yards_gained'].max()
        else:
            features['passing_longPassing'] = 0
        
        # Fumbles
        features['passing_passingFumbles'] = pass_plays['fumble'].sum()
        features['passing_passingFumblesLost'] = pass_plays['fumble_lost'].sum()
        
        # Two-point conversions
        two_pt_pass = team_pbp[team_pbp['two_point_attempt'] == 1]
        features['passing_twoPtPassAttempts'] = len(two_pt_pass[two_pt_pass['pass_attempt'] == 1])
        features['passing_twoPointPassConvs'] = len(
            two_pt_pass[(two_pt_pass['pass_attempt'] == 1) & (two_pt_pass['two_point_conv_result'] == 'success')]
        )
        
        # QB Rating (NFL passer rating formula)
        if features['passing_passingAttempts'] > 0:
            att = features['passing_passingAttempts']
            comp = features['passing_completions']
            yards = features['passing_passingYards']
            tds = features['passing_passingTouchdowns']
            ints = features['passing_interceptions']
            
            a = ((comp / att) - 0.3) * 5
            b = ((yards / att) - 3) * 0.25
            c = (tds / att) * 20
            d = 2.375 - ((ints / att) * 25)
            
            # Clamp each component between 0 and 2.375
            a = max(0, min(a, 2.375))
            b = max(0, min(b, 2.375))
            c = max(0, min(c, 2.375))
            d = max(0, min(d, 2.375))
            
            features['passing_QBRating'] = ((a + b + c + d) / 6) * 100
        else:
            features['passing_QBRating'] = 0
        
        return features
    
    def derive_rushing_features(self, team: str, season: int) -> Dict:
        """Derive rushing features for a team-season"""
        # Filter to team's offensive plays
        team_pbp = self.pbp[
            (self.pbp['posteam'] == team) & 
            (self.pbp['season'] == season)
        ]
        
        # Filter to rushing plays
        rush_plays = team_pbp[team_pbp['play_type'] == 'run']
        
        features = {}
        
        # Basic rushing stats
        features['rushing_rushingYards'] = rush_plays['yards_gained'].sum()
        features['rushing_rushingAttempts'] = rush_plays['rush_attempt'].sum()
        features['rushing_rushingTouchdowns'] = rush_plays['rush_touchdown'].sum()
        
        # Calculated stats
        if features['rushing_rushingAttempts'] > 0:
            features['rushing_avgGain'] = (
                features['rushing_rushingYards'] / features['rushing_rushingAttempts']
            )
        else:
            features['rushing_avgGain'] = 0
        
        # First downs
        features['rushing_rushingFirstDowns'] = rush_plays['first_down_rush'].sum()
        
        # Big plays (20+ yards)
        features['rushing_rushingBigPlays'] = len(rush_plays[rush_plays['yards_gained'] >= 20])
        
        # Long rushing (longest run)
        if len(rush_plays) > 0:
            features['rushing_longRushing'] = rush_plays['yards_gained'].max()
        else:
            features['rushing_longRushing'] = 0
        
        # Fumbles
        features['rushing_rushingFumbles'] = rush_plays['fumble'].sum()
        features['rushing_rushingFumblesLost'] = rush_plays['fumble_lost'].sum()

        return features

    def derive_receiving_features(self, team: str, season: int) -> Dict:
        """Derive receiving features for a team-season"""
        # Receiving stats are same as passing stats from team perspective
        pass_features = self.derive_passing_features(team, season)

        features = {}
        features['receiving_receivingYards'] = pass_features['passing_passingYards']
        features['receiving_receptions'] = pass_features['passing_completions']
        features['receiving_receivingTouchdowns'] = pass_features['passing_passingTouchdowns']

        # Calculate additional receiving metrics
        if features['receiving_receptions'] > 0:
            features['receiving_avgGain'] = (
                features['receiving_receivingYards'] / features['receiving_receptions']
            )
        else:
            features['receiving_avgGain'] = 0

        return features

    def derive_defensive_features(self, team: str, season: int) -> Dict:
        """Derive defensive features for a team-season"""
        # Filter to plays where team is on defense
        def_pbp = self.pbp[
            (self.pbp['defteam'] == team) &
            (self.pbp['season'] == season)
        ]

        features = {}

        # Sacks
        features['defensive_totalSacks'] = def_pbp['sack'].sum()
        features['defensive_avgSackYards'] = (
            abs(def_pbp[def_pbp['sack'] == 1]['yards_gained'].sum()) /
            max(1, features['defensive_totalSacks'])
        )

        # Interceptions
        features['defensiveInterceptions_interceptions'] = def_pbp['interception'].sum()
        features['defensiveInterceptions_interceptionYards'] = def_pbp[
            def_pbp['interception'] == 1
        ]['return_yards'].sum()
        features['defensiveInterceptions_interceptionTouchdowns'] = def_pbp[
            (def_pbp['interception'] == 1) & (def_pbp['return_touchdown'] == 1)
        ].shape[0]

        if features['defensiveInterceptions_interceptions'] > 0:
            features['defensive_avgInterceptionYards'] = (
                features['defensiveInterceptions_interceptionYards'] /
                features['defensiveInterceptions_interceptions']
            )
        else:
            features['defensive_avgInterceptionYards'] = 0

        # Tackles for loss
        features['defensive_tacklesForLoss'] = def_pbp[
            def_pbp['yards_gained'] < 0
        ].shape[0]

        # QB hits (not available in play-by-play, set to NaN)
        features['defensive_qbHits'] = np.nan
        features['defensive_hurries'] = np.nan  # ESPN-only feature

        # Fumbles forced/recovered
        features['defensive_fumblesForced'] = def_pbp['fumble_forced'].sum()
        features['defensive_fumblesRecovered'] = def_pbp[
            (def_pbp['fumble'] == 1) & (def_pbp['fumble_recovery_1_team'] == team)
        ].shape[0]

        # Defensive TDs
        features['defensive_defensiveTouchdowns'] = def_pbp[
            (def_pbp['return_touchdown'] == 1) |
            ((def_pbp['fumble'] == 1) & (def_pbp['td_team'] == team))
        ].shape[0]

        # Passes defended
        features['defensive_passesDefended'] = def_pbp['pass_defense_1_player_id'].notna().sum()

        # Safeties
        features['defensive_safeties'] = def_pbp[def_pbp['safety'] == 1].shape[0]

        return features

    def derive_general_features(self, team: str, season: int) -> Dict:
        """Derive general features (fumbles, penalties, etc.)"""
        team_pbp = self.pbp[
            (self.pbp['posteam'] == team) &
            (self.pbp['season'] == season)
        ]

        features = {}

        # Fumbles
        features['general_fumbles'] = team_pbp['fumble'].sum()
        features['general_fumblesLost'] = team_pbp['fumble_lost'].sum()
        features['general_fumblesRecovered'] = team_pbp[
            (team_pbp['fumble'] == 1) & (team_pbp['fumble_recovery_1_team'] == team)
        ].shape[0]
        features['general_fumblesForced'] = team_pbp['fumble_forced'].sum()
        features['general_fumblesTouchdowns'] = team_pbp[
            (team_pbp['fumble'] == 1) & (team_pbp['td_team'] == team)
        ].shape[0]

        # Penalties
        features['general_totalPenalties'] = team_pbp['penalty'].sum()
        features['general_totalPenaltyYards'] = team_pbp[
            team_pbp['penalty'] == 1
        ]['penalty_yards'].sum()

        # Games played
        features['general_gamesPlayed'] = len(
            team_pbp['game_id'].unique()
        )

        return features

    def derive_miscellaneous_features(self, team: str, season: int) -> Dict:
        """Derive miscellaneous features (downs, possession, etc.)"""
        team_pbp = self.pbp[
            (self.pbp['posteam'] == team) &
            (self.pbp['season'] == season)
        ]

        features = {}

        # First downs
        features['miscellaneous_firstDowns'] = team_pbp['first_down'].sum()
        features['miscellaneous_firstDownsPassing'] = team_pbp['first_down_pass'].sum()
        features['miscellaneous_firstDownsRushing'] = team_pbp['first_down_rush'].sum()
        features['miscellaneous_firstDownsPenalty'] = team_pbp['first_down_penalty'].sum()

        # Third down conversions
        third_down = team_pbp[team_pbp['down'] == 3]
        features['miscellaneous_thirdDownAttempts'] = len(third_down)
        features['miscellaneous_thirdDownConvs'] = third_down['third_down_converted'].sum()
        if features['miscellaneous_thirdDownAttempts'] > 0:
            features['miscellaneous_thirdDownConvPct'] = (
                features['miscellaneous_thirdDownConvs'] /
                features['miscellaneous_thirdDownAttempts'] * 100
            )
        else:
            features['miscellaneous_thirdDownConvPct'] = 0

        # Fourth down conversions
        fourth_down = team_pbp[team_pbp['down'] == 4]
        features['miscellaneous_fourthDownAttempts'] = len(fourth_down)
        features['miscellaneous_fourthDownConvs'] = fourth_down['fourth_down_converted'].sum()
        if features['miscellaneous_fourthDownAttempts'] > 0:
            features['miscellaneous_fourthDownConvPct'] = (
                features['miscellaneous_fourthDownConvs'] /
                features['miscellaneous_fourthDownAttempts'] * 100
            )
        else:
            features['miscellaneous_fourthDownConvPct'] = 0

        # Possession time (in seconds)
        # Calculate from play timestamps
        features['miscellaneous_possessionTimeSeconds'] = team_pbp['time_of_possession'].sum()

        # Total offensive plays
        features['miscellaneous_totalOffensivePlays'] = len(
            team_pbp[team_pbp['play_type'].isin(['pass', 'run'])]
        )

        return features

    def derive_team_records(self, team: str, season: int) -> Dict:
        """Derive team record features from schedules"""
        # Filter to team's games
        team_games = self.schedules[
            ((self.schedules['home_team'] == team) | (self.schedules['away_team'] == team)) &
            (self.schedules['season'] == season) &
            (self.schedules['game_type'] == 'REG')  # Regular season only
        ]

        features = {}

        # Overall record
        home_games = team_games[team_games['home_team'] == team]
        away_games = team_games[team_games['away_team'] == team]

        home_wins = len(home_games[home_games['home_score'] > home_games['away_score']])
        home_losses = len(home_games[home_games['home_score'] < home_games['away_score']])
        home_ties = len(home_games[home_games['home_score'] == home_games['away_score']])

        away_wins = len(away_games[away_games['away_score'] > away_games['home_score']])
        away_losses = len(away_games[away_games['away_score'] < away_games['home_score']])
        away_ties = len(away_games[away_games['away_score'] == away_games['home_score']])

        features['total_wins'] = home_wins + away_wins
        features['total_losses'] = home_losses + away_losses
        features['total_ties'] = home_ties + away_ties
        features['total_gamesPlayed'] = len(team_games)

        if features['total_gamesPlayed'] > 0:
            features['total_winPercent'] = (
                (features['total_wins'] + 0.5 * features['total_ties']) /
                features['total_gamesPlayed']
            )
        else:
            features['total_winPercent'] = 0

        # Home/Away records
        features['home_wins'] = home_wins
        features['home_losses'] = home_losses
        features['home_ties'] = home_ties
        if (home_wins + home_losses + home_ties) > 0:
            features['home_winPercent'] = (
                (home_wins + 0.5 * home_ties) / (home_wins + home_losses + home_ties)
            )
        else:
            features['home_winPercent'] = 0

        features['road_wins'] = away_wins
        features['road_losses'] = away_losses
        features['road_ties'] = away_ties
        if (away_wins + away_losses + away_ties) > 0:
            features['road_winPercent'] = (
                (away_wins + 0.5 * away_ties) / (away_wins + away_losses + away_ties)
            )
        else:
            features['road_winPercent'] = 0

        # Division record
        div_games = team_games[team_games['div_game'] == 1]
        div_home = div_games[div_games['home_team'] == team]
        div_away = div_games[div_games['away_team'] == team]

        div_wins = (
            len(div_home[div_home['home_score'] > div_home['away_score']]) +
            len(div_away[div_away['away_score'] > div_away['home_score']])
        )
        div_losses = (
            len(div_home[div_home['home_score'] < div_home['away_score']]) +
            len(div_away[div_away['away_score'] < div_away['home_score']])
        )
        div_ties = (
            len(div_home[div_home['home_score'] == div_home['away_score']]) +
            len(div_away[div_away['away_score'] == div_away['home_score']])
        )

        features['total_divisionWins'] = div_wins
        features['total_divisionLosses'] = div_losses
        features['total_divisionTies'] = div_ties
        if (div_wins + div_losses + div_ties) > 0:
            features['total_divisionWinPercent'] = (
                (div_wins + 0.5 * div_ties) / (div_wins + div_losses + div_ties)
            )
        else:
            features['total_divisionWinPercent'] = 0

        # Points for/against
        home_points_for = home_games['home_score'].sum()
        away_points_for = away_games['away_score'].sum()
        home_points_against = home_games['away_score'].sum()
        away_points_against = away_games['home_score'].sum()

        features['total_pointsFor'] = home_points_for + away_points_for
        features['total_pointsAgainst'] = home_points_against + away_points_against
        features['total_pointDifferential'] = features['total_pointsFor'] - features['total_pointsAgainst']

        if features['total_gamesPlayed'] > 0:
            features['total_avgPointsFor'] = features['total_pointsFor'] / features['total_gamesPlayed']
            features['total_avgPointsAgainst'] = features['total_pointsAgainst'] / features['total_gamesPlayed']
        else:
            features['total_avgPointsFor'] = 0
            features['total_avgPointsAgainst'] = 0

        # Overtime games
        ot_games = team_games[team_games['overtime'] == 1]
        ot_home = ot_games[ot_games['home_team'] == team]
        ot_away = ot_games[ot_games['away_team'] == team]

        features['total_OTWins'] = (
            len(ot_home[ot_home['home_score'] > ot_home['away_score']]) +
            len(ot_away[ot_away['away_score'] > ot_away['home_score']])
        )
        features['total_OTLosses'] = (
            len(ot_home[ot_home['home_score'] < ot_home['away_score']]) +
            len(ot_away[ot_away['away_score'] < ot_away['home_score']])
        )

        return features

    def derive_all_features(self, team: str, season: int) -> Dict:
        """Derive all features for a team-season"""
        all_features = {}

        # Derive each category
        all_features.update(self.derive_passing_features(team, season))
        all_features.update(self.derive_rushing_features(team, season))
        all_features.update(self.derive_receiving_features(team, season))
        all_features.update(self.derive_defensive_features(team, season))
        all_features.update(self.derive_general_features(team, season))
        all_features.update(self.derive_miscellaneous_features(team, season))
        all_features.update(self.derive_team_records(team, season))

        # Add metadata
        all_features['team'] = team
        all_features['season'] = season

        return all_features

    def derive_for_all_teams(self, season: int) -> pd.DataFrame:
        """Derive features for all teams in a season"""
        # Get unique teams from schedules
        teams = sorted(set(
            list(self.schedules[self.schedules['season'] == season]['home_team'].unique()) +
            list(self.schedules[self.schedules['season'] == season]['away_team'].unique())
        ))

        print(f"\nDeriving features for {len(teams)} teams in {season}...")

        all_team_features = []
        for i, team in enumerate(teams, 1):
            print(f"  [{i}/{len(teams)}] {team}...", end=' ')
            try:
                features = self.derive_all_features(team, season)
                all_team_features.append(features)
                print("✅")
            except Exception as e:
                print(f"❌ Error: {e}")

        df = pd.DataFrame(all_team_features)
        print(f"✅ Derived {len(df.columns)} features for {len(df)} teams\n")

        return df


def main():
    """Main execution"""
    print("=" * 80)
    print("ESPN FEATURE DERIVATION - Phase 2 Validation")
    print("=" * 80)

    # Derive for 2024 and 2025
    years = [2024, 2025]

    derivation = ESPNFeatureDerivation(years)
    derivation.load_data()

    # Derive for each year
    for year in years:
        print(f"\n{'=' * 80}")
        print(f"Processing {year}")
        print('=' * 80)

        df = derivation.derive_for_all_teams(year)

        # Save to parquet
        output_path = Path(f'data/derived/team_features_{year}.parquet')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"✅ Saved to: {output_path}")

    print("\n" + "=" * 80)
    print("✅ DERIVATION COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()

