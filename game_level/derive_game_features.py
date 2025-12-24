"""
Phase 5A: Game-Level Feature Derivation

Derives ESPN features at the game level (one row per team-game) for moneyline betting.

Key Function:
    derive_game_features(team, game_id, pbp, schedules) -> dict

This is adapted from full_feature_derivation.py but filters to a single game
instead of aggregating across an entire season.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from team_abbreviation_mapping import espn_to_nfl_data_py, nfl_data_py_to_espn

warnings.filterwarnings('ignore')


def derive_game_features(team: str, game_id: str, pbp: pd.DataFrame, schedules: pd.DataFrame) -> dict:
    """
    Derive all ESPN features for a team's performance in a single game.
    
    Args:
        team: ESPN team abbreviation (e.g., 'LAR', 'WSH')
        game_id: Game identifier (e.g., '2024_01_BAL_KC')
        pbp: Play-by-play data (uses nfl-data-py abbreviations)
        schedules: Schedule data (uses nfl-data-py abbreviations)
    
    Returns:
        Dictionary with team, game_id, and all derived features
    """
    features = {
        'team': team,
        'game_id': game_id
    }
    
    # Convert ESPN team abbreviation to nfl-data-py abbreviation
    nfl_team = espn_to_nfl_data_py(team)
    
    # Filter to this specific game
    pbp_game = pbp[pbp['game_id'] == game_id].copy()
    schedule_game = schedules[schedules['game_id'] == game_id].copy()
    
    if len(pbp_game) == 0:
        raise ValueError(f"No play-by-play data found for game_id: {game_id}")
    
    if len(schedule_game) == 0:
        raise ValueError(f"No schedule data found for game_id: {game_id}")
    
    # Filter to team's plays (offensive plays where team has possession)
    team_pbp = pbp_game[pbp_game['posteam'] == nfl_team].copy()
    def_pbp = pbp_game[pbp_game['defteam'] == nfl_team].copy()
    
    # Play type filters
    pass_plays = team_pbp[team_pbp['play_type'] == 'pass'].copy()
    rush_plays = team_pbp[team_pbp['play_type'] == 'run'].copy()
    
    # Get game info from schedule
    game_info = schedule_game.iloc[0]
    is_home = game_info['home_team'] == nfl_team
    team_score = game_info['home_score'] if is_home else game_info['away_score']
    opp_score = game_info['away_score'] if is_home else game_info['home_score']
    
    # Game outcome
    features['win'] = 1 if team_score > opp_score else 0
    features['loss'] = 1 if team_score < opp_score else 0
    features['tie'] = 1 if team_score == opp_score else 0
    features['team_score'] = team_score
    features['opp_score'] = opp_score
    features['point_differential'] = team_score - opp_score
    
    # Week and season info
    features['week'] = game_info['week']
    features['season'] = game_info['season']
    features['is_home'] = 1 if is_home else 0
    
    # ========================================================================
    # PASSING FEATURES
    # ========================================================================
    
    # Basic counting stats (EXACT MATCH)
    features['passing_completions'] = pass_plays['complete_pass'].sum()
    features['passing_passingAttempts'] = pass_plays['pass_attempt'].sum()
    features['passing_passingTouchdowns'] = pass_plays['pass_touchdown'].sum()
    features['passing_interceptions'] = pass_plays['interception'].sum()
    features['passing_sacks'] = team_pbp['sack'].sum()
    
    # Sack yards (EXACT MATCH)
    sack_plays = team_pbp[team_pbp['sack'] == 1]
    features['passing_sackYardsLost'] = abs(sack_plays['yards_gained'].sum())
    
    # Passing yards (EXACT MATCH) - net passing yards (gross - sack yards lost)
    gross_pass_yards = pass_plays['yards_gained'].sum()
    features['passing_passingYards'] = gross_pass_yards - features['passing_sackYardsLost']
    
    # Net passing yards (EXACT MATCH) - same as passing yards
    features['passing_netPassingYards'] = features['passing_passingYards']
    
    # Longest pass (EXACT MATCH)
    features['passing_longPassing'] = pass_plays['yards_gained'].max() if len(pass_plays) > 0 else 0
    
    # Passing yards after catch (EXACT MATCH) - available 2006+
    if 'yards_after_catch' in pass_plays.columns:
        features['passing_passingYardsAfterCatch'] = pass_plays[pass_plays['complete_pass']==1]['yards_after_catch'].sum()
    else:
        features['passing_passingYardsAfterCatch'] = 0
    
    # Passing yards at catch / air yards (EXACT MATCH) - available 2006+
    if 'air_yards' in pass_plays.columns:
        features['passing_passingYardsAtCatch'] = pass_plays[pass_plays['complete_pass']==1]['air_yards'].sum()
    else:
        features['passing_passingYardsAtCatch'] = 0
    
    # Net passing attempts (EXACT MATCH) - attempts excluding sacks
    features['passing_netPassingAttempts'] = features['passing_passingAttempts']
    
    # Total touchdowns (EXACT MATCH) - same as passing TDs for passing category
    features['passing_totalTouchdowns'] = features['passing_passingTouchdowns']
    
    # Passing first downs (PARTIAL MATCH)
    features['passing_passingFirstDowns'] = pass_plays['first_down'].sum()
    
    # Passing fumbles (PARTIAL MATCH)
    features['passing_passingFumbles'] = pass_plays['fumble'].sum()
    features['passing_passingFumblesLost'] = pass_plays['fumble_lost'].sum()
    
    # Two-point conversions (PARTIAL MATCH)
    two_pt_pass = team_pbp[(team_pbp['two_point_attempt'] == 1) & (team_pbp['play_type'] == 'pass')]
    features['passing_twoPtPassAttempts'] = len(two_pt_pass)
    features['passing_twoPointPassConvs'] = two_pt_pass['two_point_conv_result'].eq('success').sum() if 'two_point_conv_result' in two_pt_pass.columns else 0
    features['passing_twoPtPass'] = features['passing_twoPointPassConvs']  # Alias
    
    # Calculated metrics (PARTIAL MATCH)
    if features['passing_passingAttempts'] > 0:
        features['passing_completionPct'] = (features['passing_completions'] / features['passing_passingAttempts']) * 100
        features['passing_yardsPerPassAttempt'] = features['passing_passingYards'] / features['passing_passingAttempts']
        features['passing_interceptionPct'] = (features['passing_interceptions'] / features['passing_passingAttempts']) * 100
        features['passing_passingTouchdownPct'] = (features['passing_passingTouchdowns'] / features['passing_passingAttempts']) * 100

        # Net yards per pass attempt (PARTIAL MATCH) - includes sack yards
        net_attempts = features['passing_passingAttempts'] + features['passing_sacks']
        net_yards = features['passing_passingYards'] - features['passing_sackYardsLost']
        features['passing_netYardsPerPassAttempt'] = net_yards / net_attempts if net_attempts > 0 else 0
    else:
        features['passing_completionPct'] = 0
        features['passing_yardsPerPassAttempt'] = 0
        features['passing_interceptionPct'] = 0
        features['passing_passingTouchdownPct'] = 0
        features['passing_netYardsPerPassAttempt'] = 0

    # Big plays (PARTIAL MATCH) - passes of 20+ yards
    features['passing_passingBigPlays'] = len(pass_plays[pass_plays['yards_gained'] >= 20])

    # Average gain (PARTIAL MATCH)
    features['passing_avgGain'] = pass_plays['yards_gained'].mean() if len(pass_plays) > 0 else 0

    # QB Rating (CLOSE APPROXIMATION)
    if features['passing_passingAttempts'] > 0:
        att = features['passing_passingAttempts']
        comp = features['passing_completions']
        yards = features['passing_passingYards']
        td = features['passing_passingTouchdowns']
        ints = features['passing_interceptions']

        a = max(0, min(2.375, ((comp / att) - 0.3) * 5))
        b = max(0, min(2.375, ((yards / att) - 3) * 0.25))
        c = max(0, min(2.375, (td / att) * 20))
        d = max(0, min(2.375, 2.375 - ((ints / att) * 25)))

        features['passing_QBRating'] = ((a + b + c + d) / 6) * 100
    else:
        features['passing_QBRating'] = 0

    # Aliases for passing features
    features['passing_quarterbackRating'] = features['passing_QBRating']
    if features['passing_completions'] > 0:
        features['passing_yardsPerCompletion'] = features['passing_passingYards'] / features['passing_completions']
    else:
        features['passing_yardsPerCompletion'] = 0

    # ========================================================================
    # RUSHING FEATURES
    # ========================================================================

    # Basic counting stats (EXACT MATCH)
    features['rushing_rushingAttempts'] = rush_plays['rush_attempt'].sum()
    features['rushing_rushingYards'] = rush_plays['yards_gained'].sum()
    features['rushing_rushingTouchdowns'] = rush_plays['rush_touchdown'].sum()
    features['rushing_longRushing'] = rush_plays['yards_gained'].max() if len(rush_plays) > 0 else 0
    features['rushing_rushingFirstDowns'] = rush_plays['first_down'].sum()

    # Stuffs (EXACT MATCH) - runs for 0 or negative yards
    features['rushing_stuffs'] = len(rush_plays[rush_plays['yards_gained'] <= 0])

    # Rushing fumbles (PARTIAL MATCH)
    features['rushing_rushingFumbles'] = rush_plays['fumble'].sum()
    features['rushing_rushingFumblesLost'] = rush_plays['fumble_lost'].sum()

    # Two-point conversions (PARTIAL MATCH)
    two_pt_rush = team_pbp[(team_pbp['two_point_attempt'] == 1) & (team_pbp['play_type'] == 'run')]
    features['rushing_twoPtRushAttempts'] = len(two_pt_rush)
    features['rushing_twoPointRushConvs'] = two_pt_rush['two_point_conv_result'].eq('success').sum() if 'two_point_conv_result' in two_pt_rush.columns else 0
    features['rushing_twoPtRush'] = features['rushing_twoPointRushConvs']  # Alias

    # Calculated metrics (PARTIAL MATCH)
    if features['rushing_rushingAttempts'] > 0:
        features['rushing_yardsPerRushAttempt'] = features['rushing_rushingYards'] / features['rushing_rushingAttempts']
        features['rushing_rushingTouchdownPct'] = (features['rushing_rushingTouchdowns'] / features['rushing_rushingAttempts']) * 100
        features['rushing_stuffPct'] = (features['rushing_stuffs'] / features['rushing_rushingAttempts']) * 100
    else:
        features['rushing_yardsPerRushAttempt'] = 0
        features['rushing_rushingTouchdownPct'] = 0
        features['rushing_stuffPct'] = 0

    # Big plays (PARTIAL MATCH) - runs of 10+ yards
    features['rushing_rushingBigPlays'] = len(rush_plays[rush_plays['yards_gained'] >= 10])

    # Average gain (PARTIAL MATCH)
    features['rushing_avgGain'] = rush_plays['yards_gained'].mean() if len(rush_plays) > 0 else 0

    # ========================================================================
    # TOTAL OFFENSE FEATURES
    # ========================================================================

    # Total yards (EXACT MATCH)
    features['total_totalYards'] = features['passing_passingYards'] + features['rushing_rushingYards']

    # Total plays (EXACT MATCH)
    features['total_totalPlays'] = features['passing_passingAttempts'] + features['rushing_rushingAttempts'] + features['passing_sacks']

    # Yards per play (PARTIAL MATCH)
    if features['total_totalPlays'] > 0:
        features['total_yardsPerPlay'] = features['total_totalYards'] / features['total_totalPlays']
    else:
        features['total_yardsPerPlay'] = 0

    # Total touchdowns (EXACT MATCH)
    features['total_totalTouchdowns'] = features['passing_passingTouchdowns'] + features['rushing_rushingTouchdowns']

    # Total first downs (PARTIAL MATCH)
    features['total_totalFirstDowns'] = features['passing_passingFirstDowns'] + features['rushing_rushingFirstDowns']

    # Turnovers (PARTIAL MATCH)
    features['total_turnovers'] = features['passing_interceptions'] + features['passing_passingFumblesLost'] + features['rushing_rushingFumblesLost']

    # Third down conversions (PARTIAL MATCH)
    third_down_plays = team_pbp[team_pbp['down'] == 3]
    features['total_thirdDownAttempts'] = len(third_down_plays)
    features['total_thirdDownConversions'] = third_down_plays['third_down_converted'].sum()
    if features['total_thirdDownAttempts'] > 0:
        features['total_thirdDownPct'] = (features['total_thirdDownConversions'] / features['total_thirdDownAttempts']) * 100
    else:
        features['total_thirdDownPct'] = 0

    # Fourth down conversions (PARTIAL MATCH)
    fourth_down_plays = team_pbp[team_pbp['down'] == 4]
    features['total_fourthDownAttempts'] = len(fourth_down_plays)
    features['total_fourthDownConversions'] = fourth_down_plays['fourth_down_converted'].sum()
    if features['total_fourthDownAttempts'] > 0:
        features['total_fourthDownPct'] = (features['total_fourthDownConversions'] / features['total_fourthDownAttempts']) * 100
    else:
        features['total_fourthDownPct'] = 0

    # Possession time (PARTIAL MATCH) - in seconds
    features['total_possessionTime'] = team_pbp['drive_time_of_possession'].sum() if 'drive_time_of_possession' in team_pbp.columns else 0

    # Penalties (PARTIAL MATCH)
    features['total_penalties'] = team_pbp['penalty'].sum()
    features['total_penaltyYards'] = team_pbp[team_pbp['penalty']==1]['penalty_yards'].sum()

    # ========================================================================
    # DEFENSIVE FEATURES
    # ========================================================================

    # Defensive passing stats
    def_pass_plays = def_pbp[def_pbp['play_type'] == 'pass'].copy()
    features['defensive_passingYardsAllowed'] = def_pass_plays['yards_gained'].sum()
    features['defensive_passingTouchdownsAllowed'] = def_pass_plays['pass_touchdown'].sum()
    features['defensive_interceptions'] = def_pass_plays['interception'].sum()
    features['defensive_sacks'] = def_pbp['sack'].sum()

    # Defensive rushing stats
    def_rush_plays = def_pbp[def_pbp['play_type'] == 'run'].copy()
    features['defensive_rushingYardsAllowed'] = def_rush_plays['yards_gained'].sum()
    features['defensive_rushingTouchdownsAllowed'] = def_rush_plays['rush_touchdown'].sum()

    # Total defensive stats
    features['defensive_totalYardsAllowed'] = features['defensive_passingYardsAllowed'] + features['defensive_rushingYardsAllowed']
    features['defensive_pointsAllowed'] = opp_score
    features['defensive_touchdownsAllowed'] = features['defensive_passingTouchdownsAllowed'] + features['defensive_rushingTouchdownsAllowed']

    # Defensive turnovers forced
    features['defensive_fumblesRecovered'] = def_pbp['fumble_lost'].sum()  # Opponent fumbles lost = our fumbles recovered
    features['defensive_turnoversForced'] = features['defensive_interceptions'] + features['defensive_fumblesRecovered']

    # Defensive third down
    def_third_down = def_pbp[def_pbp['down'] == 3]
    features['defensive_thirdDownAttempts'] = len(def_third_down)
    features['defensive_thirdDownConversionsAllowed'] = def_third_down['third_down_converted'].sum()
    if features['defensive_thirdDownAttempts'] > 0:
        features['defensive_thirdDownPct'] = (features['defensive_thirdDownConversionsAllowed'] / features['defensive_thirdDownAttempts']) * 100
    else:
        features['defensive_thirdDownPct'] = 0

    # ========================================================================
    # TEAM RECORD FEATURES (GAME-LEVEL)
    # ========================================================================

    # For game-level, these are binary indicators (1 or 0)
    # Season-level aggregations will sum these to get total wins/losses
    features['total_wins'] = features['win']
    features['total_losses'] = features['loss']
    features['total_ties'] = features['tie']

    # Win percentage (for single game, this is just 1.0 if win, 0.0 if loss)
    features['total_winPercent'] = float(features['win'])
    features['total_leagueWinPercent'] = float(features['win'])  # Same as winPercent for single game

    # Point differential
    features['total_differential'] = features['point_differential']
    features['total_pointsFor'] = team_score
    features['total_pointsAgainst'] = opp_score

    # Streak indicators (will be calculated across games in feature engineering)
    # For now, set to 0 (will be updated in rolling feature calculation)
    features['total_streak'] = 0

    # Division/conference record (requires opponent info - set to 0 for now)
    features['total_divisionWinPercent'] = 0.0
    features['total_conferenceWinPercent'] = 0.0

    # Home/away record
    if is_home:
        features['total_homeWins'] = features['win']
        features['total_homeLosses'] = features['loss']
        features['total_awayWins'] = 0
        features['total_awayLosses'] = 0
    else:
        features['total_homeWins'] = 0
        features['total_homeLosses'] = 0
        features['total_awayWins'] = features['win']
        features['total_awayLosses'] = features['loss']

    # ========================================================================
    # SPECIAL TEAMS FEATURES (if available in play-by-play)
    # ========================================================================

    # Field goals
    fg_plays = team_pbp[team_pbp['field_goal_attempt'] == 1]
    features['kicking_fieldGoalsMade'] = fg_plays['field_goal_result'].eq('made').sum() if 'field_goal_result' in fg_plays.columns else 0
    features['kicking_fieldGoalAttempts'] = len(fg_plays)
    if features['kicking_fieldGoalAttempts'] > 0:
        features['kicking_fieldGoalPct'] = (features['kicking_fieldGoalsMade'] / features['kicking_fieldGoalAttempts']) * 100
    else:
        features['kicking_fieldGoalPct'] = 0

    # Extra points
    xp_plays = team_pbp[team_pbp['extra_point_attempt'] == 1]
    features['kicking_extraPointsMade'] = xp_plays['extra_point_result'].eq('good').sum() if 'extra_point_result' in xp_plays.columns else 0
    features['kicking_extraPointAttempts'] = len(xp_plays)

    # Punting
    punt_plays = team_pbp[team_pbp['punt_attempt'] == 1]
    features['punting_punts'] = len(punt_plays)
    features['punting_puntYards'] = punt_plays['kick_distance'].sum() if 'kick_distance' in punt_plays.columns else 0
    if features['punting_punts'] > 0:
        features['punting_yardsPerPunt'] = features['punting_puntYards'] / features['punting_punts']
    else:
        features['punting_yardsPerPunt'] = 0

    # Kickoff returns
    ko_return_plays = def_pbp[def_pbp['kickoff_attempt'] == 1]
    features['returning_kickoffReturns'] = len(ko_return_plays)
    features['returning_kickoffReturnYards'] = ko_return_plays['return_yards'].sum() if 'return_yards' in ko_return_plays.columns else 0
    if features['returning_kickoffReturns'] > 0:
        features['returning_kickoffReturnAverage'] = features['returning_kickoffReturnYards'] / features['returning_kickoffReturns']
    else:
        features['returning_kickoffReturnAverage'] = 0

    # Punt returns
    punt_return_plays = def_pbp[def_pbp['punt_attempt'] == 1]
    features['returning_puntReturns'] = len(punt_return_plays)
    features['returning_puntReturnYards'] = punt_return_plays['return_yards'].sum() if 'return_yards' in punt_return_plays.columns else 0
    if features['returning_puntReturns'] > 0:
        features['returning_puntReturnAverage'] = features['returning_puntReturnYards'] / features['returning_puntReturns']
    else:
        features['returning_puntReturnAverage'] = 0

    return features

