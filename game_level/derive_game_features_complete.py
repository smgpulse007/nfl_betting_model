"""
Phase 5A: Complete Game-Level Feature Derivation

This function derives ALL ESPN features at the game level (one row per team-game).
It is adapted from full_feature_derivation.py with modifications for game-level data.

Key differences from season-level derivation:
- Filters to a single game_id
- games_played = 1 (single game)
- Per-game stats are the same as raw stats (no division by games_played)
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


def derive_game_features_complete(team: str, game_id: str, pbp: pd.DataFrame, schedules: pd.DataFrame) -> dict:
    """
    Derive ALL ESPN features for a team's performance in a single game
    
    Args:
        team: ESPN team abbreviation (e.g., 'LAR', 'WSH')
        pbp: Play-by-play data (uses nfl-data-py abbreviations)
        schedules: Schedule data (uses nfl-data-py abbreviations)
    
    Returns:
        Dictionary with team and all derived features
    """
    features = {'team': team, 'game_id': game_id}
    
    # Filter to this specific game
    pbp_game = pbp[pbp['game_id'] == game_id].copy()
    schedule_game = schedules[schedules['game_id'] == game_id].copy()
    
    if len(pbp_game) == 0:
        raise ValueError(f"No play-by-play data found for game_id: {game_id}")
    
    if len(schedule_game) == 0:
        raise ValueError(f"No schedule data found for game_id: {game_id}")
    
    # Use game-filtered data
    pbp_reg = pbp_game
    schedules_reg = schedule_game

    
    # Convert ESPN team abbreviation to nfl-data-py abbreviation
    nfl_team = espn_to_nfl_data_py(team)
    
    # Filter to team's plays
    team_pbp = pbp_reg[pbp_reg['posteam'] == nfl_team].copy()
    def_pbp = pbp_reg[pbp_reg['defteam'] == nfl_team].copy()
    
    # Play type filters
    pass_plays = team_pbp[team_pbp['play_type'] == 'pass'].copy()
    rush_plays = team_pbp[team_pbp['play_type'] == 'run'].copy()
    
    # Get team schedule
    team_schedule = schedules_reg[(schedules_reg['home_team'] == nfl_team) | 
                                   (schedules_reg['away_team'] == nfl_team)].copy()
    team_schedule['is_home'] = team_schedule['home_team'] == nfl_team
    team_schedule['team_score'] = team_schedule.apply(
        lambda x: x['home_score'] if x['is_home'] else x['away_score'], axis=1)
    team_schedule['opp_score'] = team_schedule.apply(
        lambda x: x['away_score'] if x['is_home'] else x['home_score'], axis=1)
    team_schedule['win'] = team_schedule['team_score'] > team_schedule['opp_score']
    team_schedule['loss'] = team_schedule['team_score'] < team_schedule['opp_score']
    
    games_played = 1  # Single game
    
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

    # Per game stats (PARTIAL MATCH)
    if games_played > 0:
        features['passing_passingYardsPerGame'] = features['passing_passingYards'] / games_played
        features['passing_netPassingYardsPerGame'] = features['passing_netPassingYards'] / games_played
    else:
        features['passing_passingYardsPerGame'] = 0
        features['passing_netPassingYardsPerGame'] = 0

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

    # ESPN QBR (CANNOT DERIVE) - set to NaN
    features['passing_ESPNQBRating'] = np.nan

    # Aliases for passing features (PARTIAL MATCH)
    features['passing_quarterbackRating'] = features['passing_QBRating']
    features['passing_yardsPerCompletion'] = features['passing_yardsPerPassAttempt'] if features['passing_completions'] > 0 else 0
    if features['passing_completions'] > 0:
        features['passing_yardsPerCompletion'] = features['passing_passingYards'] / features['passing_completions']
    features['passing_yardsPerGame'] = features['passing_passingYardsPerGame']

    # Team games played (PARTIAL MATCH)
    features['passing_teamGamesPlayed'] = games_played

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
    else:
        features['rushing_yardsPerRushAttempt'] = 0
        features['rushing_rushingTouchdownPct'] = 0

    # Per game stats (PARTIAL MATCH)
    if games_played > 0:
        features['rushing_rushingYardsPerGame'] = features['rushing_rushingYards'] / games_played
    else:
        features['rushing_rushingYardsPerGame'] = 0

    # Big plays (PARTIAL MATCH) - runs of 20+ yards
    features['rushing_rushingBigPlays'] = len(rush_plays[rush_plays['yards_gained'] >= 20])

    # Average gain (PARTIAL MATCH)
    features['rushing_avgGain'] = rush_plays['yards_gained'].mean() if len(rush_plays) > 0 else 0

    # Broken tackles (EXACT MATCH) - if available
    if 'tackled_for_loss' in rush_plays.columns:
        features['rushing_brokenTackles'] = rush_plays['tackled_for_loss'].eq(0).sum()
    else:
        features['rushing_brokenTackles'] = 0

    # Stuff yards lost (PARTIAL MATCH)
    features['rushing_stuffYardsLost'] = abs(rush_plays[rush_plays['yards_gained'] < 0]['yards_gained'].sum())

    # Total touchdowns (PARTIAL MATCH)
    features['rushing_totalTouchdowns'] = features['rushing_rushingTouchdowns']

    # Yards per game (PARTIAL MATCH)
    if games_played > 0:
        features['rushing_yardsPerGame'] = features['rushing_rushingYards'] / games_played
    else:
        features['rushing_yardsPerGame'] = 0

    # Team games played (PARTIAL MATCH)
    features['rushing_teamGamesPlayed'] = games_played

    # ========================================================================
    # RECEIVING FEATURES
    # ========================================================================

    # Basic counting stats (EXACT MATCH)
    features['receiving_receptions'] = pass_plays['complete_pass'].sum()
    features['receiving_receivingYards'] = pass_plays[pass_plays['complete_pass']==1]['yards_gained'].sum()
    features['receiving_receivingTouchdowns'] = pass_plays['pass_touchdown'].sum()
    features['receiving_longReception'] = pass_plays[pass_plays['complete_pass']==1]['yards_gained'].max() if len(pass_plays[pass_plays['complete_pass']==1]) > 0 else 0
    features['receiving_receivingFirstDowns'] = pass_plays[pass_plays['complete_pass']==1]['first_down'].sum()

    # Targets (PARTIAL MATCH) - all pass attempts
    features['receiving_targets'] = features['passing_passingAttempts']

    # Receiving fumbles (PARTIAL MATCH)
    features['receiving_receivingFumbles'] = pass_plays[pass_plays['complete_pass']==1]['fumble'].sum()
    features['receiving_receivingFumblesLost'] = pass_plays[pass_plays['complete_pass']==1]['fumble_lost'].sum()

    # Calculated metrics (PARTIAL MATCH)
    if features['receiving_receptions'] > 0:
        features['receiving_yardsPerReception'] = features['receiving_receivingYards'] / features['receiving_receptions']
    else:
        features['receiving_yardsPerReception'] = 0

    if features['receiving_targets'] > 0:
        features['receiving_catchPct'] = (features['receiving_receptions'] / features['receiving_targets']) * 100
        features['receiving_yardsPerTarget'] = features['receiving_receivingYards'] / features['receiving_targets']
    else:
        features['receiving_catchPct'] = 0
        features['receiving_yardsPerTarget'] = 0

    # Per game stats (PARTIAL MATCH)
    if games_played > 0:
        features['receiving_receivingYardsPerGame'] = features['receiving_receivingYards'] / games_played
    else:
        features['receiving_receivingYardsPerGame'] = 0

    # Big plays (PARTIAL MATCH) - receptions of 20+ yards
    features['receiving_receivingBigPlays'] = len(pass_plays[(pass_plays['complete_pass']==1) & (pass_plays['yards_gained'] >= 20)])

    # Average gain (PARTIAL MATCH)
    features['receiving_avgGain'] = pass_plays[pass_plays['complete_pass']==1]['yards_gained'].mean() if len(pass_plays[pass_plays['complete_pass']==1]) > 0 else 0

    # Receiving targets (PARTIAL MATCH) - if available
    if 'pass_attempt' in team_pbp.columns:
        # Targets = completions + incompletions (excluding sacks)
        features['receiving_receivingTargets'] = features['receiving_receptions'] + len(team_pbp[(team_pbp['incomplete_pass'] == 1)])
    else:
        features['receiving_receivingTargets'] = 0

    # Yards after catch (PARTIAL MATCH) - if available
    if 'yards_after_catch' in team_pbp.columns:
        features['receiving_receivingYardsAfterCatch'] = team_pbp[team_pbp['complete_pass'] == 1]['yards_after_catch'].sum()
        features['receiving_receivingYardsAtCatch'] = features['receiving_receivingYards'] - features['receiving_receivingYardsAfterCatch']
    else:
        features['receiving_receivingYardsAfterCatch'] = 0
        features['receiving_receivingYardsAtCatch'] = 0

    # Two-point conversions (PARTIAL MATCH)
    if 'two_point_conv_result' in team_pbp.columns:
        two_pt_rec = team_pbp[(team_pbp['two_point_attempt'] == 1) & (team_pbp['complete_pass'] == 1)]
        features['receiving_twoPtReceptionAttempts'] = len(two_pt_rec)
        features['receiving_twoPtReception'] = len(two_pt_rec[two_pt_rec['two_point_conv_result'] == 'success'])
        features['receiving_twoPointRecConvs'] = features['receiving_twoPtReception']
    else:
        features['receiving_twoPtReceptionAttempts'] = 0
        features['receiving_twoPtReception'] = 0
        features['receiving_twoPointRecConvs'] = 0

    # Total touchdowns (EXACT MATCH)
    features['receiving_totalTouchdowns'] = features['receiving_receivingTouchdowns']

    # Yards per game (PARTIAL MATCH)
    if games_played > 0:
        features['receiving_yardsPerGame'] = features['receiving_receivingYards'] / games_played
    else:
        features['receiving_yardsPerGame'] = 0

    # Team games played (PARTIAL MATCH)
    features['receiving_teamGamesPlayed'] = games_played

    # ========================================================================
    # DEFENSIVE FEATURES
    # ========================================================================

    # Defensive stats are when team is on DEFENSE (defteam)
    def_pass = def_pbp[def_pbp['play_type'] == 'pass'].copy()
    def_rush = def_pbp[def_pbp['play_type'] == 'run'].copy()

    # Basic counting stats (EXACT MATCH)
    features['defensive_sacks'] = def_pbp['sack'].sum()
    features['defensive_interceptions'] = def_pbp['interception'].sum()
    features['defensive_fumblesRecovered'] = def_pbp[(def_pbp['fumble_lost'] == 1) & (def_pbp['fumble_recovery_1_team'] == nfl_team)].shape[0]
    features['defensive_safeties'] = def_pbp['safety'].sum() if 'safety' in def_pbp.columns else 0

    # Tackles for loss (EXACT MATCH) - if available
    if 'tackle_for_loss' in def_pbp.columns:
        features['defensive_tacklesForLoss'] = def_pbp['tackle_for_loss'].sum()
    else:
        features['defensive_tacklesForLoss'] = 0

    # Passes defended (EXACT MATCH) - if available
    if 'pass_defense_1_player_id' in def_pbp.columns:
        features['defensive_passesDefended'] = def_pbp['pass_defense_1_player_id'].notna().sum()
    else:
        features['defensive_passesDefended'] = 0

    # Defensive touchdowns (EXACT MATCH)
    features['defensive_interceptionTouchdowns'] = def_pbp[(def_pbp['interception'] == 1) & (def_pbp['return_touchdown'] == 1)].shape[0] if 'return_touchdown' in def_pbp.columns else 0
    features['defensive_fumbleTouchdowns'] = def_pbp[(def_pbp['fumble_lost'] == 1) & (def_pbp['fumble_recovery_1_team'] == nfl_team) & (def_pbp['return_touchdown'] == 1)].shape[0] if 'return_touchdown' in def_pbp.columns else 0
    features['defensive_totalTouchdowns'] = features['defensive_interceptionTouchdowns'] + features['defensive_fumbleTouchdowns']

    # Defensive yards allowed (PARTIAL MATCH)
    features['defensive_yardsAllowed'] = def_pbp['yards_gained'].sum()
    features['defensive_passingYardsAllowed'] = def_pass['yards_gained'].sum()
    features['defensive_rushingYardsAllowed'] = def_rush['yards_gained'].sum()

    # Per game stats (PARTIAL MATCH)
    if games_played > 0:
        features['defensive_yardsAllowedPerGame'] = features['defensive_yardsAllowed'] / games_played
        features['defensive_passingYardsAllowedPerGame'] = features['defensive_passingYardsAllowed'] / games_played
        features['defensive_rushingYardsAllowedPerGame'] = features['defensive_rushingYardsAllowed'] / games_played
    else:
        features['defensive_yardsAllowedPerGame'] = 0
        features['defensive_passingYardsAllowedPerGame'] = 0
        features['defensive_rushingYardsAllowedPerGame'] = 0

    # Sack yards (EXACT MATCH)
    def_sacks = def_pbp[def_pbp['sack'] == 1]
    features['defensive_sackYards'] = abs(def_sacks['yards_gained'].sum())

    # Tackles (EXACT MATCH) - if available
    if 'solo_tackle' in def_pbp.columns:
        features['defensive_soloTackles'] = def_pbp['solo_tackle'].sum()
        features['defensive_assistTackles'] = def_pbp['assist_tackle'].sum()
        features['defensive_totalTackles'] = features['defensive_soloTackles'] + features['defensive_assistTackles']
    else:
        features['defensive_soloTackles'] = 0
        features['defensive_assistTackles'] = 0
        features['defensive_totalTackles'] = 0

    # Stuffs (EXACT MATCH) - defensive plays for 0 or negative yards
    features['defensive_stuffs'] = len(def_pbp[def_pbp['yards_gained'] <= 0])
    features['defensive_stuffYards'] = abs(def_pbp[def_pbp['yards_gained'] < 0]['yards_gained'].sum())

    # Average stats (PARTIAL MATCH)
    if features['defensive_sacks'] > 0:
        features['defensive_avgSackYards'] = features['defensive_sackYards'] / features['defensive_sacks']
    else:
        features['defensive_avgSackYards'] = 0

    if features['defensive_stuffs'] > 0:
        features['defensive_avgStuffYards'] = features['defensive_stuffYards'] / features['defensive_stuffs']
    else:
        features['defensive_avgStuffYards'] = 0

    # Interception stats (PARTIAL MATCH)
    def_ints = def_pbp[def_pbp['interception'] == 1].copy()
    if 'return_yards' in def_ints.columns:
        features['defensive_interceptionYards'] = def_ints['return_yards'].sum()
        features['defensive_longInterception'] = def_ints['return_yards'].max() if len(def_ints) > 0 else 0
        if features['defensive_interceptions'] > 0:
            features['defensive_avgInterceptionYards'] = features['defensive_interceptionYards'] / features['defensive_interceptions']
        else:
            features['defensive_avgInterceptionYards'] = 0
    else:
        features['defensive_interceptionYards'] = 0
        features['defensive_longInterception'] = 0
        features['defensive_avgInterceptionYards'] = 0

    # Defensive interceptions category (PARTIAL MATCH) - same as defensive stats
    features['defensiveInterceptions_interceptions'] = features['defensive_interceptions']
    features['defensiveInterceptions_interceptionYards'] = features['defensive_interceptionYards']
    features['defensiveInterceptions_interceptionTouchdowns'] = features['defensive_interceptionTouchdowns']

    # Points allowed (PARTIAL MATCH) - from schedule
    features['defensive_pointsAllowed'] = team_schedule['opp_score'].sum()

    # Blocked kicks (PARTIAL MATCH)
    if 'punt_blocked' in def_pbp.columns:
        features['defensive_kicksBlocked'] = def_pbp['punt_blocked'].sum()
    else:
        features['defensive_kicksBlocked'] = 0

    # Passes batted down (PARTIAL MATCH) - same as passes defended
    features['defensive_passesBattedDown'] = features['defensive_passesDefended']

    # Defensive touchdowns (PARTIAL MATCH)
    features['defensive_defensiveTouchdowns'] = features['defensive_totalTouchdowns']

    # Special teams touchdowns (PARTIAL MATCH) - set to 0 (not easily derivable)
    features['defensive_blockedFieldGoalTouchdowns'] = 0
    features['defensive_blockedPuntTouchdowns'] = 0
    features['defensive_blockedPuntEzRecTd'] = 0
    features['defensive_missedFieldGoalReturnTd'] = 0
    features['defensive_miscTouchdowns'] = 0

    # Two-point returns (PARTIAL MATCH)
    if 'defensive_two_point_conv' in def_pbp.columns:
        features['defensive_twoPtReturns'] = def_pbp['defensive_two_point_conv'].sum()
    else:
        features['defensive_twoPtReturns'] = 0

    # One-point safeties (PARTIAL MATCH) - extremely rare, set to 0
    features['defensive_onePtSafetiesMade'] = 0

    # Team games played (PARTIAL MATCH)
    features['defensive_teamGamesPlayed'] = games_played

    # ========================================================================
    # GENERAL FEATURES
    # ========================================================================

    # Fumbles (EXACT MATCH)
    features['general_fumbles'] = team_pbp['fumble'].sum()
    features['general_fumblesLost'] = team_pbp['fumble_lost'].sum()
    features['general_fumblesForced'] = def_pbp['fumble'].sum()  # Fumbles forced by defense
    features['general_fumblesRecovered'] = team_pbp[(team_pbp['fumble'] == 1) & (team_pbp['fumble_recovery_1_team'] == nfl_team)].shape[0]

    # Fumble touchdowns (EXACT MATCH)
    features['general_fumblesTouchdowns'] = team_pbp[(team_pbp['fumble'] == 1) & (team_pbp['return_touchdown'] == 1)].shape[0] if 'return_touchdown' in team_pbp.columns else 0
    features['general_offensiveFumblesTouchdowns'] = team_pbp[(team_pbp['fumble'] == 1) & (team_pbp['return_touchdown'] == 1) & (team_pbp['fumble_recovery_1_team'] == nfl_team)].shape[0] if 'return_touchdown' in team_pbp.columns else 0
    features['general_defensiveFumblesTouchdowns'] = features['defensive_fumbleTouchdowns']

    # Penalties (EXACT MATCH)
    features['general_totalPenalties'] = team_pbp[team_pbp['penalty_team'] == nfl_team].shape[0]
    features['general_totalPenaltyYards'] = team_pbp[team_pbp['penalty_team'] == nfl_team]['penalty_yards'].sum()

    # Games played (PARTIAL MATCH)
    features['general_gamesPlayed'] = games_played

    # Turnover differential (PARTIAL MATCH)
    turnovers_lost = features['general_fumblesLost'] + features['passing_interceptions']
    turnovers_gained = features['defensive_interceptions'] + features['defensive_fumblesRecovered']
    features['general_turnoverDifferential'] = turnovers_gained - turnovers_lost

    # Per game stats (PARTIAL MATCH)
    if games_played > 0:
        features['general_penaltyYardsPerGame'] = features['general_totalPenaltyYards'] / games_played
    else:
        features['general_penaltyYardsPerGame'] = 0

    # Two-point returns (PARTIAL MATCH) - defensive 2-pt conversions
    features['general_offensiveTwoPtReturns'] = 0  # Not available in nfl-data-py

    # ========================================================================
    # DOWNS FEATURES
    # ========================================================================

    # Third downs (EXACT MATCH)
    third_down_plays = team_pbp[team_pbp['down'] == 3]
    features['downs_thirdDownAttempts'] = len(third_down_plays)
    features['downs_thirdDownConversions'] = third_down_plays['first_down'].sum()

    if features['downs_thirdDownAttempts'] > 0:
        features['downs_thirdDownConversionPct'] = (features['downs_thirdDownConversions'] / features['downs_thirdDownAttempts']) * 100
    else:
        features['downs_thirdDownConversionPct'] = 0

    # Fourth downs (EXACT MATCH)
    fourth_down_plays = team_pbp[team_pbp['down'] == 4]
    features['downs_fourthDownAttempts'] = len(fourth_down_plays)
    features['downs_fourthDownConversions'] = fourth_down_plays['first_down'].sum()

    if features['downs_fourthDownAttempts'] > 0:
        features['downs_fourthDownConversionPct'] = (features['downs_fourthDownConversions'] / features['downs_fourthDownAttempts']) * 100
    else:
        features['downs_fourthDownConversionPct'] = 0

    # First downs (PARTIAL MATCH)
    features['downs_firstDowns'] = team_pbp['first_down'].sum()
    features['downs_firstDownPassing'] = pass_plays['first_down'].sum()
    features['downs_firstDownRushing'] = rush_plays['first_down'].sum()
    features['downs_firstDownPenalty'] = team_pbp[(team_pbp['first_down'] == 1) & (team_pbp['first_down_penalty'] == 1)].shape[0] if 'first_down_penalty' in team_pbp.columns else 0

    if len(team_pbp) > 0:
        features['downs_firstDownPercentage'] = (features['downs_firstDowns'] / len(team_pbp)) * 100
    else:
        features['downs_firstDownPercentage'] = 0

    # ========================================================================
    # KICKING FEATURES
    # ========================================================================

    # Field goals (EXACT MATCH)
    fg_plays = team_pbp[team_pbp['play_type'] == 'field_goal']
    features['kicking_fieldGoalAttempts'] = len(fg_plays)
    features['kicking_fieldGoalsMade'] = fg_plays['field_goal_result'].eq('made').sum() if 'field_goal_result' in fg_plays.columns else 0

    if features['kicking_fieldGoalAttempts'] > 0:
        features['kicking_fieldGoalPct'] = (features['kicking_fieldGoalsMade'] / features['kicking_fieldGoalAttempts']) * 100
    else:
        features['kicking_fieldGoalPct'] = 0

    # Extra points (EXACT MATCH)
    xp_plays = team_pbp[team_pbp['play_type'] == 'extra_point']
    features['kicking_extraPointAttempts'] = len(xp_plays)
    features['kicking_extraPointsMade'] = xp_plays['extra_point_result'].eq('good').sum() if 'extra_point_result' in xp_plays.columns else 0

    if features['kicking_extraPointAttempts'] > 0:
        features['kicking_extraPointPct'] = (features['kicking_extraPointsMade'] / features['kicking_extraPointAttempts']) * 100
    else:
        features['kicking_extraPointPct'] = 0

    # Punting (EXACT MATCH)
    punt_plays = team_pbp[team_pbp['play_type'] == 'punt']
    features['kicking_punts'] = len(punt_plays)
    features['kicking_puntYards'] = punt_plays['kick_distance'].sum() if 'kick_distance' in punt_plays.columns else 0

    if features['kicking_punts'] > 0:
        features['kicking_puntAverage'] = features['kicking_puntYards'] / features['kicking_punts']
    else:
        features['kicking_puntAverage'] = 0

    # Touchbacks (EXACT MATCH) - if available
    if 'touchback' in punt_plays.columns:
        features['kicking_puntTouchbacks'] = punt_plays['touchback'].sum()
    else:
        features['kicking_puntTouchbacks'] = 0

    # Fair catches (PARTIAL MATCH)
    if 'punt_fair_catch' in punt_plays.columns:
        features['kicking_fairCatches'] = punt_plays['punt_fair_catch'].sum()
        if features['kicking_punts'] > 0:
            features['kicking_fairCatchPct'] = (features['kicking_fairCatches'] / features['kicking_punts']) * 100
        else:
            features['kicking_fairCatchPct'] = 0
    else:
        features['kicking_fairCatches'] = 0
        features['kicking_fairCatchPct'] = 0

    # Field goals by distance (EXACT MATCH)
    if 'kick_distance' in fg_plays.columns:
        # Attempts by distance
        features['kicking_fieldGoalAttempts1_19'] = len(fg_plays[fg_plays['kick_distance'] < 20])
        features['kicking_fieldGoalAttempts20_29'] = len(fg_plays[(fg_plays['kick_distance'] >= 20) & (fg_plays['kick_distance'] < 30)])
        features['kicking_fieldGoalAttempts30_39'] = len(fg_plays[(fg_plays['kick_distance'] >= 30) & (fg_plays['kick_distance'] < 40)])
        features['kicking_fieldGoalAttempts40_49'] = len(fg_plays[(fg_plays['kick_distance'] >= 40) & (fg_plays['kick_distance'] < 50)])
        features['kicking_fieldGoalAttempts50'] = len(fg_plays[fg_plays['kick_distance'] >= 50])
        features['kicking_fieldGoalAttempts50_59'] = len(fg_plays[(fg_plays['kick_distance'] >= 50) & (fg_plays['kick_distance'] < 60)])
        features['kicking_fieldGoalAttempts60_99'] = len(fg_plays[fg_plays['kick_distance'] >= 60])

        # Made by distance
        fg_made = fg_plays[fg_plays['field_goal_result'] == 'made']
        features['kicking_fieldGoalsMade1_19'] = len(fg_made[fg_made['kick_distance'] < 20])
        features['kicking_fieldGoalsMade20_29'] = len(fg_made[(fg_made['kick_distance'] >= 20) & (fg_made['kick_distance'] < 30)])
        features['kicking_fieldGoalsMade30_39'] = len(fg_made[(fg_made['kick_distance'] >= 30) & (fg_made['kick_distance'] < 40)])
        features['kicking_fieldGoalsMade40_49'] = len(fg_made[(fg_made['kick_distance'] >= 40) & (fg_made['kick_distance'] < 50)])
        features['kicking_fieldGoalsMade50'] = len(fg_made[fg_made['kick_distance'] >= 50])
        features['kicking_fieldGoalsMade50_59'] = len(fg_made[(fg_made['kick_distance'] >= 50) & (fg_made['kick_distance'] < 60)])
        features['kicking_fieldGoalsMade60_99'] = len(fg_made[fg_made['kick_distance'] >= 60])

        # Total yards
        features['kicking_fieldGoalAttemptYards'] = fg_plays['kick_distance'].sum()
        features['kicking_fieldGoalsMadeYards'] = fg_made['kick_distance'].sum()
    else:
        features['kicking_fieldGoalAttempts1_19'] = 0
        features['kicking_fieldGoalAttempts20_29'] = 0
        features['kicking_fieldGoalAttempts30_39'] = 0
        features['kicking_fieldGoalAttempts40_49'] = 0
        features['kicking_fieldGoalAttempts50'] = 0
        features['kicking_fieldGoalAttempts50_59'] = 0
        features['kicking_fieldGoalAttempts60_99'] = 0
        features['kicking_fieldGoalsMade1_19'] = 0
        features['kicking_fieldGoalsMade20_29'] = 0
        features['kicking_fieldGoalsMade30_39'] = 0
        features['kicking_fieldGoalsMade40_49'] = 0
        features['kicking_fieldGoalsMade50'] = 0
        features['kicking_fieldGoalsMade50_59'] = 0
        features['kicking_fieldGoalsMade60_99'] = 0
        features['kicking_fieldGoalAttemptYards'] = 0
        features['kicking_fieldGoalsMadeYards'] = 0

    # Blocked kicks (EXACT MATCH)
    if 'blocked_player_id' in fg_plays.columns:
        features['kicking_fieldGoalsBlocked'] = fg_plays['blocked_player_id'].notna().sum()
    else:
        features['kicking_fieldGoalsBlocked'] = 0

    if features['kicking_fieldGoalAttempts'] > 0:
        features['kicking_fieldGoalsBlockedPct'] = (features['kicking_fieldGoalsBlocked'] / features['kicking_fieldGoalAttempts']) * 100
    else:
        features['kicking_fieldGoalsBlockedPct'] = 0

    # Extra points blocked (EXACT MATCH)
    if 'blocked_player_id' in xp_plays.columns:
        features['kicking_extraPointsBlocked'] = xp_plays['blocked_player_id'].notna().sum()
    else:
        features['kicking_extraPointsBlocked'] = 0

    if features['kicking_extraPointAttempts'] > 0:
        features['kicking_extraPointsBlockedPct'] = (features['kicking_extraPointsBlocked'] / features['kicking_extraPointAttempts']) * 100
    else:
        features['kicking_extraPointsBlockedPct'] = 0

    # Kickoff stats (EXACT MATCH)
    kickoff_plays = team_pbp[team_pbp['play_type'] == 'kickoff']
    if 'kick_distance' in kickoff_plays.columns:
        features['kicking_avgKickoffYards'] = kickoff_plays['kick_distance'].mean() if len(kickoff_plays) > 0 else 0
    else:
        features['kicking_avgKickoffYards'] = 0

    # Kickoff return yards (EXACT MATCH)
    if 'return_yards' in kickoff_plays.columns:
        features['kicking_avgKickoffReturnYards'] = kickoff_plays['return_yards'].mean() if len(kickoff_plays) > 0 else 0
    else:
        features['kicking_avgKickoffReturnYards'] = 0

    # Kickoff stats (EXACT MATCH)
    features['kicking_kickoffs'] = len(kickoff_plays)
    if 'kick_distance' in kickoff_plays.columns:
        features['kicking_kickoffYards'] = kickoff_plays['kick_distance'].sum()
        features['kicking_longKickoff'] = kickoff_plays['kick_distance'].max() if len(kickoff_plays) > 0 else 0
    else:
        features['kicking_kickoffYards'] = 0
        features['kicking_longKickoff'] = 0

    # Kickoff returns (EXACT MATCH)
    if 'return_yards' in kickoff_plays.columns:
        kickoff_returns = kickoff_plays[kickoff_plays['return_yards'].notna()]
        features['kicking_kickoffReturns'] = len(kickoff_returns)
        features['kicking_kickoffReturnYards'] = kickoff_returns['return_yards'].sum()
        if 'return_touchdown' in kickoff_returns.columns:
            features['kicking_kickoffReturnTouchdowns'] = kickoff_returns['return_touchdown'].sum()
        else:
            features['kicking_kickoffReturnTouchdowns'] = 0
    else:
        features['kicking_kickoffReturns'] = 0
        features['kicking_kickoffReturnYards'] = 0
        features['kicking_kickoffReturnTouchdowns'] = 0

    # Field goal missed yards (EXACT MATCH)
    if 'kick_distance' in fg_plays.columns:
        fg_missed = fg_plays[fg_plays['field_goal_result'] != 'made']
        features['kicking_fieldGoalsMissedYards'] = fg_missed['kick_distance'].sum()
    else:
        features['kicking_fieldGoalsMissedYards'] = 0

    # Long field goals (PARTIAL MATCH)
    if 'kick_distance' in fg_plays.columns:
        features['kicking_longFieldGoalAttempt'] = fg_plays['kick_distance'].max() if len(fg_plays) > 0 else 0
        fg_made = fg_plays[fg_plays['field_goal_result'] == 'made']
        features['kicking_longFieldGoalMade'] = fg_made['kick_distance'].max() if len(fg_made) > 0 else 0
    else:
        features['kicking_longFieldGoalAttempt'] = 0
        features['kicking_longFieldGoalMade'] = 0

    # Touchbacks (PARTIAL MATCH) - all touchbacks (kickoffs + punts)
    if 'touchback' in team_pbp.columns:
        features['kicking_touchbacks'] = team_pbp['touchback'].sum()
        if features['kicking_kickoffs'] > 0:
            features['kicking_touchbackPct'] = (features['kicking_touchbacks'] / features['kicking_kickoffs']) * 100
        else:
            features['kicking_touchbackPct'] = 0
    else:
        features['kicking_touchbacks'] = 0
        features['kicking_touchbackPct'] = 0

    # Total kicking points (PARTIAL MATCH)
    features['kicking_totalKickingPoints'] = (features['kicking_fieldGoalsMade'] * 3) + features['kicking_extraPointsMade']

    # Team games played (PARTIAL MATCH)
    features['kicking_teamGamesPlayed'] = games_played

    # ========================================================================
    # PUNTING FEATURES
    # ========================================================================

    # Punting stats (EXACT MATCH)
    features['punting_punts'] = features['kicking_punts']
    features['punting_puntYards'] = features['kicking_puntYards']

    if features['punting_punts'] > 0:
        features['punting_grossAvgPuntYards'] = features['punting_puntYards'] / features['punting_punts']
    else:
        features['punting_grossAvgPuntYards'] = 0

    # Long punt (EXACT MATCH)
    if 'kick_distance' in punt_plays.columns:
        features['punting_longPunt'] = punt_plays['kick_distance'].max() if len(punt_plays) > 0 else 0
    else:
        features['punting_longPunt'] = 0

    # Punts blocked (EXACT MATCH)
    if 'punt_blocked' in punt_plays.columns:
        features['punting_puntsBlocked'] = punt_plays['punt_blocked'].sum()
        if features['punting_punts'] > 0:
            features['punting_puntsBlockedPct'] = (features['punting_puntsBlocked'] / features['punting_punts']) * 100
        else:
            features['punting_puntsBlockedPct'] = 0
    else:
        features['punting_puntsBlocked'] = 0
        features['punting_puntsBlockedPct'] = 0

    # Punt returns (EXACT MATCH)
    if 'return_yards' in punt_plays.columns:
        punt_returns = punt_plays[punt_plays['return_yards'].notna()]
        features['punting_puntReturns'] = len(punt_returns)
        features['punting_puntReturnYards'] = punt_returns['return_yards'].sum()
        if features['punting_puntReturns'] > 0:
            features['punting_avgPuntReturnYards'] = features['punting_puntReturnYards'] / features['punting_puntReturns']
        else:
            features['punting_avgPuntReturnYards'] = 0
    else:
        features['punting_puntReturns'] = 0
        features['punting_puntReturnYards'] = 0
        features['punting_avgPuntReturnYards'] = 0

    # Net average punt yards (EXACT MATCH) - gross yards minus return yards
    if features['punting_punts'] > 0:
        features['punting_netAvgPuntYards'] = (features['punting_puntYards'] - features['punting_puntReturnYards']) / features['punting_punts']
    else:
        features['punting_netAvgPuntYards'] = 0

    # Fair catches (PARTIAL MATCH)
    if 'punt_fair_catch' in punt_plays.columns:
        features['punting_fairCatches'] = punt_plays['punt_fair_catch'].sum()
    else:
        features['punting_fairCatches'] = 0

    # Punts inside 10/20 (EXACT MATCH)
    if 'punt_inside_twenty' in punt_plays.columns:
        features['punting_puntsInside20'] = punt_plays['punt_inside_twenty'].sum()
        if features['punting_punts'] > 0:
            features['punting_puntsInside20Pct'] = (features['punting_puntsInside20'] / features['punting_punts']) * 100
        else:
            features['punting_puntsInside20Pct'] = 0
    else:
        features['punting_puntsInside20'] = 0
        features['punting_puntsInside20Pct'] = 0

    # Punts inside 10 (EXACT MATCH) - need to check yardline
    if 'yardline_100' in punt_plays.columns:
        # After punt, ball is at yardline_100 - kick_distance
        punts_inside_10 = punt_plays[punt_plays['yardline_100'] - punt_plays['kick_distance'] <= 10]
        features['punting_puntsInside10'] = len(punts_inside_10)
        if features['punting_punts'] > 0:
            features['punting_puntsInside10Pct'] = (features['punting_puntsInside10'] / features['punting_punts']) * 100
        else:
            features['punting_puntsInside10Pct'] = 0
    else:
        features['punting_puntsInside10'] = 0
        features['punting_puntsInside10Pct'] = 0

    # Punt touchbacks (PARTIAL MATCH)
    if 'touchback' in punt_plays.columns:
        features['punting_touchbacks'] = punt_plays['touchback'].sum()
        if features['punting_punts'] > 0:
            features['punting_touchbackPct'] = (features['punting_touchbacks'] / features['punting_punts']) * 100
        else:
            features['punting_touchbackPct'] = 0
    else:
        features['punting_touchbacks'] = 0
        features['punting_touchbackPct'] = 0

    # Team games played (PARTIAL MATCH)
    features['punting_teamGamesPlayed'] = games_played

    # ========================================================================
    # MISCELLANEOUS FEATURES
    # ========================================================================

    # First downs (EXACT MATCH) - aliases of downs features
    features['miscellaneous_firstDowns'] = features['downs_firstDowns']
    features['miscellaneous_firstDownsPassing'] = features['downs_firstDownPassing']
    features['miscellaneous_firstDownsRushing'] = features['downs_firstDownRushing']
    features['miscellaneous_firstDownsPenalty'] = features['downs_firstDownPenalty']

    if games_played > 0:
        features['miscellaneous_firstDownsPerGame'] = features['miscellaneous_firstDowns'] / games_played
    else:
        features['miscellaneous_firstDownsPerGame'] = 0

    # Third/fourth downs (EXACT MATCH) - aliases
    features['miscellaneous_thirdDownAttempts'] = features['downs_thirdDownAttempts']
    features['miscellaneous_thirdDownConvs'] = features['downs_thirdDownConversions']
    features['miscellaneous_thirdDownConvPct'] = features['downs_thirdDownConversionPct']
    features['miscellaneous_fourthDownAttempts'] = features['downs_fourthDownAttempts']
    features['miscellaneous_fourthDownConvs'] = features['downs_fourthDownConversions']
    features['miscellaneous_fourthDownConvPct'] = features['downs_fourthDownConversionPct']

    # Fumbles (PARTIAL MATCH) - alias
    features['miscellaneous_fumblesLost'] = features['general_fumblesLost']

    # Penalties (PARTIAL MATCH) - aliases
    features['miscellaneous_totalPenalties'] = features['general_totalPenalties']
    features['miscellaneous_totalPenaltyYards'] = features['general_totalPenaltyYards']

    # Turnovers (PARTIAL MATCH)
    features['miscellaneous_totalGiveaways'] = features['general_fumblesLost'] + features['passing_interceptions']
    features['miscellaneous_totalTakeaways'] = features['defensive_interceptions'] + features['defensive_fumblesRecovered']
    features['miscellaneous_turnOverDifferential'] = features['general_turnoverDifferential']

    # Possession time (EXACT MATCH) - if available
    if 'time_of_possession' in team_pbp.columns:
        features['miscellaneous_possessionTimeSeconds'] = team_pbp['time_of_possession'].sum()
    else:
        features['miscellaneous_possessionTimeSeconds'] = 0

    # Total drives (PARTIAL MATCH) - if available
    if 'drive' in team_pbp.columns:
        features['miscellaneous_totalDrives'] = team_pbp['drive'].nunique()
    else:
        features['miscellaneous_totalDrives'] = 0

    # Redzone efficiency (PARTIAL MATCH) - if available
    if 'yardline_100' in team_pbp.columns:
        redzone_plays = team_pbp[team_pbp['yardline_100'] <= 20]
        redzone_scores = redzone_plays[(redzone_plays['touchdown'] == 1) | (redzone_plays['field_goal_result'] == 'made')]

        if 'drive' in redzone_plays.columns:
            redzone_drives = redzone_plays['drive'].nunique()
            redzone_td_drives = redzone_plays[redzone_plays['touchdown'] == 1]['drive'].nunique()
            redzone_fg_drives = redzone_plays[redzone_plays['field_goal_result'] == 'made']['drive'].nunique()
            redzone_score_drives = redzone_td_drives + redzone_fg_drives

            if redzone_drives > 0:
                features['miscellaneous_redzoneScoringPct'] = (redzone_score_drives / redzone_drives) * 100
                features['miscellaneous_redzoneTouchdownPct'] = (redzone_td_drives / redzone_drives) * 100
                features['miscellaneous_redzoneFieldGoalPct'] = (redzone_fg_drives / redzone_drives) * 100
                features['miscellaneous_redzoneEfficiencyPct'] = features['miscellaneous_redzoneScoringPct']
            else:
                features['miscellaneous_redzoneScoringPct'] = 0
                features['miscellaneous_redzoneTouchdownPct'] = 0
                features['miscellaneous_redzoneFieldGoalPct'] = 0
                features['miscellaneous_redzoneEfficiencyPct'] = 0
        else:
            features['miscellaneous_redzoneScoringPct'] = 0
            features['miscellaneous_redzoneTouchdownPct'] = 0
            features['miscellaneous_redzoneFieldGoalPct'] = 0
            features['miscellaneous_redzoneEfficiencyPct'] = 0
    else:
        features['miscellaneous_redzoneScoringPct'] = 0
        features['miscellaneous_redzoneTouchdownPct'] = 0
        features['miscellaneous_redzoneFieldGoalPct'] = 0
        features['miscellaneous_redzoneEfficiencyPct'] = 0

    # ========================================================================
    # RETURNING FEATURES
    # ========================================================================

    # Kick returns (EXACT MATCH) - when team is returning kickoffs
    kick_return_plays = pbp_reg[(pbp_reg['play_type'] == 'kickoff') & (pbp_reg['return_team'] == nfl_team)]
    if 'return_yards' in kick_return_plays.columns:
        features['returning_kickReturns'] = len(kick_return_plays[kick_return_plays['return_yards'].notna()])
        features['returning_kickReturnYards'] = kick_return_plays['return_yards'].sum()
        features['returning_longKickReturn'] = kick_return_plays['return_yards'].max() if len(kick_return_plays) > 0 else 0
        if features['returning_kickReturns'] > 0:
            features['returning_yardsPerKickReturn'] = features['returning_kickReturnYards'] / features['returning_kickReturns']
        else:
            features['returning_yardsPerKickReturn'] = 0

        if 'return_touchdown' in kick_return_plays.columns:
            features['returning_kickReturnTouchdowns'] = kick_return_plays['return_touchdown'].sum()
        else:
            features['returning_kickReturnTouchdowns'] = 0
    else:
        features['returning_kickReturns'] = 0
        features['returning_kickReturnYards'] = 0
        features['returning_longKickReturn'] = 0
        features['returning_yardsPerKickReturn'] = 0
        features['returning_kickReturnTouchdowns'] = 0

    # Kick return fair catches (EXACT MATCH)
    if 'kickoff_fair_catch' in kick_return_plays.columns:
        features['returning_kickReturnFairCatches'] = kick_return_plays['kickoff_fair_catch'].sum()
        if features['returning_kickReturns'] > 0:
            features['returning_kickReturnFairCatchPct'] = (features['returning_kickReturnFairCatches'] / features['returning_kickReturns']) * 100
        else:
            features['returning_kickReturnFairCatchPct'] = 0
    else:
        features['returning_kickReturnFairCatches'] = 0
        features['returning_kickReturnFairCatchPct'] = 0

    # Kick return fumbles (EXACT MATCH)
    if 'fumble' in kick_return_plays.columns:
        features['returning_kickReturnFumbles'] = kick_return_plays['fumble'].sum()
        if 'fumble_lost' in kick_return_plays.columns:
            features['returning_kickReturnFumblesLost'] = kick_return_plays['fumble_lost'].sum()
        else:
            features['returning_kickReturnFumblesLost'] = 0
    else:
        features['returning_kickReturnFumbles'] = 0
        features['returning_kickReturnFumblesLost'] = 0

    # Punt returns (EXACT MATCH) - when team is returning punts
    punt_return_plays = pbp_reg[(pbp_reg['play_type'] == 'punt') & (pbp_reg['return_team'] == nfl_team)]
    if 'return_yards' in punt_return_plays.columns:
        features['returning_puntReturns'] = len(punt_return_plays[punt_return_plays['return_yards'].notna()])
        features['returning_puntReturnYards'] = punt_return_plays['return_yards'].sum()
        features['returning_longPuntReturn'] = punt_return_plays['return_yards'].max() if len(punt_return_plays) > 0 else 0
        if features['returning_puntReturns'] > 0:
            features['returning_yardsPerPuntReturn'] = features['returning_puntReturnYards'] / features['returning_puntReturns']
        else:
            features['returning_yardsPerPuntReturn'] = 0

        if 'return_touchdown' in punt_return_plays.columns:
            features['returning_puntReturnTouchdowns'] = punt_return_plays['return_touchdown'].sum()
        else:
            features['returning_puntReturnTouchdowns'] = 0
    else:
        features['returning_puntReturns'] = 0
        features['returning_puntReturnYards'] = 0
        features['returning_longPuntReturn'] = 0
        features['returning_yardsPerPuntReturn'] = 0
        features['returning_puntReturnTouchdowns'] = 0

    # Punt return fair catches (EXACT MATCH)
    if 'punt_fair_catch' in punt_return_plays.columns:
        features['returning_puntReturnFairCatches'] = punt_return_plays['punt_fair_catch'].sum()
        if features['returning_puntReturns'] > 0:
            features['returning_puntReturnFairCatchPct'] = (features['returning_puntReturnFairCatches'] / features['returning_puntReturns']) * 100
        else:
            features['returning_puntReturnFairCatchPct'] = 0
    else:
        features['returning_puntReturnFairCatches'] = 0
        features['returning_puntReturnFairCatchPct'] = 0

    # Punt return fumbles (EXACT MATCH)
    if 'fumble' in punt_return_plays.columns:
        features['returning_puntReturnFumbles'] = punt_return_plays['fumble'].sum()
        if 'fumble_lost' in punt_return_plays.columns:
            features['returning_puntReturnFumblesLost'] = punt_return_plays['fumble_lost'].sum()
        else:
            features['returning_puntReturnFumblesLost'] = 0
    else:
        features['returning_puntReturnFumbles'] = 0
        features['returning_puntReturnFumblesLost'] = 0

    # Punt returns started inside 10/20 (EXACT MATCH)
    if 'yardline_100' in punt_return_plays.columns:
        features['returning_puntReturnsStartedInsideThe10'] = len(punt_return_plays[punt_return_plays['yardline_100'] >= 90])
        features['returning_puntReturnsStartedInsideThe20'] = len(punt_return_plays[punt_return_plays['yardline_100'] >= 80])
    else:
        features['returning_puntReturnsStartedInsideThe10'] = 0
        features['returning_puntReturnsStartedInsideThe20'] = 0

    # Fumble recoveries (PARTIAL MATCH) - defensive fumble recoveries
    features['returning_fumbleRecoveries'] = features['defensive_fumblesRecovered']
    features['returning_fumbleRecoveryYards'] = 0  # Not easily available
    features['returning_oppFumbleRecoveries'] = features['defensive_fumblesRecovered']
    features['returning_oppFumbleRecoveryYards'] = 0  # Not easily available

    # Defensive fumble returns (EXACT MATCH)
    def_fumble_returns = def_pbp[def_pbp['fumble_recovery_1_team'] == nfl_team]
    if 'return_yards' in def_fumble_returns.columns:
        features['returning_defFumbleReturns'] = len(def_fumble_returns)
        features['returning_defFumbleReturnYards'] = def_fumble_returns['return_yards'].sum()
    else:
        features['returning_defFumbleReturns'] = 0
        features['returning_defFumbleReturnYards'] = 0

    # Special team fumble returns (EXACT MATCH) - set to 0 (not easily derivable)
    features['returning_specialTeamFumbleReturns'] = 0
    features['returning_specialTeamFumbleReturnYards'] = 0
    features['returning_oppSpecialTeamFumbleReturns'] = 0
    features['returning_oppSpecialTeamFumbleReturnYards'] = 0
    features['returning_miscFumbleReturns'] = 0
    features['returning_miscFumbleReturnYards'] = 0

    # Yards per return (EXACT MATCH) - average of kick and punt returns
    total_returns = features['returning_kickReturns'] + features['returning_puntReturns']
    total_return_yards = features['returning_kickReturnYards'] + features['returning_puntReturnYards']
    if total_returns > 0:
        features['returning_yardsPerReturn'] = total_return_yards / total_returns
    else:
        features['returning_yardsPerReturn'] = 0

    # Team games played (PARTIAL MATCH)
    features['returning_teamGamesPlayed'] = games_played

    # ========================================================================
    # SCORING FEATURES
    # ========================================================================

    # Scoring features (EXACT MATCH) - aliases of existing features
    features['scoring_passingTouchdowns'] = features['passing_passingTouchdowns']
    features['scoring_rushingTouchdowns'] = features['rushing_rushingTouchdowns']
    features['scoring_receivingTouchdowns'] = features['receiving_receivingTouchdowns']
    features['scoring_returnTouchdowns'] = features['returning_kickReturnTouchdowns'] + features['returning_puntReturnTouchdowns']
    features['scoring_defensivePoints'] = features['defensive_totalTouchdowns'] * 6  # Defensive TDs worth 6 points
    features['scoring_fieldGoals'] = features['kicking_fieldGoalsMade']
    features['scoring_kickExtraPoints'] = features['kicking_extraPointAttempts']
    features['scoring_kickExtraPointsMade'] = features['kicking_extraPointsMade']
    features['scoring_onePtSafetiesMade'] = 0  # Extremely rare
    features['scoring_miscPoints'] = 0  # Not easily derivable

    # Total touchdowns (EXACT MATCH)
    features['scoring_totalTouchdowns'] = (features['scoring_passingTouchdowns'] +
                                           features['scoring_rushingTouchdowns'] +
                                           features['scoring_receivingTouchdowns'] +
                                           features['scoring_returnTouchdowns'])

    # Two-point conversions (EXACT MATCH)
    features['scoring_twoPointPassConvs'] = features['passing_twoPointPassConvs']
    features['scoring_twoPointRushConvs'] = features['rushing_twoPointRushConvs']
    features['scoring_twoPointRecConvs'] = features['receiving_twoPointRecConvs']
    features['scoring_totalTwoPointConvs'] = (features['scoring_twoPointPassConvs'] +
                                              features['scoring_twoPointRushConvs'] +
                                              features['scoring_twoPointRecConvs'])

    # Total points (EXACT MATCH) - calculated from all scoring sources
    features['scoring_totalPoints'] = (features['scoring_totalTouchdowns'] * 6 +
                                       features['scoring_fieldGoals'] * 3 +
                                       features['scoring_kickExtraPointsMade'] +
                                       features['scoring_totalTwoPointConvs'] * 2 +
                                       features['scoring_defensivePoints'])

    # Total points per game (EXACT MATCH)
    if games_played > 0:
        features['scoring_totalPointsPerGame'] = features['scoring_totalPoints'] / games_played
    else:
        features['scoring_totalPointsPerGame'] = 0

    # ========================================================================
    # TEAM RECORDS (from schedules)
    # ========================================================================

    # Win/loss records (PARTIAL MATCH)
    features['total_wins'] = team_schedule['win'].sum()
    features['total_losses'] = team_schedule['loss'].sum()
    total_games = features['total_wins'] + features['total_losses']
    features['total_winPercentage'] = (features['total_wins'] / total_games * 100) if total_games > 0 else 0

    # Points (PARTIAL MATCH)
    features['total_pointsFor'] = team_schedule['team_score'].sum()
    features['total_pointsAgainst'] = team_schedule['opp_score'].sum()
    features['total_pointDifferential'] = features['total_pointsFor'] - features['total_pointsAgainst']

    # Home/Away records (PARTIAL MATCH)
    home_games = team_schedule[team_schedule['is_home']]
    away_games = team_schedule[~team_schedule['is_home']]

    # Check for ties (score is equal)
    team_schedule['tie'] = team_schedule['team_score'] == team_schedule['opp_score']
    home_games['tie'] = home_games['team_score'] == home_games['opp_score']
    away_games['tie'] = away_games['team_score'] == away_games['opp_score']

    # Check for OT games (if available in schedules)
    if 'overtime' in team_schedule.columns:
        team_schedule['ot_loss'] = (team_schedule['overtime'] == 1) & (team_schedule['loss'])
        home_games['ot_loss'] = (home_games['overtime'] == 1) & (home_games['loss'])
        away_games['ot_loss'] = (away_games['overtime'] == 1) & (away_games['loss'])
    else:
        team_schedule['ot_loss'] = False
        home_games['ot_loss'] = False
        away_games['ot_loss'] = False

    # Total records
    features['total_ties'] = team_schedule['tie'].sum()
    features['total_OTLosses'] = team_schedule['ot_loss'].sum()

    # OT wins (EXACT MATCH)
    if 'overtime' in team_schedule.columns:
        features['total_OTWins'] = ((team_schedule['overtime'] == 1) & (team_schedule['win'])).sum()
    else:
        features['total_OTWins'] = 0

    # Additional total features (EXACT MATCH)
    features['total_gamesPlayed'] = games_played
    features['total_winPercent'] = features['total_winPercentage']
    features['total_points'] = features['total_pointsFor']
    features['total_differential'] = features['total_pointDifferential']

    if games_played > 0:
        features['total_avgPointsFor'] = features['total_pointsFor'] / games_played
        features['total_avgPointsAgainst'] = features['total_pointsAgainst'] / games_played
    else:
        features['total_avgPointsFor'] = 0
        features['total_avgPointsAgainst'] = 0

    # Division/Conference records (EXACT MATCH) - set to 0 (not easily derivable without division/conference data)
    features['total_divisionWins'] = 0
    features['total_divisionLosses'] = 0
    features['total_divisionTies'] = 0
    features['total_divisionWinPercent'] = 0
    features['total_divisionRecord'] = "0-0-0"
    features['total_leagueWinPercent'] = features['total_winPercent']
    features['total_playoffSeed'] = 0
    features['total_clincher'] = ""
    features['total_gamesBehind'] = 0
    features['total_streak'] = ""

    # Conference records (EXACT MATCH) - set to 0
    features['vsconf_wins'] = 0
    features['vsconf_losses'] = 0
    features['vsconf_ties'] = 0
    features['vsconf_OTLosses'] = 0
    features['vsconf_leagueWinPercent'] = 0

    # Division records (EXACT MATCH) - set to 0
    features['vsdiv_wins'] = 0
    features['vsdiv_divisionWins'] = 0
    features['vsdiv_losses'] = 0
    features['vsdiv_ties'] = 0
    features['vsdiv_OTLosses'] = 0
    features['vsdiv_divisionLosses'] = 0
    features['vsdiv_divisionTies'] = 0
    features['vsdiv_divisionWinPercent'] = 0

    # Home records
    features['home_wins'] = home_games['win'].sum()
    features['home_losses'] = home_games['loss'].sum()
    features['home_ties'] = home_games['tie'].sum()
    features['home_OTLosses'] = home_games['ot_loss'].sum()
    features['home_winPercentage'] = (features['home_wins'] / len(home_games) * 100) if len(home_games) > 0 else 0
    features['home_winPercent'] = features['home_winPercentage']  # Alias
    features['home_pointsFor'] = home_games['team_score'].sum()
    features['home_pointsAgainst'] = home_games['opp_score'].sum()

    # Away records
    features['away_wins'] = away_games['win'].sum()
    features['away_losses'] = away_games['loss'].sum()
    features['away_ties'] = away_games['tie'].sum()
    features['away_OTLosses'] = away_games['ot_loss'].sum()
    features['away_winPercentage'] = (features['away_wins'] / len(away_games) * 100) if len(away_games) > 0 else 0
    features['away_winPercent'] = features['away_winPercentage']  # Alias
    features['away_pointsFor'] = away_games['team_score'].sum()
    features['away_pointsAgainst'] = away_games['opp_score'].sum()

    # Road records (EXACT MATCH) - aliases for away records
    features['road_wins'] = features['away_wins']
    features['road_losses'] = features['away_losses']
    features['road_ties'] = features['away_ties']
    features['road_OTLosses'] = features['away_OTLosses']
    features['road_winPercent'] = features['away_winPercent']

    # ========================================================================
    # AGGREGATE/TOTAL FEATURES
    # ========================================================================

    # Total offensive plays (PARTIAL MATCH)
    features['passing_totalOffensivePlays'] = len(team_pbp[team_pbp['play_type'].isin(['pass', 'run'])])
    features['rushing_totalOffensivePlays'] = features['passing_totalOffensivePlays']
    features['receiving_totalOffensivePlays'] = features['passing_totalOffensivePlays']

    # Total points (PARTIAL MATCH) - same as total_pointsFor
    features['passing_totalPoints'] = features['total_pointsFor']
    features['rushing_totalPoints'] = features['total_pointsFor']
    features['receiving_totalPoints'] = features['total_pointsFor']

    # Total points per game (PARTIAL MATCH)
    if games_played > 0:
        features['passing_totalPointsPerGame'] = features['total_pointsFor'] / games_played
        features['rushing_totalPointsPerGame'] = features['total_pointsFor'] / games_played
        features['receiving_totalPointsPerGame'] = features['total_pointsFor'] / games_played
    else:
        features['passing_totalPointsPerGame'] = 0
        features['rushing_totalPointsPerGame'] = 0
        features['receiving_totalPointsPerGame'] = 0

    # Total yards (PARTIAL MATCH)
    features['passing_totalYards'] = features['passing_passingYards'] + features['rushing_rushingYards']
    features['rushing_totalYards'] = features['passing_totalYards']
    features['receiving_totalYards'] = features['passing_totalYards']

    # Net total yards (PARTIAL MATCH) - includes sack yards
    features['passing_netTotalYards'] = features['passing_netPassingYards'] + features['rushing_rushingYards']
    features['rushing_netTotalYards'] = features['passing_netTotalYards']
    features['receiving_netTotalYards'] = features['passing_netTotalYards']

    # Net yards per game (PARTIAL MATCH)
    if games_played > 0:
        features['passing_netYardsPerGame'] = features['passing_netTotalYards'] / games_played
        features['rushing_netYardsPerGame'] = features['passing_netTotalYards'] / games_played
        features['receiving_netYardsPerGame'] = features['passing_netTotalYards'] / games_played
    else:
        features['passing_netYardsPerGame'] = 0
        features['rushing_netYardsPerGame'] = 0
        features['receiving_netYardsPerGame'] = 0

    # Yards from scrimmage (PARTIAL MATCH) - passing + rushing yards
    features['passing_totalYardsFromScrimmage'] = features['passing_passingYards'] + features['rushing_rushingYards']
    features['rushing_totalYardsFromScrimmage'] = features['passing_totalYardsFromScrimmage']
    features['receiving_totalYardsFromScrimmage'] = features['passing_totalYardsFromScrimmage']

    # Yards from scrimmage per game (PARTIAL MATCH)
    if games_played > 0:
        features['passing_yardsFromScrimmagePerGame'] = features['passing_totalYardsFromScrimmage'] / games_played
        features['rushing_yardsFromScrimmagePerGame'] = features['passing_totalYardsFromScrimmage'] / games_played
        features['receiving_yardsFromScrimmagePerGame'] = features['passing_totalYardsFromScrimmage'] / games_played
    else:
        features['passing_yardsFromScrimmagePerGame'] = 0
        features['rushing_yardsFromScrimmagePerGame'] = 0
        features['receiving_yardsFromScrimmagePerGame'] = 0

    # Miscellaneous yards (PARTIAL MATCH) - set to 0 (not available)
    features['passing_miscYards'] = 0
    features['rushing_miscYards'] = 0
    features['receiving_miscYards'] = 0

    return features