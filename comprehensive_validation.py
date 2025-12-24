"""Comprehensive validation of 50% EXACT MATCH + 50% PARTIAL MATCH features"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
from team_abbreviation_mapping import espn_to_nfl_data_py, nfl_data_py_to_espn

print("=" * 120)
print("COMPREHENSIVE FEATURE VALIDATION")
print("=" * 120)

# Load ESPN data
print("\n[1/5] Loading ESPN data...")
espn_stats = pd.read_parquet('data/espn_raw/team_stats_2024.parquet')
espn_records = pd.read_parquet('data/espn_raw/team_records_2024.parquet')
espn = pd.merge(espn_stats, espn_records, on='team', how='outer')
espn = espn.set_index('team').sort_index()
print(f"  ✅ Loaded ESPN data: {len(espn)} teams, {len(espn.columns)} features")

# Load nfl-data-py
print("\n[2/5] Loading nfl-data-py data...")
pbp = pd.read_parquet('data/cache/pbp_2024.parquet')
schedules = pd.read_parquet('data/cache/schedules_2024.parquet')
print(f"  ✅ Loaded play-by-play: {len(pbp):,} plays")
print(f"  ✅ Loaded schedules: {len(schedules)} games")

# Filter to regular season only (weeks 1-18)
pbp_reg = pbp[pbp['week'] <= 18].copy()
print(f"  ✅ Filtered to regular season: {len(pbp_reg):,} plays")

# Load feature mapping
print("\n[3/5] Loading feature mapping...")
mapping = pd.read_csv('results/comprehensive_feature_mapping.csv')
exact_match = mapping[mapping['category'] == 'EXACT MATCH']
partial_match = mapping[mapping['category'] == 'PARTIAL MATCH']
print(f"  ✅ EXACT MATCH features: {len(exact_match)}")
print(f"  ✅ PARTIAL MATCH features: {len(partial_match)}")

# Select features to test (50% EXACT, 50% PARTIAL)
print("\n[4/5] Selecting features to test...")

# EXACT MATCH features (select 50 diverse ones)
exact_features = [
    # Passing (10)
    'passing_passingYards', 'passing_passingAttempts', 'passing_completions',
    'passing_passingTouchdowns', 'passing_interceptions', 'passing_sacks',
    'passing_sackYardsLost', 'passing_longPassing', 'passing_20+PassingPlays',
    'passing_40+PassingPlays',
    
    # Rushing (10)
    'rushing_rushingYards', 'rushing_rushingAttempts', 'rushing_rushingTouchdowns',
    'rushing_longRushing', 'rushing_20+RushingPlays', 'rushing_40+RushingPlays',
    'rushing_rushingFirstDowns', 'rushing_stuffs', 'rushing_stuffsPercentage',
    'rushing_rushingFirstDownPercentage',
    
    # Receiving (10)
    'receiving_receptions', 'receiving_receivingYards', 'receiving_receivingTouchdowns',
    'receiving_longReception', 'receiving_20+ReceivingPlays', 'receiving_40+ReceivingPlays',
    'receiving_receivingFirstDowns', 'receiving_receivingFirstDownPercentage',
    'receiving_targets', 'receiving_catchPercentage',
    
    # Defensive (10)
    'defensive_sacks', 'defensive_interceptions', 'defensive_fumblesRecovered',
    'defensive_tacklesForLoss', 'defensive_passesDefended', 'defensive_safeties',
    'defensive_touchdowns', 'defensive_interceptionTouchdowns',
    'defensive_fumbleReturnTouchdowns', 'defensive_blockedKickTouchdowns',
    
    # General (5)
    'general_fumbles', 'general_fumblesLost', 'general_totalPenalties',
    'general_totalPenaltyYards', 'general_gamesPlayed',
    
    # Downs (5)
    'downs_thirdDownAttempts', 'downs_thirdDownConversions',
    'downs_fourthDownAttempts', 'downs_fourthDownConversions',
    'downs_firstDowns',
]

# PARTIAL MATCH features (select 50 diverse ones)
partial_features = [
    # Calculated rates/percentages (20)
    'passing_completionPct', 'passing_yardsPerPassAttempt', 'passing_adjustedYardsPerPassAttempt',
    'passing_netYardsPerPassAttempt', 'passing_sackPercentage', 'passing_touchdownPercentage',
    'passing_interceptionPercentage', 'rushing_yardsPerRushAttempt',
    'receiving_yardsPerReception', 'receiving_yardsPerTarget',
    'downs_thirdDownConversionPercentage', 'downs_fourthDownConversionPercentage',
    'downs_firstDownPercentage', 'general_turnoverDifferential',
    'general_penaltyYardsPerGame', 'general_pointsPerGame',
    'scoring_pointsPerGame', 'scoring_pointsAgainstPerGame',
    'miscellaneous_timeOfPossessionSeconds', 'miscellaneous_timeOfPossessionSecondsPerGame',
    
    # Team records (20)
    'total_wins', 'total_losses', 'total_winPercentage', 'total_pointsFor',
    'total_pointsAgainst', 'total_pointDifferential', 'home_wins', 'home_losses',
    'away_wins', 'away_losses', 'division_wins', 'division_losses',
    'conference_wins', 'conference_losses', 'streak', 'vsAP25_wins',
    'vsAP25_losses', 'vsUSA_wins', 'vsUSA_losses', 'lastTenGames',
    
    # Advanced metrics (10)
    'passing_QBRating', 'passing_adjustedNetYardsPerPassAttempt',
    'rushing_yardsAfterContact', 'rushing_yardsAfterContactPerAttempt',
    'receiving_yardsAfterCatch', 'receiving_yardsAfterCatchPerReception',
    'defensive_thirdDownConversionPercentageDefense', 'defensive_fourthDownConversionPercentageDefense',
    'defensive_redZoneAttempts', 'defensive_redZoneConversions',
]

# Filter to features that exist in ESPN data
exact_features = [f for f in exact_features if f in espn.columns]
partial_features = [f for f in partial_features if f in espn.columns]

print(f"  ✅ Selected {len(exact_features)} EXACT MATCH features (available in ESPN)")
print(f"  ✅ Selected {len(partial_features)} PARTIAL MATCH features (available in ESPN)")
print(f"  ✅ Total features to validate: {len(exact_features) + len(partial_features)}")

print("\n[5/5] Deriving features from nfl-data-py...")
print("  (This will take a few minutes...)")
print()

def derive_all_features(team: str, pbp_reg: pd.DataFrame, schedules: pd.DataFrame) -> dict:
    """Derive all features for a team

    Args:
        team: ESPN team abbreviation (e.g., 'LAR', 'WSH')
        pbp_reg: Play-by-play data (uses nfl-data-py abbreviations)
        schedules: Schedule data (uses nfl-data-py abbreviations)
    """
    features = {'team': team}

    # Convert ESPN team abbreviation to nfl-data-py abbreviation
    nfl_team = espn_to_nfl_data_py(team)

    # Filter to team's plays (regular season only)
    team_pbp = pbp_reg[pbp_reg['posteam'] == nfl_team].copy()
    pass_plays = team_pbp[team_pbp['play_type'] == 'pass'].copy()
    rush_plays = team_pbp[team_pbp['play_type'] == 'run'].copy()

    # === PASSING FEATURES ===
    # Basic counting stats
    features['passing_completions'] = pass_plays['complete_pass'].sum()
    features['passing_passingAttempts'] = pass_plays['pass_attempt'].sum()
    features['passing_passingTouchdowns'] = pass_plays['pass_touchdown'].sum()
    features['passing_interceptions'] = pass_plays['interception'].sum()

    # Passing yards - ESPN DOES include sack yards lost
    # Net passing yards = gross passing yards - sack yards lost
    gross_pass_yards = pass_plays['yards_gained'].sum()
    sack_plays = team_pbp[team_pbp['sack'] == 1]
    sack_yards_lost = abs(sack_plays['yards_gained'].sum())  # Make positive
    features['passing_passingYards'] = gross_pass_yards - sack_yards_lost

    # Sacks
    features['passing_sacks'] = sack_plays['sack'].sum()
    features['passing_sackYardsLost'] = sack_yards_lost

    # Long plays
    features['passing_longPassing'] = pass_plays['yards_gained'].max() if len(pass_plays) > 0 else 0
    features['passing_20+PassingPlays'] = len(pass_plays[pass_plays['yards_gained'] >= 20])
    features['passing_40+PassingPlays'] = len(pass_plays[pass_plays['yards_gained'] >= 40])

    # Calculated passing stats
    if features['passing_passingAttempts'] > 0:
        features['passing_completionPct'] = (features['passing_completions'] / features['passing_passingAttempts']) * 100
        features['passing_yardsPerPassAttempt'] = features['passing_passingYards'] / features['passing_passingAttempts']
        features['passing_touchdownPercentage'] = (features['passing_passingTouchdowns'] / features['passing_passingAttempts']) * 100
        features['passing_interceptionPercentage'] = (features['passing_interceptions'] / features['passing_passingAttempts']) * 100
    else:
        features['passing_completionPct'] = 0
        features['passing_yardsPerPassAttempt'] = 0
        features['passing_touchdownPercentage'] = 0
        features['passing_interceptionPercentage'] = 0

    # Sack percentage
    total_dropbacks = features['passing_passingAttempts'] + features['passing_sacks']
    if total_dropbacks > 0:
        features['passing_sackPercentage'] = (features['passing_sacks'] / total_dropbacks) * 100
    else:
        features['passing_sackPercentage'] = 0

    # Net yards per pass attempt (yards - sack yards) / (attempts + sacks)
    if total_dropbacks > 0:
        features['passing_netYardsPerPassAttempt'] = features['passing_passingYards'] / total_dropbacks
    else:
        features['passing_netYardsPerPassAttempt'] = 0

    # Adjusted yards per attempt: (yards + 20*TD - 45*INT) / attempts
    if features['passing_passingAttempts'] > 0:
        adj_yards = features['passing_passingYards'] + (20 * features['passing_passingTouchdowns']) - (45 * features['passing_interceptions'])
        features['passing_adjustedYardsPerPassAttempt'] = adj_yards / features['passing_passingAttempts']
    else:
        features['passing_adjustedYardsPerPassAttempt'] = 0

    # Adjusted net yards per attempt: (yards + 20*TD - 45*INT - sack_yards) / (attempts + sacks)
    if total_dropbacks > 0:
        adj_net_yards = features['passing_passingYards'] + (20 * features['passing_passingTouchdowns']) - (45 * features['passing_interceptions'])
        features['passing_adjustedNetYardsPerPassAttempt'] = adj_net_yards / total_dropbacks
    else:
        features['passing_adjustedNetYardsPerPassAttempt'] = 0

    # QB Rating (NFL passer rating formula)
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

    # === RUSHING FEATURES ===
    features['rushing_rushingAttempts'] = rush_plays['rush_attempt'].sum()
    features['rushing_rushingYards'] = rush_plays['yards_gained'].sum()
    features['rushing_rushingTouchdowns'] = rush_plays['rush_touchdown'].sum()
    features['rushing_longRushing'] = rush_plays['yards_gained'].max() if len(rush_plays) > 0 else 0
    features['rushing_20+RushingPlays'] = len(rush_plays[rush_plays['yards_gained'] >= 20])
    features['rushing_40+RushingPlays'] = len(rush_plays[rush_plays['yards_gained'] >= 40])

    # Rushing first downs
    features['rushing_rushingFirstDowns'] = rush_plays['first_down'].sum()

    # Stuffs (rushes for 0 or negative yards)
    stuffs = rush_plays[rush_plays['yards_gained'] <= 0]
    features['rushing_stuffs'] = len(stuffs)

    if features['rushing_rushingAttempts'] > 0:
        features['rushing_yardsPerRushAttempt'] = features['rushing_rushingYards'] / features['rushing_rushingAttempts']
        features['rushing_stuffsPercentage'] = (features['rushing_stuffs'] / features['rushing_rushingAttempts']) * 100
        features['rushing_rushingFirstDownPercentage'] = (features['rushing_rushingFirstDowns'] / features['rushing_rushingAttempts']) * 100
    else:
        features['rushing_yardsPerRushAttempt'] = 0
        features['rushing_stuffsPercentage'] = 0
        features['rushing_rushingFirstDownPercentage'] = 0

    # === RECEIVING FEATURES ===
    # Note: Receiving stats are from the OFFENSE perspective (team's receivers catching passes)
    features['receiving_receptions'] = pass_plays['complete_pass'].sum()  # Same as completions
    features['receiving_receivingYards'] = pass_plays[pass_plays['complete_pass'] == 1]['yards_gained'].sum()
    features['receiving_receivingTouchdowns'] = pass_plays['pass_touchdown'].sum()
    features['receiving_longReception'] = pass_plays[pass_plays['complete_pass'] == 1]['yards_gained'].max() if len(pass_plays[pass_plays['complete_pass'] == 1]) > 0 else 0
    features['receiving_20+ReceivingPlays'] = len(pass_plays[(pass_plays['complete_pass'] == 1) & (pass_plays['yards_gained'] >= 20)])
    features['receiving_40+ReceivingPlays'] = len(pass_plays[(pass_plays['complete_pass'] == 1) & (pass_plays['yards_gained'] >= 40)])

    # Receiving first downs
    features['receiving_receivingFirstDowns'] = pass_plays[pass_plays['complete_pass'] == 1]['first_down'].sum()

    # Targets (all pass attempts)
    features['receiving_targets'] = features['passing_passingAttempts']

    if features['receiving_targets'] > 0:
        features['receiving_catchPercentage'] = (features['receiving_receptions'] / features['receiving_targets']) * 100
    else:
        features['receiving_catchPercentage'] = 0

    if features['receiving_receptions'] > 0:
        features['receiving_yardsPerReception'] = features['receiving_receivingYards'] / features['receiving_receptions']
        features['receiving_receivingFirstDownPercentage'] = (features['receiving_receivingFirstDowns'] / features['receiving_receptions']) * 100
    else:
        features['receiving_yardsPerReception'] = 0
        features['receiving_receivingFirstDownPercentage'] = 0

    if features['receiving_targets'] > 0:
        features['receiving_yardsPerTarget'] = features['receiving_receivingYards'] / features['receiving_targets']
    else:
        features['receiving_yardsPerTarget'] = 0

    # === DEFENSIVE FEATURES ===
    # Defensive stats are when team is on DEFENSE (defteam)
    def_pbp = pbp_reg[pbp_reg['defteam'] == nfl_team].copy()
    def_pass = def_pbp[def_pbp['play_type'] == 'pass'].copy()

    features['defensive_sacks'] = def_pbp['sack'].sum()
    features['defensive_interceptions'] = def_pbp['interception'].sum()
    features['defensive_fumblesRecovered'] = def_pbp[(def_pbp['fumble_lost'] == 1) & (def_pbp['fumble_recovery_1_team'] == nfl_team)].shape[0]
    features['defensive_tacklesForLoss'] = def_pbp['tackle_for_loss'].sum() if 'tackle_for_loss' in def_pbp.columns else 0
    features['defensive_passesDefended'] = def_pbp['pass_defense_1_player_id'].notna().sum() if 'pass_defense_1_player_id' in def_pbp.columns else 0
    features['defensive_safeties'] = def_pbp[def_pbp['safety'] == 1].shape[0]

    # Defensive touchdowns
    features['defensive_interceptionTouchdowns'] = def_pbp[(def_pbp['interception'] == 1) & (def_pbp['return_touchdown'] == 1)].shape[0]
    features['defensive_fumbleReturnTouchdowns'] = def_pbp[(def_pbp['fumble_lost'] == 1) & (def_pbp['return_touchdown'] == 1)].shape[0]
    features['defensive_blockedKickTouchdowns'] = 0  # Not easily available in pbp
    features['defensive_touchdowns'] = features['defensive_interceptionTouchdowns'] + features['defensive_fumbleReturnTouchdowns']

    # === GENERAL FEATURES ===
    features['general_fumbles'] = team_pbp['fumble'].sum()
    features['general_fumblesLost'] = team_pbp['fumble_lost'].sum()
    features['general_totalPenalties'] = team_pbp[team_pbp['penalty_team'] == nfl_team].shape[0]
    features['general_totalPenaltyYards'] = team_pbp[team_pbp['penalty_team'] == nfl_team]['penalty_yards'].sum()
    features['general_gamesPlayed'] = len(team_pbp['game_id'].unique())

    # Turnover differential
    turnovers_lost = features['general_fumblesLost'] + features['passing_interceptions']
    turnovers_gained = features['defensive_interceptions'] + features['defensive_fumblesRecovered']
    features['general_turnoverDifferential'] = turnovers_gained - turnovers_lost

    # Per game stats
    if features['general_gamesPlayed'] > 0:
        features['general_penaltyYardsPerGame'] = features['general_totalPenaltyYards'] / features['general_gamesPlayed']
    else:
        features['general_penaltyYardsPerGame'] = 0

    # === DOWNS FEATURES ===
    # Third downs
    third_down_plays = team_pbp[team_pbp['down'] == 3]
    features['downs_thirdDownAttempts'] = len(third_down_plays)
    features['downs_thirdDownConversions'] = third_down_plays['first_down'].sum()

    if features['downs_thirdDownAttempts'] > 0:
        features['downs_thirdDownConversionPercentage'] = (features['downs_thirdDownConversions'] / features['downs_thirdDownAttempts']) * 100
    else:
        features['downs_thirdDownConversionPercentage'] = 0

    # Fourth downs
    fourth_down_plays = team_pbp[team_pbp['down'] == 4]
    features['downs_fourthDownAttempts'] = len(fourth_down_plays)
    features['downs_fourthDownConversions'] = fourth_down_plays['first_down'].sum()

    if features['downs_fourthDownAttempts'] > 0:
        features['downs_fourthDownConversionPercentage'] = (features['downs_fourthDownConversions'] / features['downs_fourthDownAttempts']) * 100
    else:
        features['downs_fourthDownConversionPercentage'] = 0

    # Total first downs
    features['downs_firstDowns'] = team_pbp['first_down'].sum()

    if len(team_pbp) > 0:
        features['downs_firstDownPercentage'] = (features['downs_firstDowns'] / len(team_pbp)) * 100
    else:
        features['downs_firstDownPercentage'] = 0

    # === TEAM RECORDS ===
    # Get team's schedule (use nfl-data-py abbreviation)
    team_schedule = schedules[(schedules['home_team'] == nfl_team) | (schedules['away_team'] == nfl_team)].copy()
    team_schedule = team_schedule[team_schedule['week'] <= 18]  # Regular season only

    # Determine wins/losses
    team_schedule['is_home'] = team_schedule['home_team'] == nfl_team
    team_schedule['team_score'] = team_schedule.apply(lambda x: x['home_score'] if x['is_home'] else x['away_score'], axis=1)
    team_schedule['opp_score'] = team_schedule.apply(lambda x: x['away_score'] if x['is_home'] else x['home_score'], axis=1)
    team_schedule['win'] = team_schedule['team_score'] > team_schedule['opp_score']
    team_schedule['loss'] = team_schedule['team_score'] < team_schedule['opp_score']

    features['total_wins'] = team_schedule['win'].sum()
    features['total_losses'] = team_schedule['loss'].sum()
    total_games = features['total_wins'] + features['total_losses']
    features['total_winPercentage'] = (features['total_wins'] / total_games * 100) if total_games > 0 else 0

    features['total_pointsFor'] = team_schedule['team_score'].sum()
    features['total_pointsAgainst'] = team_schedule['opp_score'].sum()
    features['total_pointDifferential'] = features['total_pointsFor'] - features['total_pointsAgainst']

    # Home/Away records
    home_games = team_schedule[team_schedule['is_home']]
    away_games = team_schedule[~team_schedule['is_home']]

    features['home_wins'] = home_games['win'].sum()
    features['home_losses'] = home_games['loss'].sum()
    features['away_wins'] = away_games['win'].sum()
    features['away_losses'] = away_games['loss'].sum()

    # Points per game
    if total_games > 0:
        features['general_pointsPerGame'] = features['total_pointsFor'] / total_games
        features['scoring_pointsPerGame'] = features['total_pointsFor'] / total_games
        features['scoring_pointsAgainstPerGame'] = features['total_pointsAgainst'] / total_games
    else:
        features['general_pointsPerGame'] = 0
        features['scoring_pointsPerGame'] = 0
        features['scoring_pointsAgainstPerGame'] = 0

    # Time of possession (approximate from play count)
    total_plays = len(team_pbp)
    # Assume average play takes 6 seconds, game has 3600 seconds
    # Team's share = their plays / total plays in their games
    features['miscellaneous_timeOfPossessionSeconds'] = total_plays * 6  # Rough estimate
    if total_games > 0:
        features['miscellaneous_timeOfPossessionSecondsPerGame'] = features['miscellaneous_timeOfPossessionSeconds'] / total_games
    else:
        features['miscellaneous_timeOfPossessionSecondsPerGame'] = 0

    return features

# Derive for all teams
teams = sorted(espn.index.unique())
derived_list = []

for i, team in enumerate(teams, 1):
    print(f"  [{i}/{len(teams)}] {team}...", end=' ')
    features = derive_all_features(team, pbp_reg, schedules)
    derived_list.append(features)
    print("✅")

derived = pd.DataFrame(derived_list).set_index('team')
print(f"\n  ✅ Derived {len(derived)} teams, {len(derived.columns)} features")

print("\n" + "=" * 120)
print("VALIDATION RESULTS")
print("=" * 120)

# Combine all features to test
all_features_to_test = exact_features + partial_features

# Calculate correlations
results = []
common_teams = sorted(set(espn.index) & set(derived.index))

print(f"\nValidating {len(all_features_to_test)} features across {len(common_teams)} teams...\n")

for feature in all_features_to_test:
    # Check if feature exists in both datasets
    if feature not in espn.columns:
        print(f"⚠️  {feature:<50} NOT IN ESPN DATA")
        continue

    if feature not in derived.columns:
        print(f"⚠️  {feature:<50} NOT IN DERIVED DATA")
        continue

    # Get values
    espn_vals = espn.loc[common_teams, feature].values
    derived_vals = derived.loc[common_teams, feature].values

    # Remove NaN values
    mask = ~(np.isnan(espn_vals) | np.isnan(derived_vals))
    if mask.sum() < 3:
        print(f"⚠️  {feature:<50} INSUFFICIENT DATA (n={mask.sum()})")
        continue

    espn_clean = espn_vals[mask]
    derived_clean = derived_vals[mask]

    # Calculate correlation
    if len(espn_clean) >= 3 and espn_clean.std() > 0 and derived_clean.std() > 0:
        r, p = pearsonr(espn_clean, derived_clean)

        # Determine category
        if feature in exact_features:
            category = "EXACT MATCH"
            threshold = 0.95
        else:
            category = "PARTIAL MATCH"
            threshold = 0.85

        # Status
        if r >= threshold:
            status = "✅ PASS"
        elif r >= 0.70:
            status = "⚠️  MEDIUM"
        else:
            status = "❌ FAIL"

        # Calculate mean absolute error
        mae = np.mean(np.abs(espn_clean - derived_clean))
        mean_espn = np.mean(espn_clean)
        mape = (mae / mean_espn * 100) if mean_espn != 0 else 0

        results.append({
            'feature': feature,
            'category': category,
            'r': r,
            'p': p,
            'n': mask.sum(),
            'mae': mae,
            'mape': mape,
            'status': status,
            'threshold': threshold
        })

        print(f"{status} {feature:<50} r={r:6.4f} (p={p:.2e}, n={mask.sum()}, MAPE={mape:5.1f}%) [{category}]")
    else:
        print(f"⚠️  {feature:<50} NO VARIANCE")

# Create results dataframe
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv('results/comprehensive_validation_results.csv', index=False)
print(f"\n✅ Saved detailed results to: results/comprehensive_validation_results.csv")

# Summary statistics
print("\n" + "=" * 120)
print("SUMMARY STATISTICS")
print("=" * 120)

if len(results_df) > 0:
    # Overall
    print(f"\nTotal features validated: {len(results_df)}")
    print(f"  EXACT MATCH: {len(results_df[results_df['category']=='EXACT MATCH'])}")
    print(f"  PARTIAL MATCH: {len(results_df[results_df['category']=='PARTIAL MATCH'])}")

    # By status
    print(f"\nBy Status:")
    print(f"  ✅ PASS (r >= threshold): {len(results_df[results_df['status']=='✅ PASS'])} ({len(results_df[results_df['status']=='✅ PASS'])/len(results_df)*100:.1f}%)")
    print(f"  ⚠️  MEDIUM (0.70 <= r < threshold): {len(results_df[results_df['status']=='⚠️  MEDIUM'])} ({len(results_df[results_df['status']=='⚠️  MEDIUM'])/len(results_df)*100:.1f}%)")
    print(f"  ❌ FAIL (r < 0.70): {len(results_df[results_df['status']=='❌ FAIL'])} ({len(results_df[results_df['status']=='❌ FAIL'])/len(results_df)*100:.1f}%)")

    # By category
    print(f"\nEXACT MATCH Features (threshold r > 0.95):")
    exact_df = results_df[results_df['category'] == 'EXACT MATCH']
    if len(exact_df) > 0:
        print(f"  Mean r: {exact_df['r'].mean():.4f}")
        print(f"  Median r: {exact_df['r'].median():.4f}")
        print(f"  Pass rate: {len(exact_df[exact_df['r'] >= 0.95])/len(exact_df)*100:.1f}%")
        print(f"  Mean MAPE: {exact_df['mape'].mean():.1f}%")

    print(f"\nPARTIAL MATCH Features (threshold r > 0.85):")
    partial_df = results_df[results_df['category'] == 'PARTIAL MATCH']
    if len(partial_df) > 0:
        print(f"  Mean r: {partial_df['r'].mean():.4f}")
        print(f"  Median r: {partial_df['r'].median():.4f}")
        print(f"  Pass rate: {len(partial_df[partial_df['r'] >= 0.85])/len(partial_df)*100:.1f}%")
        print(f"  Mean MAPE: {partial_df['mape'].mean():.1f}%")

    # Top 10 best correlations
    print(f"\n🏆 TOP 10 BEST CORRELATIONS:")
    top10 = results_df.nlargest(10, 'r')
    for idx, row in top10.iterrows():
        print(f"  {row['feature']:<50} r={row['r']:.4f} (MAPE={row['mape']:.1f}%) [{row['category']}]")

    # Bottom 10 worst correlations
    print(f"\n💀 BOTTOM 10 WORST CORRELATIONS:")
    bottom10 = results_df.nsmallest(10, 'r')
    for idx, row in bottom10.iterrows():
        print(f"  {row['feature']:<50} r={row['r']:.4f} (MAPE={row['mape']:.1f}%) [{row['category']}]")

print("\n" + "=" * 120)
print("✅ COMPREHENSIVE VALIDATION COMPLETE!")
print("=" * 120)

