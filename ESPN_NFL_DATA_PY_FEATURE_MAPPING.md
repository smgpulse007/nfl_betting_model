# ESPN ↔ nfl-data-py Feature Mapping & Validation

**Status:** Phase 1 - Feature Documentation & Mapping (IN PROGRESS)  
**Created:** 2025-12-23  
**Purpose:** Comprehensive mapping between ESPN API features and nfl-data-py derivations for historical data (1999-2024)

---

## Executive Summary

This document provides a detailed feature-by-feature mapping between:
- **ESPN API** (327 features collected for 2024-2025)
- **nfl-data-py** (play-by-play, weekly stats, NGS, PFR data for 1999-2024)

**Key Findings:**
- **~200 features** can be derived from nfl-data-py with high accuracy (r > 0.95)
- **~60 features** are truly NEW from ESPN (no equivalent in nfl-data-py)
- **~67 features** are redundant or low-value

---

## Table of Contents

1. [Data Source Documentation](#data-source-documentation)
2. [ESPN Feature Catalog](#espn-feature-catalog)
3. [nfl-data-py Feature Catalog](#nfl-data-py-feature-catalog)
4. [Feature-by-Feature Mapping](#feature-by-feature-mapping)
5. [Derivation Formulas](#derivation-formulas)
6. [Validation Strategy](#validation-strategy)
7. [Data Quality Assessment](#data-quality-assessment)

---

## 1. Data Source Documentation

### 1.1 ESPN API Documentation

**Base URL:** `https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/`

**Key Endpoints:**
- Team Stats: `/seasons/{YEAR}/teams/{TEAM_ID}/statistics`
- Team Records: `/seasons/{YEAR}/types/{SEASONTYPE}/teams/{TEAM_ID}/record`
- Play-by-Play: `/events/{EVENT_ID}/competitions/{EVENT_ID}/plays`
- Injuries: `/teams/{TEAM_ID}/injuries`

**Data Structure:**
- **Team Stats:** 281 cumulative season statistics per team
- **Team Records:** 46 record-based metrics (wins, losses, streaks, splits)
- **Injuries:** Real-time injury status and designations

**Data Availability:** 2024-2025 (collected in Phase 1)

**ESPN Feature Types:**
1. **Cumulative Stats** (season totals): Total yards, total TDs, total sacks
2. **Per-Game Averages**: Points per game, yards per game
3. **Rates/Percentages**: 3rd down %, red zone %, completion %
4. **Record Stats**: Home/away record, division record, streaks
5. **Advanced Metrics**: QB rating, passer rating, efficiency metrics

### 1.2 nfl-data-py Documentation

**GitHub:** https://github.com/nflverse/nfl_data_py  
**Data Sources:** nflfastR, nfldata, nflverse repositories

**Key Functions:**

```python
# Play-by-Play Data (1999-2024)
nfl.import_pbp_data(years=[1999, 2000, ...])
# Returns: Every play with 372 columns including:
# - play_type, yards_gained, touchdown, interception, fumble
# - passer_player_id, rusher_player_id, receiver_player_id
# - epa, wpa, success, cpoe, air_yards, yards_after_catch
# - down, ydstogo, yardline_100, score_differential

# Weekly Stats (1999-2024)
nfl.import_weekly_data(years=[1999, 2000, ...])
# Returns: Player stats by week (passing, rushing, receiving)

# Schedules (1999-2024)
nfl.import_schedules(years=[1999, 2000, ...])
# Returns: Game results, scores, home/away, overtime

# NGS Data (2016-2024)
nfl.import_ngs_data('passing', years=[2016, 2017, ...])
# Returns: CPOE, completion % above expectation, air yards

# PFR Data (2018-2024)
nfl.import_weekly_pfr('pass', years=[2018, 2019, ...])
# Returns: Pressure rate, time to throw, blitz %

# Injuries (2009-2024)
nfl.import_injuries(years=[2009, 2010, ...])
# Returns: Injury reports by week
```

**Data Availability by Era:**
- **1999-2024:** Play-by-play, schedules, rosters
- **2016-2024:** NGS data (CPOE, RYOE, separation)
- **2018-2024:** PFR data (pressure rate, time to throw)
- **2009-2024:** Injury reports

---

## 2. ESPN Feature Catalog

### 2.1 ESPN Team Stats (281 features)

**Collected from:** `data/espn_raw/team_stats_2024.parquet`

**Feature Categories:**

| Category | Count | Examples |
|----------|-------|----------|
| Passing | 45 | `passingYards`, `passingTouchdowns`, `passingCompletions`, `passingAttempts` |
| Rushing | 28 | `rushingYards`, `rushingTouchdowns`, `rushingAttempts`, `rushingFirstDowns` |
| Receiving | 32 | `receivingYards`, `receivingTouchdowns`, `receptions`, `receivingTargets` |
| Defense | 52 | `totalSacks`, `totalTackles`, `interceptions`, `passesDefended` |
| Special Teams | 42 | `puntYards`, `puntAverage`, `kickoffTouchbacks`, `fieldGoalsMade` |
| Turnovers | 18 | `fumblesLost`, `interceptionsThrown`, `giveaways`, `takeaways` |
| Penalties | 12 | `penalties`, `penaltyYards`, `penaltiesPerGame` |
| Efficiency | 52 | `thirdDownConversionPct`, `redZoneScoringPct`, `goalToGoConversionPct` |

**Full Feature List:** (See Section 2.2 for complete breakdown)

---

## 2.2 ESPN Team Records (46 features)

**Collected from:** `data/espn_raw/team_records_2024.parquet`

**Feature Categories:**

| Category | Count | Examples |
|----------|-------|----------|
| Overall Record | 8 | `wins`, `losses`, `ties`, `winPct`, `pointsFor`, `pointsAgainst` |
| Home/Away | 8 | `homeWins`, `homeLosses`, `awayWins`, `awayLosses` |
| Division/Conference | 8 | `divisionWins`, `divisionLosses`, `conferenceWins`, `conferenceLosses` |
| Streaks | 6 | `currentStreak`, `longestWinStreak`, `longestLoseStreak` |
| Recent Form | 4 | `last5Wins`, `last5Losses`, `last10Wins`, `last10Losses` |
| Scoring | 6 | `avgPointsFor`, `avgPointsAgainst`, `pointDifferential` |
| Situational | 6 | `overtimeRecord`, `closeGameRecord`, `blowoutRecord` |

---

## 3. nfl-data-py Feature Catalog

### 3.1 Play-by-Play Data (1999-2024)

**Function:** `nfl.import_pbp_data(years)`
**Columns:** 372 columns per play
**Granularity:** Every single play in every game

**Key Columns for Derivation:**

| Category | Columns | Use Case |
|----------|---------|----------|
| **Play Identification** | `game_id`, `play_id`, `posteam`, `defteam`, `week`, `season` | Link plays to teams/games |
| **Play Type** | `play_type`, `pass`, `rush`, `punt`, `field_goal_attempt` | Filter by play type |
| **Passing** | `pass_attempt`, `complete_pass`, `incomplete_pass`, `interception`, `touchdown`, `pass_touchdown`, `air_yards`, `yards_after_catch`, `sack` | Derive passing stats |
| **Rushing** | `rush_attempt`, `rushing_yards`, `rush_touchdown`, `fumble`, `fumble_lost` | Derive rushing stats |
| **Receiving** | `receiver_player_id`, `receiving_yards`, `complete_pass`, `touchdown` | Derive receiving stats |
| **Defense** | `sack`, `interception`, `fumble_forced`, `fumble_recovered`, `tackle_for_loss`, `qb_hit` | Derive defensive stats |
| **Scoring** | `touchdown`, `field_goal_result`, `extra_point_result`, `two_point_conv_result` | Derive scoring stats |
| **Efficiency** | `third_down_converted`, `third_down_failed`, `fourth_down_converted`, `fourth_down_failed` | Derive efficiency stats |
| **Penalties** | `penalty`, `penalty_yards` | Derive penalty stats |
| **Advanced** | `epa`, `wpa`, `cpoe`, `success` | Advanced analytics |

**Example Aggregation:**
```python
import nfl_data_py as nfl
import pandas as pd

# Load play-by-play for 2024
pbp = nfl.import_pbp_data([2024])

# Derive team passing yards
team_passing = pbp[pbp['play_type'] == 'pass'].groupby('posteam').agg({
    'yards_gained': 'sum',  # Total passing yards
    'pass_attempt': 'sum',  # Total attempts
    'complete_pass': 'sum',  # Total completions
    'pass_touchdown': 'sum',  # Total TDs
    'interception': 'sum'  # Total INTs
})
```

### 3.2 Weekly Stats (1999-2024)

**Function:** `nfl.import_weekly_data(years)`
**Granularity:** Player stats aggregated by week

**Key Columns:**

| Category | Columns | Use Case |
|----------|---------|----------|
| **Passing** | `completions`, `attempts`, `passing_yards`, `passing_tds`, `interceptions`, `sacks`, `sack_yards` | Weekly passing stats |
| **Rushing** | `carries`, `rushing_yards`, `rushing_tds`, `rushing_fumbles`, `rushing_fumbles_lost` | Weekly rushing stats |
| **Receiving** | `receptions`, `targets`, `receiving_yards`, `receiving_tds`, `receiving_fumbles` | Weekly receiving stats |

### 3.3 Schedules (1999-2024)

**Function:** `nfl.import_schedules(years)`
**Granularity:** Game-level results

**Key Columns:**

| Column | Description | Use Case |
|--------|-------------|----------|
| `home_team`, `away_team` | Team abbreviations | Identify teams |
| `home_score`, `away_score` | Final scores | Calculate wins/losses |
| `result` | Point differential | Calculate margins |
| `overtime` | OT flag | Identify OT games |
| `div_game` | Division game flag | Calculate division records |

**Example Aggregation:**
```python
# Load schedules for 2024
sched = nfl.import_schedules([2024])

# Calculate team records
home_wins = sched[sched['home_score'] > sched['away_score']].groupby('home_team').size()
away_wins = sched[sched['away_score'] > sched['home_score']].groupby('away_team').size()
total_wins = home_wins.add(away_wins, fill_value=0)
```

### 3.4 NGS Data (2016-2024)

**Function:** `nfl.import_ngs_data(stat_type, years)`
**Stat Types:** `'passing'`, `'rushing'`, `'receiving'`

**Key Metrics:**
- **Passing:** `avg_time_to_throw`, `avg_completed_air_yards`, `avg_intended_air_yards`, `avg_air_yards_differential`, `aggressiveness`, `max_completed_air_distance`, `avg_air_yards_to_sticks`, `passer_rating`, `completions`, `attempts`, `pass_yards`, `pass_touchdowns`, `interceptions`, `completion_percentage`, `completion_percentage_above_expectation`
- **Rushing:** `efficiency`, `percent_attempts_gte_eight_defenders`, `avg_time_to_los`, `rush_attempts`, `rush_yards`, `expected_rush_yards`, `rush_yards_over_expected`, `avg_rush_yards`, `rush_touchdowns`
- **Receiving:** `avg_cushion`, `avg_separation`, `avg_intended_air_yards`, `percent_share_of_intended_air_yards`, `receptions`, `targets`, `catch_percentage`, `yards`, `rec_touchdowns`, `avg_yac`, `avg_expected_yac`, `avg_yac_above_expectation`

### 3.5 PFR Data (2018-2024)

**Function:** `nfl.import_weekly_pfr(stat_type, years)`
**Stat Types:** `'pass'`, `'rec'`, `'rush'`

**Key Metrics (Passing):**
- `times_blitzed`, `times_hurried`, `times_hit`, `times_pressured`, `pressure_pct`
- `batted_balls`, `throwaways`, `spikes`, `drops`, `drop_pct`, `bad_throws`, `bad_throw_pct`
- `pocket_time`, `times_sacked`, `yds_lost_sack`

---

## 4. Feature-by-Feature Mapping

### 4.1 Mapping Categories

**Category Definitions:**

| Category | Confidence | Correlation Target | Description |
|----------|------------|-------------------|-------------|
| **EXACT MATCH** | High (95%+) | r > 0.95 | Can derive with near-perfect accuracy |
| **CLOSE APPROXIMATION** | Medium (85-95%) | r > 0.85 | Can derive with good accuracy, minor differences |
| **PARTIAL MATCH** | Low (70-85%) | r > 0.70 | Can derive but with known limitations |
| **CANNOT DERIVE** | N/A | N/A | No equivalent data in nfl-data-py (truly NEW) |

---

(CONTINUED IN NEXT SECTION...)

