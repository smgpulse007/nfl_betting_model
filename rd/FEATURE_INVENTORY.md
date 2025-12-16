# NFL Data-Py Feature Inventory

## Executive Summary

**Total Data Sources:** 18+  
**Total Unique Columns:** ~805  
**Data Available From:** 1999-present (varies by source)

This document catalogs ALL available features from nfl-data-py/nflverse that could be used for NFL betting prediction models.

---

## 1. PLAY-BY-PLAY DATA (Core) - 397 Columns

**Source:** `nfl.import_pbp_data([years])`  
**Availability:** 1999-present  
**Granularity:** Per-play  
**2024 Sample:** 48,000+ plays

### 1.1 Expected Points Added (EPA) Features
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `epa` | Expected Points Added per play | ⭐⭐⭐⭐⭐ Core metric |
| `total_home_epa` | Cumulative home EPA in game | ⭐⭐⭐⭐ |
| `total_away_epa` | Cumulative away EPA in game | ⭐⭐⭐⭐ |
| `total_home_pass_epa` | Home passing EPA | ⭐⭐⭐⭐ |
| `total_home_rush_epa` | Home rushing EPA | ⭐⭐⭐⭐ |
| `air_epa` | EPA from air yards | ⭐⭐⭐ |
| `yac_epa` | EPA from yards after catch | ⭐⭐⭐ |
| `qb_epa` | QB-specific EPA (fumble-adjusted) | ⭐⭐⭐⭐⭐ |

### 1.2 Win Probability (WP) Features
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `wp` | Win probability for possession team | ⭐⭐⭐⭐ |
| `wpa` | Win Probability Added | ⭐⭐⭐⭐ |
| `vegas_wp` | Vegas-adjusted win probability | ⭐⭐⭐⭐⭐ |
| `vegas_wpa` | Vegas-adjusted WPA | ⭐⭐⭐⭐⭐ |
| `home_wp` | Home team win probability | ⭐⭐⭐⭐ |

### 1.3 Completion Probability Features
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `cp` | Completion probability | ⭐⭐⭐ |
| `cpoe` | Completion % Over Expected | ⭐⭐⭐⭐⭐ Key QB metric |

### 1.4 Expected YAC Features
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `xyac_epa` | Expected EPA from YAC | ⭐⭐⭐ |
| `xyac_mean_yardage` | Expected YAC yards | ⭐⭐⭐ |
| `xyac_success` | Probability of positive EPA | ⭐⭐⭐ |
| `xyac_fd` | Probability of first down | ⭐⭐⭐ |

### 1.5 Pass/Rush Tendency Features
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `xpass` | Pass probability (0-1) | ⭐⭐⭐⭐ |
| `pass_oe` | Pass % Over Expected | ⭐⭐⭐⭐ |
| `shotgun` | Shotgun formation | ⭐⭐ |
| `no_huddle` | No-huddle offense | ⭐⭐ |

### 1.6 Participation/Personnel Data (NEW in PBP!)
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `offense_formation` | Formation (SHOTGUN, SINGLEBACK, etc.) | ⭐⭐⭐ |
| `offense_personnel` | Personnel grouping (11, 12, 21, etc.) | ⭐⭐⭐⭐ |
| `defenders_in_box` | Number of defenders in box | ⭐⭐⭐⭐ |
| `defense_personnel` | Defensive personnel | ⭐⭐⭐ |
| `number_of_pass_rushers` | Pass rushers on play | ⭐⭐⭐⭐ |
| `time_to_throw` | Seconds from snap to throw | ⭐⭐⭐⭐⭐ |
| `was_pressure` | QB was pressured | ⭐⭐⭐⭐⭐ |
| `route` | Primary receiver route | ⭐⭐⭐ |
| `defense_man_zone_type` | Man vs Zone coverage | ⭐⭐⭐⭐⭐ |
| `defense_coverage_type` | Cover 0/1/2/3/4/6 | ⭐⭐⭐⭐⭐ |

### 1.7 Game Context Features
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `spread_line` | Vegas spread | ⭐⭐⭐⭐⭐ |
| `total_line` | Vegas over/under | ⭐⭐⭐⭐⭐ |
| `result` | Final score differential | Target variable |
| `total` | Total points scored | Target variable |
| `roof` | dome/outdoors/closed/open | ⭐⭐⭐ |
| `surface` | grass/turf | ⭐⭐⭐ |
| `temp` | Temperature | ⭐⭐⭐ |
| `wind` | Wind speed | ⭐⭐⭐ |

---

## 2. SCHEDULE DATA - 46 Columns

**Source:** `nfl.import_schedules([years])`  
**Availability:** 1999-present  
**Granularity:** Per-game

### Key Features
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `away_rest` | Days rest for away team | ⭐⭐⭐⭐⭐ |
| `home_rest` | Days rest for home team | ⭐⭐⭐⭐⭐ |
| `away_moneyline` | Away team moneyline odds | ⭐⭐⭐⭐⭐ |
| `home_moneyline` | Home team moneyline odds | ⭐⭐⭐⭐⭐ |
| `spread_line` | Closing spread | ⭐⭐⭐⭐⭐ |
| `away_spread_odds` | Away spread juice | ⭐⭐⭐⭐ |
| `home_spread_odds` | Home spread juice | ⭐⭐⭐⭐ |
| `total_line` | Closing total | ⭐⭐⭐⭐⭐ |
| `under_odds` | Under juice | ⭐⭐⭐⭐ |
| `over_odds` | Over juice | ⭐⭐⭐⭐ |
| `away_qb_id` / `home_qb_id` | Starting QB IDs | ⭐⭐⭐⭐⭐ |
| `referee` | Head referee | ⭐⭐⭐ |
| `div_game` | Divisional game flag | ⭐⭐⭐ |

---

## 3. NEXT GEN STATS - 74 Columns Total

**Source:** `nfl.import_ngs_data(stat_type, [years])`  
**Availability:** 2016-present  
**Granularity:** Per-player per-week

### 3.1 NGS Passing (29 columns)
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `avg_time_to_throw` | Average time to throw | ⭐⭐⭐⭐⭐ |
| `avg_completed_air_yards` | Avg completed air yards | ⭐⭐⭐⭐ |
| `avg_intended_air_yards` | Avg intended air yards | ⭐⭐⭐⭐ |
| `aggressiveness` | % throws into tight coverage | ⭐⭐⭐⭐⭐ |
| `expected_completion_percentage` | xComp% | ⭐⭐⭐⭐⭐ |
| `completion_percentage_above_expectation` | CPOE | ⭐⭐⭐⭐⭐ |
| `avg_air_yards_to_sticks` | Air yards relative to 1st down | ⭐⭐⭐⭐ |

### 3.2 NGS Rushing (22 columns)
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `efficiency` | Rushing efficiency (yards traveled/yards gained) | ⭐⭐⭐⭐ |
| `percent_attempts_gte_eight_defenders` | % runs vs stacked box | ⭐⭐⭐⭐⭐ |
| `avg_time_to_los` | Time to line of scrimmage | ⭐⭐⭐⭐ |
| `expected_rush_yards` | xRush yards | ⭐⭐⭐⭐⭐ |
| `rush_yards_over_expected` | RYOE | ⭐⭐⭐⭐⭐ |
| `rush_yards_over_expected_per_att` | RYOE/att | ⭐⭐⭐⭐⭐ |

### 3.3 NGS Receiving (23 columns)
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `avg_cushion` | Avg distance from defender at snap | ⭐⭐⭐⭐ |
| `avg_separation` | Avg separation at catch | ⭐⭐⭐⭐⭐ |
| `avg_expected_yac` | Expected YAC | ⭐⭐⭐⭐ |
| `avg_yac_above_expectation` | YAC over expected | ⭐⭐⭐⭐⭐ |
| `percent_share_of_intended_air_yards` | Air yards market share | ⭐⭐⭐⭐ |

---

## 4. WEEKLY PLAYER STATS - 53 Columns

**Source:** `nfl.import_weekly_data([years])`  
**Availability:** 1999-present  
**Granularity:** Per-player per-week

### Key Features
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `passing_epa` | Weekly passing EPA | ⭐⭐⭐⭐⭐ |
| `rushing_epa` | Weekly rushing EPA | ⭐⭐⭐⭐⭐ |
| `receiving_epa` | Weekly receiving EPA | ⭐⭐⭐⭐⭐ |
| `dakota` | DAKOTA (EPA + CPOE composite) | ⭐⭐⭐⭐⭐ |
| `pacr` | Passing Air Conversion Ratio | ⭐⭐⭐⭐ |
| `racr` | Receiving Air Conversion Ratio | ⭐⭐⭐⭐ |
| `wopr` | Weighted Opportunity Rating | ⭐⭐⭐⭐ |
| `target_share` | Target market share | ⭐⭐⭐⭐ |
| `air_yards_share` | Air yards market share | ⭐⭐⭐⭐ |

---

## 5. INJURY REPORTS - 16 Columns

**Source:** `nfl.import_injuries([years])`  
**Availability:** 2009-present  
**Granularity:** Per-player per-week  
**2024 Sample:** 6,215 records

### Key Features
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `report_status` | Game status (Out/Doubtful/Questionable) | ⭐⭐⭐⭐⭐ |
| `practice_status` | Practice participation (DNP/LP/FP) | ⭐⭐⭐⭐⭐ |
| `report_primary_injury` | Primary injury type | ⭐⭐⭐⭐ |
| `position` | Player position | ⭐⭐⭐⭐⭐ |

---

## 6. SNAP COUNTS - 16 Columns

**Source:** `nfl.import_snap_counts([years])`
**Availability:** 2012-present
**Granularity:** Per-player per-game
**2024 Sample:** 26,615 records

### Key Features
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `offense_snaps` | Offensive snap count | ⭐⭐⭐⭐⭐ |
| `offense_pct` | % of offensive snaps | ⭐⭐⭐⭐⭐ |
| `defense_snaps` | Defensive snap count | ⭐⭐⭐⭐ |
| `defense_pct` | % of defensive snaps | ⭐⭐⭐⭐ |

---

## 7. DEPTH CHARTS - 15 Columns

**Source:** `nfl.import_depth_charts([years])`
**Availability:** 2001-present
**Granularity:** Per-player per-week
**2024 Sample:** 37,312 records

### Key Features
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `depth_team` | Depth chart position (1/2/3) | ⭐⭐⭐⭐⭐ |
| `position` | Position | ⭐⭐⭐⭐ |
| `formation` | Offensive/Defensive formation | ⭐⭐⭐ |

---

## 8. PRO-FOOTBALL-REFERENCE STATS - 57 Columns

**Source:** `nfl.import_weekly_pfr(stat_type, [years])`
**Availability:** 2018-present
**Granularity:** Per-player per-game

### 8.1 PFR Passing (24 columns)
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `passing_drops` | Drops by receivers | ⭐⭐⭐⭐ |
| `passing_drop_pct` | Drop percentage | ⭐⭐⭐⭐ |
| `passing_bad_throws` | Bad throws by QB | ⭐⭐⭐⭐⭐ |
| `passing_bad_throw_pct` | Bad throw % | ⭐⭐⭐⭐⭐ |
| `times_blitzed` | Times QB was blitzed | ⭐⭐⭐⭐⭐ |
| `times_hurried` | Times QB was hurried | ⭐⭐⭐⭐⭐ |
| `times_pressured` | Total pressures | ⭐⭐⭐⭐⭐ |
| `times_pressured_pct` | Pressure rate | ⭐⭐⭐⭐⭐ |

### 8.2 PFR Rushing (16 columns)
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `rushing_yards_before_contact` | YBC | ⭐⭐⭐⭐ |
| `rushing_yards_before_contact_avg` | YBC/att | ⭐⭐⭐⭐ |
| `rushing_yards_after_contact` | YAC | ⭐⭐⭐⭐⭐ |
| `rushing_yards_after_contact_avg` | YAC/att | ⭐⭐⭐⭐⭐ |
| `rushing_broken_tackles` | Broken tackles | ⭐⭐⭐⭐ |

### 8.3 PFR Receiving (17 columns)
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `receiving_drop` | Drops | ⭐⭐⭐⭐ |
| `receiving_drop_pct` | Drop % | ⭐⭐⭐⭐ |
| `receiving_broken_tackles` | Broken tackles | ⭐⭐⭐⭐ |
| `receiving_rat` | Receiver rating | ⭐⭐⭐⭐ |

---

## 9. FTN CHARTING DATA - 29 Columns

**Source:** `nfl.import_ftn_data([years])`
**Availability:** 2022-present
**Granularity:** Per-play
**2024 Sample:** 48,031 plays

### Key Features (Manually Charted!)
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `is_play_action` | Play action pass | ⭐⭐⭐⭐ |
| `is_screen_pass` | Screen pass | ⭐⭐⭐ |
| `is_rpo` | Run-pass option | ⭐⭐⭐⭐ |
| `is_trick_play` | Trick play | ⭐⭐⭐ |
| `is_qb_out_of_pocket` | QB left pocket | ⭐⭐⭐⭐ |
| `is_interception_worthy` | INT-worthy throw | ⭐⭐⭐⭐⭐ |
| `is_throw_away` | Throw away | ⭐⭐⭐ |
| `is_catchable_ball` | Catchable ball | ⭐⭐⭐⭐ |
| `is_contested_ball` | Contested catch | ⭐⭐⭐⭐ |
| `is_drop` | Receiver drop | ⭐⭐⭐⭐⭐ |
| `n_blitzers` | Number of blitzers | ⭐⭐⭐⭐⭐ |
| `n_pass_rushers` | Number of pass rushers | ⭐⭐⭐⭐⭐ |
| `is_qb_fault_sack` | QB-caused sack | ⭐⭐⭐⭐⭐ |
| `read_thrown` | Which read QB threw to | ⭐⭐⭐⭐ |
| `qb_location` | Under center/Shotgun/Pistol | ⭐⭐⭐ |

---

## 10. OFFICIALS DATA - 5 Columns

**Source:** `nfl.import_officials([years])`
**Availability:** 2015-present
**Granularity:** Per-game

### Key Features
| Column | Description | Betting Value |
|--------|-------------|---------------|
| `name` | Official name | ⭐⭐⭐ |
| `off_pos` | Official position (R, U, HL, etc.) | ⭐⭐⭐ |

---

# HIGH-VALUE FEATURE CATEGORIES FOR BETTING

## 🏆 TIER S: Highest Predictive Value

1. **EPA Metrics** (from PBP)
   - `epa`, `qb_epa`, `passing_epa`, `rushing_epa`
   - Rolling averages are key for prediction

2. **Completion Probability Over Expected (CPOE)**
   - Best single QB metric
   - Available in PBP and NGS

3. **Pressure Metrics** (from PFR + Participation)
   - `times_pressured_pct`, `was_pressure`
   - O-line vs D-line matchup predictor

4. **Rest Days** (from Schedule)
   - `home_rest`, `away_rest`
   - Short week = significant disadvantage

5. **Injury Status** (from Injuries)
   - QB injuries = 3-7 point swing
   - Key player availability

## 🥇 TIER A: High Value

1. **Next Gen Stats**
   - `avg_separation`, `avg_time_to_throw`
   - `rush_yards_over_expected`

2. **Coverage Type** (from Participation)
   - `defense_man_zone_type`, `defense_coverage_type`
   - Matchup-specific insights

3. **Vegas Lines** (from Schedule)
   - `spread_line`, `total_line`, moneylines
   - Market efficiency baseline

4. **Snap Counts**
   - Player workload trends
   - Injury/fatigue indicators

## 🥈 TIER B: Moderate Value

1. **FTN Charting**
   - `is_interception_worthy`, `is_drop`
   - Luck-adjusted metrics

2. **PFR Advanced**
   - `rushing_yards_after_contact`
   - `passing_bad_throw_pct`

3. **Weather/Surface**
   - `temp`, `wind`, `surface`, `roof`
   - Totals impact

## 🥉 TIER C: Situational Value

1. **Officials**
   - Penalty tendencies by crew
   - Small but measurable effect

2. **Depth Charts**
   - Backup identification
   - Injury replacement quality

---

# COMPLEX MODEL OPPORTUNITIES

## 1. Deep Learning Potential

With 805+ features, deep learning becomes viable:

- **Transformer Models**: Sequence of plays → game outcome
- **Graph Neural Networks**: Player relationships, matchups
- **LSTM/GRU**: Time series of team performance

## 2. Feature Interactions

Complex models can capture:
- QB pressure rate × O-line injuries
- Receiver separation × CB coverage type
- Rush RYOE × Defenders in box

## 3. Ensemble Stacking

- Level 1: Specialized models (EPA model, Injury model, Weather model)
- Level 2: Meta-learner combines predictions
- Level 3: Betting strategy optimizer

## 4. Real-Time Features

Participation data enables:
- Pre-snap formation analysis
- Personnel grouping matchups
- Coverage tendency prediction

---

# DATA AVAILABILITY MATRIX

| Data Source | Years Available | Update Frequency |
|-------------|-----------------|------------------|
| Play-by-Play | 1999-present | Real-time |
| Schedules | 1999-present | Weekly |
| NGS | 2016-present | Weekly |
| Weekly Stats | 1999-present | Weekly |
| Injuries | 2009-present | Daily |
| Snap Counts | 2012-present | Weekly |
| Depth Charts | 2001-present | Weekly |
| PFR Stats | 2018-present | Weekly |
| FTN Charting | 2022-present | Weekly |
| Officials | 2015-present | Weekly |

---

# RECOMMENDED NEXT STEPS

1. **Immediate (TIER 2)**: Implement injury impact scoring
2. **Short-term**: Add NGS metrics (CPOE, separation, RYOE)
3. **Medium-term**: Build pressure/coverage matchup model
4. **Long-term**: Deep learning on full play-by-play sequences

