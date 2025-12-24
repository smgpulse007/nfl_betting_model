# Phase 4: Historical Feature Derivation Report

**Date:** 2024-12-24  
**Objective:** Derive 191 approved features (r >= 0.85) for all seasons 1999-2023 and combine with 2024 data  
**Status:** ✅ **COMPLETE**

---

## Executive Summary

Phase 4 successfully derived **191 approved features** for all **26 seasons** (1999-2024), creating a comprehensive training dataset with **829 team-seasons** and **158,339 data points** without any imputation.

### Key Achievements

- ✅ **100% historical coverage**: Derived features for all 25 historical seasons (1999-2023)
- ✅ **191 high-quality features**: Only features with r >= 0.85 correlation to ESPN
- ✅ **829 team-seasons**: 26 years × ~32 teams (31 teams in 1999-2001, 32 teams from 2002+)
- ✅ **No imputation**: All features derived from real historical play-by-play data
- ✅ **No missing values**: 100% data completeness across all features
- ✅ **1.04 MB dataset**: Efficient storage in Parquet format

---

## Methodology

### 1. Feature Selection

**Approved Features (r >= 0.85):**
- Total: 191 features
- EXACT MATCH: 115 features (mean r=0.9900)
- PARTIAL MATCH: 75 features (mean r=0.9780)
- CLOSE APPROXIMATION: 1 feature (QB Rating, r=0.9728)

**Selection Criteria:**
- Pearson correlation r >= 0.85 with ESPN ground truth
- Validated on 2024 season data
- Excludes 58 low-quality features (r < 0.85)

### 2. Data Sources

**nfl-data-py Play-by-Play Data:**
- Years: 1999-2024 (26 seasons)
- Total plays: ~1,150,000 plays
- Regular season only (weeks 1-18)
- Downloaded and cached locally

**nfl-data-py Schedules:**
- Years: 1999-2024
- Regular season games only
- Used for team records and game outcomes

### 3. Derivation Process

**For each year (1999-2023):**
1. Load play-by-play data for the year
2. Load schedules for the year
3. Filter to regular season only (week <= 18)
4. Derive all 368 features for each team
5. Filter to 191 approved features
6. Save year-specific dataset
7. Combine all years into single dataset

**Team Abbreviation Handling:**
- Applied ESPN↔nfl-data-py mapping (LAR↔LA, WSH↔WAS)
- Handled team relocations:
  - St. Louis Rams (STL) → Los Angeles Rams (LAR) in 2016
  - San Diego Chargers (SD) → Los Angeles Chargers (LAC) in 2017
  - Oakland Raiders (OAK) → Las Vegas Raiders (LV) in 2020

### 4. Data Validation

**Quality Checks:**
- ✅ All 25 historical years processed successfully
- ✅ No missing values across all features
- ✅ Consistent feature definitions across all years
- ✅ Team abbreviations correctly mapped

---

## Results

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Rows** | 829 team-seasons |
| **Years Covered** | 26 (1999-2024) |
| **Unique Teams** | 35 (includes relocated teams) |
| **Features** | 191 approved (r >= 0.85) |
| **Total Data Points** | 158,339 |
| **Missing Values** | 0 (100% complete) |
| **File Size** | 1.04 MB (Parquet) |

### Teams Per Year

| Period | Teams | Notes |
|--------|-------|-------|
| 1999-2001 | 31 teams | Before Houston Texans expansion |
| 2002-2024 | 32 teams | After Houston Texans joined |

### Unique Teams (35 total)

**Current Teams (32):**
ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN, DET, GB, HOU, IND, JAX, KC, LAC, LAR, LV, MIA, MIN, NE, NO, NYG, NYJ, PHI, PIT, SEA, SF, TB, TEN, WSH

**Historical Teams (3):**
- **OAK** (Oakland Raiders, 1999-2019) → LV (Las Vegas Raiders, 2020+)
- **SD** (San Diego Chargers, 1999-2016) → LAC (Los Angeles Chargers, 2017+)
- **STL** (St. Louis Rams, 1999-2015) → LAR (Los Angeles Rams, 2016+)

---

## Feature Categories

### Approved Features by Category

| Category | Count | Examples |
|----------|-------|----------|
| **Passing** | 28 | completions, attempts, yards, TDs, INTs, sacks, QB rating |
| **Rushing** | 18 | yards, attempts, TDs, fumbles, first downs |
| **Receiving** | 22 | receptions, yards, TDs, targets, YAC |
| **Defensive** | 16 | sacks, INTs, fumbles, tackles, TDs |
| **Kicking** | 32 | FGs by distance, XPs, long FG |
| **Punting** | 12 | punts, yards, averages, inside 20 |
| **Returning** | 8 | kick/punt returns, TDs, long returns |
| **Scoring** | 12 | TDs by type, FGs, XPs, total points |
| **Team Records** | 28 | wins, losses, home/away records, point differential |
| **General/Misc** | 15 | fumbles, penalties, turnovers, first downs |

---

## Processing Performance

### Execution Time

- **Total time**: ~3 minutes
- **Per year**: ~7 seconds average
- **Per team**: ~0.2 seconds average

### Data Volume Processed

- **Total plays**: ~1,150,000 plays across 26 years
- **Average plays per year**: ~44,000 plays
- **Average plays per team**: ~1,400 plays per season

---

## Output Files

### Primary Dataset

**`data/derived_features/espn_derived_1999_2024_complete.parquet`**
- Shape: (829, 193)
- Columns: team, year, + 191 features
- Size: 1.04 MB
- Format: Parquet (compressed, efficient)

### Historical Data (by year)

**`data/derived_features/historical/espn_derived_YYYY.parquet`**
- 25 files (1999-2023)
- One file per year
- Same 191 features across all years

### Intermediate Files

**`data/derived_features/espn_derived_1999_2023.parquet`**
- Historical data only (1999-2023)
- Shape: (797, 193)
- Used for combining with 2024 data

---

## Data Quality

### Completeness

- ✅ **100% feature coverage**: All 191 approved features derived for all years
- ✅ **100% data completeness**: No missing values
- ✅ **100% year coverage**: All 26 years successfully processed

### Consistency

- ✅ **Consistent definitions**: Same derivation logic across all years
- ✅ **Consistent team mapping**: Proper handling of team relocations
- ✅ **Consistent filtering**: Regular season only (weeks 1-18) for all years

### Accuracy

- ✅ **Validated on 2024**: All features validated against ESPN ground truth
- ✅ **High correlations**: Mean r=0.9852, median r=0.9998
- ✅ **56 perfect correlations**: Features with r=1.0000

---

## Next Steps

### Immediate Actions

1. ✅ **Dataset ready for model training**
   - 829 team-seasons
   - 191 high-quality features
   - 26 years of historical data

2. **Feature Engineering** (recommended)
   - Calculate rolling averages (3-game, 5-game, season)
   - Create momentum indicators
   - Add opponent-adjusted metrics
   - Calculate year-over-year changes

3. **Model Training** (ready to proceed)
   - Combine with 32 TIER S+A features = 223 total features
   - Train on 1999-2023 data (797 team-seasons)
   - Validate on 2024 data (32 teams)
   - Compare performance vs. Vegas lines

### Future Enhancements

1. **Add game-level features**
   - Derive features at game level instead of season level
   - Enable week-by-week predictions
   - Increase training data to ~7,000 games

2. **Add opponent features**
   - Derive opponent stats for each game
   - Calculate strength of schedule
   - Add head-to-head matchup features

3. **Add advanced metrics**
   - EPA (Expected Points Added)
   - Success rate
   - CPOE (Completion Percentage Over Expected)
   - Win probability added

---

## Conclusion

Phase 4 successfully created a comprehensive training dataset spanning **26 years** (1999-2024) with **191 high-quality features** and **zero missing values**. This dataset enables:

1. **Historical model training** without imputation bias
2. **Robust validation** on 26 years of data
3. **Independent feature set** to complement existing TIER S+A features
4. **Production-ready pipeline** for real-time feature derivation

**Status:** ✅ **READY FOR MODEL TRAINING**

---

**Next Phase:** Phase 5 - Model Training and Evaluation
