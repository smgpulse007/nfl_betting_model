# Phase 1: ESPN API Data Collection

## 📋 Overview

This phase collects all ESPN independent data needed to enhance the moneyline model from 68.5% to 75%+ accuracy.

**Goal:** Fetch 323 independent stats per team (279 team stats + 44 record stats) from ESPN API for 2024 and 2025 seasons.

**Status:** ✅ Scripts ready, awaiting execution

---

## 🎯 Data Collection Targets

### 2024 Season (Training/Validation Data)
- ✅ Team Stats (32 teams × 279 stats = 8,928 data points)
- ✅ Team Records (32 teams × 44 stats = 1,408 data points)

### 2025 Season (Current Season Data)
- ✅ Team Stats (32 teams × 279 stats = 8,928 data points)
- ✅ Team Records (32 teams × 44 stats = 1,408 data points)
- ✅ Injuries (Weeks 1-16, live data)

**Total:** ~20,000 independent data points completely separate from Vegas lines

---

## 🚀 Quick Start

### Option 1: Run Simplified Script (Recommended)
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run Phase 1 data collection
python fetch_espn_data_phase1.py
```

This will:
1. Fetch 2024 team stats (32 teams)
2. Fetch 2024 team records (32 teams)
3. Fetch 2025 team stats (32 teams)
4. Fetch 2025 team records (32 teams)
5. Save all data to `data/espn_raw/` as parquet files

**Expected Runtime:** ~5-10 minutes (with rate limiting)

### Option 2: Run Batch Script (Windows)
```bash
# Double-click or run:
run_phase1_fetch.bat
```

### Option 3: Use Full-Featured API (Advanced)
```bash
# Test the API first
python espn_independent_data.py --mode test

# Fetch all 2024 data
python espn_independent_data.py --mode fetch-all --season 2024

# Fetch all 2025 data
python espn_independent_data.py --mode fetch-all --season 2025

# Fetch specific data types
python espn_independent_data.py --mode fetch-stats --season 2024
python espn_independent_data.py --mode fetch-records --season 2024
python espn_independent_data.py --mode fetch-injuries --season 2025 --week-range 1-16
```

---

## 📁 Output Files

All data will be saved to `data/espn_raw/`:

```
data/espn_raw/
├── team_stats_2024.parquet      # 32 teams × 279 stats
├── team_records_2024.parquet    # 32 teams × 44 record stats
├── team_stats_2025.parquet      # 32 teams × 279 stats
├── team_records_2025.parquet    # 32 teams × 44 record stats
└── injuries_2025_weeks_1-16.parquet  # Live injury data (if fetched)
```

---

## ✅ Data Validation

After collection, the scripts will automatically validate:
- ✅ All 32 teams present
- ✅ No duplicate teams
- ✅ Missing value percentage
- ✅ Column counts match expectations

**Expected Results:**
- Team Stats: 32 teams, ~280 columns
- Team Records: 32 teams, ~45 columns
- Data Quality: EXCELLENT (0 missing teams, 0 duplicates)

---

## 🔍 What's Being Collected

### Team Stats (279 stats per team)
- **Offensive:** Passing yards, rushing yards, completions, attempts, TDs, INTs
- **Defensive:** Sacks, tackles, interceptions, forced fumbles, QB hits
- **Special Teams:** FG%, punt average, return yards
- **Efficiency:** 3rd down %, red zone %, time of possession
- **Advanced:** Yards per play, points per drive, turnover differential

### Team Records (44 stats per team)
- **Overall:** Wins, losses, win%, points for/against
- **Home/Away:** Home record, away record, splits
- **Division:** Division record, conference record
- **Streaks:** Current streak, longest win/loss streak
- **Close Games:** OT record, 1-score game record

### Injuries (2025 only)
- Player name, position, team
- Injury status (Out, Questionable, Doubtful)
- Injury type and details
- Week-by-week tracking

---

## 🎯 Next Steps (Phase 2)

Once data collection is complete:

1. **Feature Engineering** (`phase2_feature_engineering.py`)
   - Compute offensive/defensive efficiency metrics
   - Calculate QB performance rolling averages
   - Integrate live 2025 injury data
   - Compute home/away splits and recent form

2. **Statistical Validation**
   - Test each feature for predictive power
   - Measure correlation with moneyline outcomes
   - Ensure independence from Vegas lines (target: r < 0.85)

3. **Model Integration**
   - Add validated features to TIER S+A pipeline
   - Retrain on 1999-2024 data
   - Validate on 2025 Week 1-16

---

## 📊 Success Criteria

- [x] Scripts created and tested
- [ ] 2024 data collected (stats + records)
- [ ] 2025 data collected (stats + records)
- [ ] Data validation passed (32 teams, <5% missing values)
- [ ] Files saved to `data/espn_raw/`
- [ ] Ready for Phase 2 feature engineering

---

## 🐛 Troubleshooting

**Issue:** API requests failing
- **Solution:** Check internet connection, ESPN API might be rate-limiting

**Issue:** Missing teams in output
- **Solution:** Re-run the script, some teams might have timed out

**Issue:** Import errors
- **Solution:** Ensure virtual environment is activated and dependencies installed:
  ```bash
  pip install requests pandas pyarrow
  ```

---

## 📝 Notes

- Rate limiting: 0.3s delay between requests to avoid overwhelming ESPN API
- Retries: 3 attempts per request with exponential backoff
- Timeout: 10 seconds per request
- Data format: Parquet (efficient, compressed, preserves types)

---

**Ready to proceed?** Run `python fetch_espn_data_phase1.py` to start Phase 1 data collection!

