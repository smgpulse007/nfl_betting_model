# Moneyline Model Enhancement Plan
## Focus: Independent Data Sources to Reduce Vegas Dependency

**Current Status:** 68.5% accuracy (172/251), 90% on high-confidence picks (80%+)  
**Target:** 75%+ accuracy, 25% high-confidence picks (vs current 15%)  
**Key Strategy:** Add independent features from ESPN API, reduce Vegas line dependency

---

## 📊 Current Feature Audit Results

### Vegas-Dependent Features (13 total) ❌
- `spread_line`, `total_line`
- `home_moneyline`, `away_moneyline`
- `home_implied_prob`, `away_implied_prob`
- `home_spread_odds`, `away_spread_odds`
- `over_odds`, `under_odds`

**Problem:** Model correlation with Vegas = 0.932 (too high)

### Independent Features (73 total) ✅
**TIER S (Currently Have):**
- CPOE (Completion % Over Expected)
- Pressure Rate
- Injury Impact (2024 only, 2025 imputed)
- QB Out status
- Rest Days

**TIER A (Currently Have):**
- RYOE (Rush Yards Over Expected)
- Receiver Separation
- Time to Throw

**TIER 1 (Currently Have):**
- Elo Ratings
- Weather (temp, wind, surface)
- Primetime flags
- Division games

**Missing (ESPN API Can Provide):**
- Team offensive/defensive stats
- QB/RB/WR performance metrics
- Home/away splits
- Recent form (3-5 game windows)
- Live 2025 injury data
- Turnover differential
- Third down conversion %
- Red zone efficiency

---

## 🔥 ESPN API Capabilities (Tested & Verified)

### ✅ Working Endpoints

**1. Team Statistics** (279 stats per team!)
```
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/types/2/teams/12/statistics
```
**Available Data:**
- Fumbles, fumbles lost, fumbles forced, fumbles recovered
- Offensive stats: passing yards, rushing yards, total yards
- Defensive stats: yards allowed, points allowed, sacks
- Efficiency: red zone %, third down %, turnovers
- **Value:** ⭐⭐⭐⭐⭐ (Independent team performance)

**2. Team Record with Splits** (44 stats per team!)
```
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/types/2/teams/12/record
```
**Available Data:**
- OT wins/losses
- Average points for/against
- Home/away splits
- Division/conference record
- Streak information
- **Value:** ⭐⭐⭐⭐⭐ (Context for team performance)

**3. Team Roster**
```
https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/12/roster
```
**Available Data:**
- Player positions
- Jersey numbers
- Status (active/injured)
- Experience
- **Value:** ⭐⭐⭐⭐ (Injury tracking)

**4. Game Injuries**
```
https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={id}
```
**Available Data:**
- Injured players by team
- Position
- Status (out, questionable, doubtful)
- Injury details
- **Value:** ⭐⭐⭐⭐⭐ (Critical for 2025 games)

---

## 📋 Implementation Roadmap

### Phase 1: Data Collection (Week 1) ✅ IN PROGRESS

**Completed:**
- [x] ESPN API research and documentation
- [x] Current feature audit (identified 13 Vegas-dependent features)
- [x] Built `espn_independent_data.py` wrapper
- [x] Tested team stats API (279 stats available)
- [x] Tested team record API (44 stats available)
- [x] Tested roster API (injury tracking)

**Next Steps:**
- [ ] Fetch all 32 teams' stats for 2024 season
- [ ] Fetch all 32 teams' stats for 2025 season (current)
- [ ] Fetch game-by-game injuries for 2025 Week 1-16
- [ ] Parse and structure data into features
- [ ] Save to parquet files for integration

### Phase 2: Feature Engineering (Week 2)

**Offensive Efficiency Features:**
- [ ] Points per game (rolling 3-game, 5-game, season)
- [ ] Yards per game (passing, rushing, total)
- [ ] Red zone efficiency (TDs per red zone trip)
- [ ] Third down conversion %
- [ ] Turnover differential

**Defensive Efficiency Features:**
- [ ] Points allowed per game
- [ ] Yards allowed per game
- [ ] Sacks per game
- [ ] Interceptions per game
- [ ] Opponent third down conversion %

**Context Features:**
- [ ] Home/away point differential
- [ ] Recent form (win % last 3, 5 games)
- [ ] Streak (current win/loss streak)
- [ ] Division record
- [ ] Conference record

**Player Features (QB focus):**
- [ ] QB rating (rolling 3-game)
- [ ] Completion % (rolling 3-game)
- [ ] Yards per attempt (rolling 3-game)
- [ ] TD/INT ratio (rolling 3-game)

**Injury Features (2025 Live Data):**
- [ ] Replace imputed 2025 injuries with real ESPN data
- [ ] QB out status (live)
- [ ] Star player out count (WR1, RB1, etc.)
- [ ] Injury impact score (position-weighted)

### Phase 3: Model Retraining (Week 3)

**Data Preparation:**
- [ ] Merge ESPN features with existing TIER S+A features
- [ ] Remove or downweight Vegas-dependent features
- [ ] Normalize/scale new features
- [ ] Handle missing data (imputation strategy)

**Model Training:**
- [ ] Train XGBoost on 1999-2024 with new features
- [ ] Train Logistic Regression on 1999-2024 with new features
- [ ] Validate on 2025 Week 1-16 (should improve from 68.5%)
- [ ] Analyze feature importance (ensure Vegas features drop)

**Evaluation:**
- [ ] Overall accuracy (target: 75%+)
- [ ] High-confidence picks (target: 25% of games at 80%+ confidence)
- [ ] Correlation with Vegas (target: <0.85, down from 0.932)
- [ ] ROI analysis (Kelly Criterion)

### Phase 4: Production Deployment (Week 4)

**Live Data Pipeline:**
- [ ] Automated ESPN API fetching for current week
- [ ] Real-time injury updates
- [ ] Team stats updates (weekly)
- [ ] Player stats updates (weekly)

**Prediction Generation:**
- [ ] Generate predictions for upcoming week
- [ ] Confidence scoring
- [ ] Betting recommendations (high-confidence only)
- [ ] Track results and accuracy

**Monitoring:**
- [ ] Weekly accuracy reports
- [ ] Feature drift detection
- [ ] Model recalibration (if needed)
- [ ] A/B testing (old vs new model)

---

## 🎯 Success Metrics

### Primary Metrics
1. **Overall Accuracy:** 75%+ (vs current 68.5%)
2. **High-Confidence Accuracy:** 90%+ on 80%+ confidence picks (maintain current)
3. **High-Confidence Volume:** 25% of games (vs current 15%)
4. **Vegas Correlation:** <0.85 (vs current 0.932)

### Secondary Metrics
5. **ROI:** Positive returns with Kelly Criterion betting
6. **Feature Independence:** Top 10 features should be non-Vegas
7. **2025 Data Quality:** Real injury data vs imputed
8. **Prediction Stability:** Low variance week-to-week

---

## 📁 Files Created

1. **`ESPN_API_RESEARCH.md`** - Comprehensive ESPN API documentation
2. **`audit_current_features.py`** - Feature dependency analysis
3. **`espn_independent_data.py`** - ESPN API wrapper (tested & working)
4. **`MONEYLINE_ENHANCEMENT_PLAN.md`** - This document

---

## 🚀 Next Immediate Actions

1. **Fetch 2024 Team Stats** - Get baseline for all 32 teams
2. **Fetch 2025 Team Stats** - Get current season data
3. **Fetch 2025 Injuries** - Replace imputed data with real data
4. **Build Feature Pipeline** - Integrate ESPN data with existing features
5. **Retrain Model** - Test on 2025 Week 1-16 to validate improvement

**Estimated Timeline:** 3-4 weeks to production-ready enhanced model


