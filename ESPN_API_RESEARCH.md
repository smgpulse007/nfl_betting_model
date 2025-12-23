# ESPN API Deep Research for Moneyline Model Enhancement

**Goal:** Maximize data sources independent of Vegas lines to improve moneyline prediction accuracy (currently 68.5%, target 75%+)

---

## 🎯 Current Dataset Audit

### What We Have (nfl-data-py)

**✅ TIER S Features (High Value):**
- CPOE (Completion % Over Expected) - NGS data 2016+
- Pressure Rate - PFR data 2018+
- Injury Impact - 2009-2024 (2025 imputed)
- QB Out status
- Rest Days

**✅ TIER A Features (High Value):**
- RYOE (Rush Yards Over Expected) - NGS data 2016+
- Receiver Separation - NGS data 2016+
- Time to Throw - NGS data 2016+

**✅ TIER 1 Features:**
- Elo Ratings (FiveThirtyEight-style)
- Primetime games (TNF, MNF, SNF)
- Weather (grass/turf, temperature, wind)
- Division games
- Short week flags

**⚠️ Vegas-Dependent Features:**
- Spread line (directly from Vegas)
- Total line (directly from Vegas)
- Moneyline odds (directly from Vegas)
- Implied probabilities (derived from Vegas)

### What We're Missing (Independent of Vegas)

**❌ Real-time 2025 Data:**
- Current week injuries (using 2024 medians)
- Live player status (active/inactive)
- Recent performance trends

**❌ Team Performance Metrics:**
- Offensive/defensive rankings
- Red zone efficiency
- Third down conversion rates
- Turnover differential
- Sack rates (team-level)

**❌ Player-Level Stats:**
- QB rating, completion %, yards per attempt
- RB yards per carry, receptions
- WR targets, catch rate, yards after catch
- Individual player snap counts

**❌ Game Context:**
- Playoff implications
- Rivalry games
- Coaching matchups
- Home/away splits
- Time of possession

---

## 📊 ESPN API Endpoints for NFL (Independent Data)

### 1. **Team Statistics** (CRITICAL)
```
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{year}/types/2/teams/{id}/statistics
```
**Data Available:**
- Offensive stats: total yards, passing yards, rushing yards, points per game
- Defensive stats: yards allowed, points allowed, sacks, interceptions
- Red zone efficiency
- Third down conversion %
- Turnover differential
- Time of possession

**Value:** ⭐⭐⭐⭐⭐ (Independent team performance metrics)

### 2. **Team Records & Splits** (HIGH VALUE)
```
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{year}/types/2/teams/{id}/record
```
**Data Available:**
- Overall record
- Home/away splits
- Division record
- Conference record
- Streak (win/loss)

**Value:** ⭐⭐⭐⭐ (Context for team momentum)

### 3. **Athlete (Player) Stats** (CRITICAL)
```
https://sports.core.api.espn.com/v3/sports/football/nfl/athletes?limit=1000&active=true
https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{id}/stats
```
**Data Available:**
- QB: passing yards, TDs, INTs, completion %, rating
- RB: rushing yards, yards per carry, TDs, receptions
- WR: receptions, yards, TDs, targets
- Season stats + recent game log

**Value:** ⭐⭐⭐⭐⭐ (Key player performance independent of Vegas)

### 4. **Team Roster with Depth Chart** (CRITICAL FOR INJURIES)
```
https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{id}/roster
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{year}/teams/{id}/depthcharts
```
**Data Available:**
- Full roster with positions
- Depth chart (starter vs backup)
- Player status (active, injured, out, questionable)
- Jersey numbers, experience

**Value:** ⭐⭐⭐⭐⭐ (Real-time injury data for 2025)

### 5. **Game Summary with Live Stats** (HIGH VALUE)
```
https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={id}
```
**Data Available:**
- Box score
- Team stats (total yards, turnovers, time of possession)
- Play-by-play
- Injuries (game-specific)
- Weather
- Attendance

**Value:** ⭐⭐⭐⭐ (Comprehensive game context)

### 6. **Leaders (League-Wide Stats)** (MEDIUM VALUE)
```
https://site.api.espn.com/apis/site/v3/sports/football/nfl/leaders
```
**Data Available:**
- Top passers, rushers, receivers
- Defensive leaders (sacks, INTs, tackles)
- Can filter by team

**Value:** ⭐⭐⭐ (Identify star players)

### 7. **Team Schedule with Results** (MEDIUM VALUE)
```
https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{id}/schedule
```
**Data Available:**
- Past results with scores
- Upcoming games
- Opponent strength

**Value:** ⭐⭐⭐ (Strength of schedule)

### 8. **Standings** (MEDIUM VALUE)
```
https://site.api.espn.com/apis/site/v2/sports/football/nfl/standings
```
**Data Available:**
- Division standings
- Conference standings
- Playoff picture
- Clinching scenarios

**Value:** ⭐⭐⭐ (Playoff motivation context)

### 9. **News Feed** (LOW VALUE FOR MODELING)
```
https://site.api.espn.com/apis/site/v2/sports/football/nfl/news
```
**Data Available:**
- Latest news articles
- Injury reports
- Trade news

**Value:** ⭐⭐ (Qualitative, hard to quantify)

---

## 🔥 Priority Data to Integrate (Ranked by Impact)

### **TIER S+ (Must Have - Independent of Vegas)**

1. **Team Offensive/Defensive Stats** ⭐⭐⭐⭐⭐
   - Points per game, yards per game
   - Red zone efficiency
   - Third down conversion %
   - Turnover differential
   - **Why:** Direct measure of team strength, not influenced by Vegas

2. **Live Injury Data (2025)** ⭐⭐⭐⭐⭐
   - Roster status (active/out/questionable)
   - Depth chart changes
   - **Why:** Current model uses 2024 injury data for 2025 games

3. **QB Performance Metrics** ⭐⭐⭐⭐⭐
   - Passing yards, TDs, INTs, completion %, rating
   - Recent 3-game average
   - **Why:** QB is the most important position

### **TIER A (High Value - Independent)**

4. **Home/Away Splits** ⭐⭐⭐⭐
   - Win %, points scored/allowed at home vs away
   - **Why:** Home field advantage varies by team

5. **Recent Form (Last 3-5 Games)** ⭐⭐⭐⭐
   - Win/loss streak
   - Points scored/allowed trend
   - **Why:** Momentum matters

6. **Defensive Stats** ⭐⭐⭐⭐
   - Sacks, interceptions, forced fumbles
   - Yards allowed per game
   - **Why:** Defense wins championships

### **TIER B (Medium Value)**

7. **Strength of Schedule** ⭐⭐⭐
   - Opponent win %
   - **Why:** Context for team record

8. **Playoff Implications** ⭐⭐⭐
   - Clinching scenarios
   - Elimination status
   - **Why:** Motivation factor

---

## 📋 Implementation Plan

### Phase 1: Data Collection (Week 1)
- [ ] Build ESPN API wrapper for team stats
- [ ] Build ESPN API wrapper for player stats
- [ ] Build ESPN API wrapper for live injuries
- [ ] Build ESPN API wrapper for home/away splits
- [ ] Test on 2024 data (validate against known results)

### Phase 2: Feature Engineering (Week 2)
- [ ] Compute offensive/defensive efficiency metrics
- [ ] Compute QB performance rolling averages
- [ ] Integrate live 2025 injury data
- [ ] Compute home/away performance differentials
- [ ] Compute recent form (3-game, 5-game windows)

### Phase 3: Model Retraining (Week 3)
- [ ] Add new features to existing TIER S+A pipeline
- [ ] Retrain XGBoost on 1999-2024 data
- [ ] Validate on 2025 Week 1-16 (should improve from 68.5%)
- [ ] Analyze feature importance (ensure Vegas features drop)

### Phase 4: Production (Week 4)
- [ ] Deploy live data fetching for current week
- [ ] Generate predictions with new features
- [ ] Monitor accuracy on Week 17+
- [ ] Iterate based on results

---

## 🎯 Expected Impact

**Current Moneyline Accuracy:** 68.5% (172/251)
**Target Accuracy:** 75%+ (188/251)
**Improvement Needed:** +6.5% (+16 correct picks)

**Key Hypothesis:**
- Vegas lines already incorporate public information
- Our edge comes from **better weighting** of independent metrics
- XGBoost showed 90% accuracy on 80%+ confidence picks
- More independent features → more confident picks → higher accuracy

**Success Metrics:**
1. Reduce correlation with Vegas from 0.932 to <0.85
2. Increase high-confidence picks (80%+) from 15% to 25% of games
3. Maintain 90%+ accuracy on high-confidence picks
4. Overall accuracy 75%+

---

## 📁 Next Steps

1. **Audit current features** - Which are Vegas-dependent?
2. **Build ESPN data fetchers** - Start with team stats, injuries, QB stats
3. **Backtest on 2024** - Validate new features improve accuracy
4. **Deploy for 2025** - Use live data for remaining weeks


