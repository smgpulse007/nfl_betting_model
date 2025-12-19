# Alternative Data Sources for Independent Edge

## 🚨 The Core Problem

Our current models are **highly correlated with Vegas implied probabilities**:

| Model | Vegas Agreement | Correlation with implied_prob |
|-------|-----------------|-------------------------------|
| Logistic | 100% | r = 0.980 |
| RandomForest | 100% | r = 0.980 |
| CatBoost | 94% | r = 0.977 |
| XGBoost | 81% | r = 0.914 |

**Week 16 2025 Confirmation**: All models agree with Vegas on 81-100% of games.

**Conclusion**: We're just following the market. To find alpha, we need features that Vegas doesn't incorporate or underweights.

---

## ✅ ESPN API - VERIFIED WORKING

We have successfully tested the ESPN unofficial API. Key findings:

### Working Endpoints (No Auth Required)
- ✅ **Injuries** - Real-time injury status for all 32 teams
- ✅ **Depth Charts** - Roster changes, starter updates
- ✅ **Current Odds** - Spreads, totals, moneylines
- ✅ **Player Game Logs** - Historical player stats
- ✅ **Team Statistics** - 11 stat categories
- ✅ **Past Performances** - Historical ATS/O-U records

### Sample Data Retrieved (Week 16 2025)
```
Team Injury Summary:
  ARI : Out:2 | Quest:7  | Key: Xavier Weaver, Marvin Harrison Jr.
  KC  : Quest:8          | Key: Trent McDuffie, Derrick Nnadi
  PHI : Out:2 | Quest:1  | Key: Lane Johnson, Jalen Carter
  SEA : Out:7 | Quest:3  | Key: Riq Woolen, Nick Emmanwori
```

---

## 📊 ESPN Unofficial API Endpoints (No Auth Required)

### Real-Time & Live Data
```
# Live scoreboard (updates during games)
https://cdn.espn.com/core/nfl/scoreboard?xhr=1

# Play-by-play (live)
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{EVENT_ID}/competitions/{EVENT_ID}/plays?limit=300

# Win probabilities (live in-game)
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{EVENT_ID}/competitions/{EVENT_ID}/probabilities?limit=200
```

### Team Data
```
# Team injuries (CRITICAL - live data Vegas might lag on)
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{TEAM_ID}/injuries

# Depth charts (roster changes)
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{YEAR}/teams/{TEAM_ID}/depthcharts

# Team statistics
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{YEAR}/types/{SEASONTYPE}/teams/{TEAM_ID}/statistics
```

### Player Data (for feature engineering)
```
# All active athletes
https://sports.core.api.espn.com/v3/sports/football/nfl/athletes?limit=20000&active=true

# Player game log
https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{ATHLETE_ID}/gamelog

# Player splits
https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{ATHLETE_ID}/splits
```

### Betting/Odds (for line movement tracking)
```
# Current odds
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{EVENT_ID}/competitions/{EVENT_ID}/odds

# Historical odds movement
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{EVENT_ID}/competitions/{EVENT_ID}/odds/{BET_PROVIDER_ID}/history/0/movement?limit=100

# Team past performances (spread, O/U, moneyline)
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{TEAM_ID}/odds/{BET_PROVIDER_ID}/past-performances?limit=200
```

---

## 🎯 Potential Edge Sources (Features Vegas May Underweight)

### 1. **Line Movement / Sharp Money Indicators**
- Track opening vs closing lines
- Large late-game line moves suggest sharp money
- Reverse line movement (line moves opposite to public betting %)

### 2. **Real-Time Injury Updates**
- Vegas sets lines early in the week
- Late injury reports (Friday/Saturday) may not be fully priced in
- Track practice participation (Limited, Full, DNP)

### 3. **Weather at Kickoff (Not Forecasted)**
- Vegas prices in forecasted weather
- Actual gameday conditions may differ
- Wind speed, precipitation, temperature at kickoff

### 4. **Coaching Tendencies / Situational Analytics**
- 4th down aggressiveness
- Red zone play calling
- Tendency changes under new coordinators

### 5. **Rest/Travel Combinations**
- Cross-country travel + short week
- Altitude (Denver games)
- Time zone changes

### 6. **Referee Tendencies**
- Penalty rates per crew
- Home/away flag differential
- Impact on total points

### 7. **Public Betting Percentages (Fade the Public)**
- When >70% public on one side, fade
- Combine with line movement for confirmation

---

## 📁 Other Data Sources to Explore

| Source | Data Type | URL/API |
|--------|-----------|---------|
| **The Odds API** | Live odds from multiple books | https://the-odds-api.com/ |
| **Pro Football Reference** | Historical stats, advanced metrics | https://www.pro-football-reference.com/ |
| **nflfastR** | Play-by-play data, EPA | R package / Python port |
| **Weather API** | Real-time weather | OpenWeatherMap, WeatherAPI |
| **Twitter/X** | Breaking injury news | API (paid) |
| **Action Network** | Public betting % | Web scraping |

---

## 🔬 Next Steps for v0.4.0

1. **Build ESPN API data fetcher** - Get live injuries, depth charts, line movements
2. **Create "Vegas-independent" feature set** - Features that don't use spread_line or implied_prob
3. **Track line movement** - Build historical database of opening/closing lines
4. **Weather integration** - Real-time weather at kickoff
5. **Backtest without market features** - See if we can beat 50% without Vegas lines

---

## 🧪 Experiment: Remove Market Features

To test our true predictive power, try training models WITHOUT:
- `spread_line`
- `total_line`  
- `home_implied_prob`
- `away_implied_prob`

If accuracy drops to ~50%, we have NO independent signal.
If accuracy stays >55%, we have genuine edge.

---

*Last Updated: 2025-12-19*

