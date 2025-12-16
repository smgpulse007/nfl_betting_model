# NFL Betting Model - R&D Enhancement Roadmap

> **Generated**: December 16, 2024  
> **Branch**: `feature/model-enhancements-rd`  
> **Status**: Planning Phase

---

## Executive Summary

Based on data profiling of 6,991 games (1999-2024) and exploration of available data sources, 
this document outlines enhancement opportunities ranked by **ROI on Effort**.

### Current Model Performance (2025 Backtest)
| Bet Type | Accuracy | ROI | Status |
|----------|----------|-----|--------|
| Moneyline | 55.1% | -1.4% | ⚠️ Needs improvement |
| Spread | 79.8%* | +48%* | *Inflated due to feature design |
| Totals | 54.3% | +2.1% | ✅ Slight edge |

---

## 🎯 Enhancement Tiers (Ranked by ROI/Effort)

### TIER 1: High ROI, Low Effort (Do First)
| Enhancement | Est. ROI | Effort | Data Available | Notes |
|-------------|----------|--------|----------------|-------|
| **Rolling EPA Features** | +2-3% | 2 days | ✅ PBP data | Off/Def EPA over 3/5 weeks |
| **Primetime Flags** | +0.5% | 2 hrs | ✅ Schedules | TNF/SNF/MNF have different dynamics |
| **QB-in-Game Flag** | +1-2% | 4 hrs | ✅ Schedules | `home_qb_id`, `away_qb_id` in data |
| **Surface Type** | +0.5% | 2 hrs | ✅ Schedules | Grass vs turf affects totals |
| **Extreme Weather** | +0.5% | 2 hrs | ✅ Schedules | Wind >20mph, precip flags |

### TIER 2: High ROI, Medium Effort (Strategic Investments)
| Enhancement | Est. ROI | Effort | Data Available | Notes |
|-------------|----------|--------|----------------|-------|
| **Injury Impact Score** | +3-5% | 1 week | ✅ 6,215 records | Weight by position WAR |
| **QB Elo (Separate)** | +2-3% | 3 days | ✅ PBP + rosters | Track QB quality separately |
| **Recent Performance** | +1-2% | 2 days | ✅ Weekly data | Last 3 games momentum |
| **Line Movement** | +2-4% | 1 week | ⚠️ API needed | Opening vs closing lines |
| **Travel/Rest Deep** | +1% | 1 day | ✅ Schedules | Cross-country, timezone |

### TIER 3: Medium ROI, High Effort (R&D Phase)
| Enhancement | Est. ROI | Effort | Data Available | Notes |
|-------------|----------|--------|----------------|-------|
| **LLM Sentiment** | +1-3% | 2 weeks | ⚠️ Scraping | News/Twitter via Ollama |
| **Next Gen Stats** | +1-2% | 1 week | ✅ 614 records | Time to throw, separation |
| **Public Betting %** | +2-3% | 2 weeks | ⚠️ API needed | Fade the public signals |
| **Coaching Situational** | +1% | 1 week | Manual | New coach, playoff scenarios |

### TIER 4: Low Priority / Speculative
| Enhancement | Est. ROI | Effort | Notes |
|-------------|----------|--------|-------|
| Referee tendencies | +0.5% | 3 days | 99.9% data available |
| Altitude effects | +0.2% | 2 hrs | Only Denver matters |
| Rivalry intensity | +0.3% | 1 day | Hard to quantify |

---

## 📊 Data Profiling Findings

### Missing Data Summary
| Field | Missing % | Impact |
|-------|-----------|--------|
| `nfl_detail_id` | 96% | Not needed |
| Moneyline odds | 28% | Pre-2015 only, manageable |
| `home_implied_prob` | 28% | Same as above |
| `surface` | 0.6% | Easy to backfill |

### Unused Columns (Quick Wins)
- `weekday` - Thursday games are different
- `overtime` - Useful for totals analysis  
- `away_qb_id` / `home_qb_id` - QB matching
- `roof` - More granular than `is_dome`
- `result` / `total` - Already have but not using

---

## 🤖 LLM Integration Strategy (Ollama/vLLM)

### Architecture Options
```
Option A: Local Ollama (Simpler)
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ News/Social │ --> │ Ollama API   │ --> │ Sentiment   │
│ Scraper     │     │ (llama3.2)   │     │ Features    │
└─────────────┘     └──────────────┘     └─────────────┘

Option B: vLLM Container (Faster)
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ News/Social │ --> │ vLLM Server  │ --> │ Structured  │
│ Scraper     │     │ (RTX 4090)   │     │ Reasoning   │
└─────────────┘     └──────────────┘     └─────────────┘
```

### Use Cases (Ranked by Value)
1. **Injury Context** (+High) - "Hamstring" vs "ACL" severity parsing
2. **News Sentiment** (+Medium) - Locker room drama, coach conflicts  
3. **Weather Reasoning** (+Low) - Already have structured data
4. **Game Narrative** (+Low) - "Revenge game" detection

### Recommended Model
- **Ollama + Llama 3.2 8B** for quick sentiment
- **vLLM + Mistral 7B** for batch processing game previews
- RTX 4090 can run 7B models at ~50 tokens/sec

---

## 🎲 Variance Modeling Strategy

### High Variance Scenarios (Reduce Bet Size)
- Backup QB starting (0.5x Kelly)
- Wind >20 mph (0.7x Kelly for totals)
- Division rivals late season (0.8x Kelly)
- Primetime road underdogs (0.7x Kelly)

### Low Variance Scenarios (Increase Bet Size)
- Strong favorite at home, good weather (1.2x Kelly)
- Elo and Vegas strongly agree (1.1x Kelly)
- Blowout setup (top-5 vs bottom-5 team)

---

## 📅 Implementation Timeline

### Week 1: Quick Wins (Tier 1)
- [ ] Add rolling EPA features
- [ ] Add primetime/surface/extreme weather flags
- [ ] Backtest impact

### Week 2-3: Injuries + QB Elo
- [ ] Build injury impact scoring system
- [ ] Implement QB-specific Elo
- [ ] Re-run 2024-2025 backtests

### Week 4: LLM Prototype
- [ ] Set up Ollama with Llama 3.2
- [ ] Build news scraper for game previews
- [ ] Extract injury severity signals

### Week 5-6: Line Movement + Public Betting
- [ ] Integrate The Odds API for live odds
- [ ] Track opening vs closing spreads
- [ ] Build fade-the-public signals

---

## 🔬 Success Metrics

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| ML ROI | -1.4% | +3% | Break even minimum |
| Spread Edge | ~5%? | +8% | After proper evaluation |
| Totals ROI | +2.1% | +5% | Best current performer |
| Brier Score | TBD | <0.22 | Calibration measure |

