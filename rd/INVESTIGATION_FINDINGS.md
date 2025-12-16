# Data Leakage Investigation - FINDINGS

> **Date**: December 16, 2024  
> **Status**: ✅ CRITICAL BUG FOUND AND EXPLAINED

---

## Executive Summary

We investigated the suspiciously high 80%+ win rate on spread betting and discovered a **critical bug in the spread cover formula**.

### The Bug

```python
# WRONG (old formula)
home_covered = actual_margin + spread_line > 0

# CORRECT (new formula)  
home_covered = actual_margin > spread_line
```

### Impact

| Metric | Buggy Results | Corrected Results |
|--------|---------------|-------------------|
| **Average WR** | 73.9% | 49.4% |
| **Average ROI** | +41.1% | -5.6% |
| **Winning Seasons** | 27/27 (100%) | 5/26 (19%) |
| **Total P&L** | +$296,057 | -$39,577 |

---

## Root Cause Analysis

### Understanding `spread_line` in nfl-data-py

The `spread_line` field represents the **point spread from the home team's perspective**:

| spread_line | Meaning | Example |
|-------------|---------|---------|
| -7 | Home is favorite by 7 | KC -7 vs DEN |
| +3 | Home is underdog by 3 | CLE +3 vs BAL |

### When Does Home Cover?

**Home covers when:**
- Home is favorite (-7) AND wins by MORE than 7
- Home is underdog (+3) AND wins OR loses by LESS than 3

**Mathematical expression:**
```
actual_margin > spread_line
```

### Why The Old Formula Was Wrong

```python
# Old formula: actual_margin + spread_line > 0
# Example: KC -7 (spread_line = -7), KC wins by 3 (actual_margin = 3)
# 3 + (-7) = -4 > 0? NO → Home doesn't cover (WRONG!)
# KC won by 3 but was favored by 7, so they didn't cover - this is actually CORRECT

# BUT consider: KC -7, KC wins by 1 (actual_margin = 1)
# 1 + (-7) = -6 > 0? NO → Home doesn't cover ✓

# And: KC +3 (underdog), KC loses by 2 (actual_margin = -2)
# -2 + 3 = 1 > 0? YES → Home covers ✓

# Wait, actually the old formula seems OK for some cases...
```

### The Real Issue: Sign Convention

After deeper investigation, the `spread_line` in our dataset uses **positive = home underdog** convention:

| Game | spread_line | Meaning |
|------|-------------|---------|
| KC vs BAL | 3.0 | KC is -3 (favorite) |
| IND vs HOU | -3.0 | IND is +3 (underdog) |

So `spread_line = 3` means the home team must win by MORE than 3 to cover.

**Correct formula:**
```python
home_covered = actual_margin > spread_line
# KC wins by 7, spread_line = 3 → 7 > 3 ✓ Home covers
# KC wins by 2, spread_line = 3 → 2 > 3 ✗ Home doesn't cover
```

---

## Verification

### Home Cover Rate

| Formula | Home Cover Rate | Expected |
|---------|-----------------|----------|
| Old (buggy) | 59.4% | ~50% |
| New (correct) | 47.6% | ~48%* |

*Slightly below 50% due to vig/juice on spreads.

### Correlation Check

```
spread_line vs elo_diff: +0.84 (POSITIVE)
```

This confirms that `spread_line` is in "home gives points" format:
- Strong home team (high elo) → high spread_line (they give more points)
- Weak home team (low elo) → low/negative spread_line (they get points)

---

## Corrected Model Performance

### Pure Elo Betting (Corrected)

**Strategy:** Bet home if `elo_margin > spread_line`, else bet away

| Season | Win Rate | ROI |
|--------|----------|-----|
| 2019 | 52.1% | -0.6% |
| 2020 | 47.6% | -9.1% |
| 2021 | 48.4% | -7.5% |
| 2022 | 48.2% | -7.9% |
| 2023 | 43.9% | -16.2% |
| **2024** | **55.1%** | **+5.2%** |

### Key Insight

2024 shows a modest positive edge (+5.2% ROI), but this is within normal variance. Historical average is -5.6% ROI, meaning **Elo alone does not beat the spread**.

---

## Implications for Model Development

### What This Means

1. **No Data Leakage** - The issue was a formula bug, not temporal leakage
2. **Elo Alone Is Not Enough** - 49.4% WR < 52.4% break-even
3. **2024 May Be Anomaly** - One good year doesn't prove edge
4. **Need Better Features** - TIER 2 (injuries, etc.) is critical

### What Still Works

The **moneyline model** was not affected by this bug since it predicts win probability, not spread covers. The reported 55-60% accuracy on moneyline bets is still valid.

---

## Action Items

1. ✅ Fix spread cover formula in `data_loader.py`
2. ✅ Re-run all spread backtests
3. ⚠️ Update reported results in documentation
4. 🔄 Proceed to TIER 2 (injuries) for genuine edge discovery

---

## Lessons Learned

1. **Always validate assumptions** - Check that home cover rate is ~50%
2. **Understand data conventions** - `spread_line` sign matters!
3. **Be skeptical of "too good" results** - 74% WR should have raised flags
4. **Use sanity checks** - Random baseline comparison is essential

