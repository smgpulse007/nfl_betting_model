# Phase 5A Strategy Update

## Current Status

✅ **Created basic game-level derivation function** (`derive_game_features.py`)  
✅ **Successfully tested on BAL @ KC Week 1 2024**  
✅ **Zero missing values in derived features**  
❌ **Only 52/191 approved features derived (27.2% coverage)**

## Problem

The current `derive_game_features()` function only implements ~50 features. We need all 191 approved features.

Missing feature categories:
- Receiving features (receptions, yards, TDs, targets, YAC)
- Detailed kicking features (FG ranges, blocked kicks)
- Detailed punting features (inside 20, touchbacks, blocked)
- Detailed defensive features (tackles, sack yards, INT details)
- Scoring features (aggregated scoring stats)
- Miscellaneous features (first downs breakdown, turnovers)
- Home/road record features (season-level aggregations)

## Revised Strategy

### Option A: Manual Feature Addition (SLOW, ERROR-PRONE)
- Add missing features one category at a time
- Requires ~10-15 code edits (150 lines each)
- High risk of bugs and inconsistencies
- Estimated time: 4-6 hours

### Option B: Adapt Existing Function (FAST, RELIABLE) ✅ RECOMMENDED
- Copy logic from `full_feature_derivation.py` (1,503 lines)
- Modify to work on game-level data:
  1. Remove `games_played` variable (always = 1 for single game)
  2. Remove per-game calculations (e.g., `yardsPerGame`)
  3. Keep all counting stats and percentages
  4. Filter to single game_id instead of full season
- Estimated time: 1-2 hours

## Implementation Plan (Option B)

### Step 1: Create Comprehensive Game-Level Function
1. Copy `derive_all_features()` from `full_feature_derivation.py`
2. Rename to `derive_game_features_complete()`
3. Modify function signature:
   ```python
   def derive_game_features_complete(team: str, game_id: str, pbp: pd.DataFrame, schedules: pd.DataFrame) -> dict
   ```
4. Add game filtering at the start:
   ```python
   pbp_game = pbp[pbp['game_id'] == game_id].copy()
   schedule_game = schedules[schedules['game_id'] == game_id].copy()
   ```
5. Replace all `pbp_reg` with `pbp_game`
6. Replace all `schedules_reg` with `schedule_game`
7. Remove `games_played` variable
8. Remove all `/games_played` calculations (per-game stats)
9. Keep all other logic identical

### Step 2: Test on Single Game
1. Run on BAL @ KC Week 1 2024
2. Verify all 191 approved features are derived
3. Check for missing values
4. Validate key features against ESPN

### Step 3: Validation Checkpoint
- ✅ 191/191 features derived (100% coverage)
- ✅ Zero missing values
- ✅ Key features match ESPN (within tolerance)
- ✅ Ready for Phase 5B (full 2024 season)

## Decision

**PROCEED WITH OPTION B**

Rationale:
- Faster implementation (1-2 hours vs 4-6 hours)
- Lower error risk (reusing tested logic)
- Higher confidence in correctness
- Easier to maintain and debug

## Next Steps

1. Create `derive_game_features_complete.py` with full feature set
2. Test on single game (BAL @ KC)
3. Verify 100% feature coverage
4. If validation passes → Proceed to Phase 5B
5. If validation fails → Debug and re-test

## Files to Create/Modify

**New Files:**
- `game_level/derive_game_features_complete.py` (comprehensive function)
- `game_level/test_complete_derivation.py` (test script)

**Existing Files (keep for reference):**
- `game_level/derive_game_features.py` (partial implementation - archive)
- `game_level/test_single_game.py` (basic test - archive)

## Success Criteria

✅ All 191 approved features derived  
✅ Zero missing values  
✅ Matches ESPN validation (where possible)  
✅ Ready for Phase 5B (full 2024 season derivation)

