# PHASE 5A: SINGLE GAME DERIVATION & VALIDATION REPORT
====================================================================================================

**Date:** 2025-12-24 10:18:38
**Status:** ✅ PASSED

## 1. Test Game Information

- **Game ID:** `2024_01_BAL_KC`
- **Teams:** BAL vs KC
- **Week:** 1
- **Season:** 2024

## 2. Game Results

### BAL (LOSS)

- **Points:** 20
- **Passing Yards:** 261
- **Rushing Yards:** 185
- **Turnovers:** 1
- **Third Down %:** 43.8%

### KC (WIN)

- **Points:** 27
- **Passing Yards:** 271
- **Rushing Yards:** 73
- **Turnovers:** 1
- **Third Down %:** 40.0%

## 3. Feature Coverage

- **Approved Features (r >= 0.85):** 191
- **Derived Features:** 368
- **Coverage:** 191/191 (100.0%)
- **Status:** ✅ 100% coverage achieved

## 4. Data Quality

- **Missing Values (in approved features):** 0
- **Completeness:** 100.0%
- **Status:** ✅ Zero missing values

## 5. Validation Criteria

| Criterion | Required | Actual | Status |
|-----------|----------|--------|--------|
| Feature Coverage | 100% | 100.0% | ✅ PASS |
| Missing Values | 0 | 0 | ✅ PASS |
| Data Integrity | Valid | Valid | ✅ PASS |

## 6. Next Steps

✅ **Phase 5A Complete** - Single game derivation validated

**Proceed to Phase 5B:**
- Derive features for all 2024 games (~576 team-games)
- Validate aggregation to season level (r > 0.95)
- Create comprehensive validation report

## 7. Feature Categories Derived

| Category | Count |
|----------|-------|
| defensive | 11 |
| defensiveInterceptions | 3 |
| general | 3 |
| home | 4 |
| kicking | 31 |
| miscellaneous | 13 |
| passing | 37 |
| punting | 14 |
| receiving | 18 |
| returning | 11 |
| road | 4 |
| rushing | 18 |
| scoring | 12 |
| total | 12 |
