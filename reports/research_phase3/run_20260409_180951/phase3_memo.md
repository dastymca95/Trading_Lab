# Phase 3 Review - Cross-Asset Comparison + Translation Candidacy

## Inputs
- Phase 1 run: `C:\Users\Dasty\PycharmProjects\Trading_Lab\reports\research_phase1\run_20260408_214802`
- Phase 2 run: `C:\Users\Dasty\PycharmProjects\Trading_Lab\reports\research_phase2\run_20260409_180003`
- Scope: governed Phase 2 shortlist only; no rerun, no new filters, no runner translation.

## Candidate Scorecards
### DE40 | US_MID | LONG
- Decision: PROMOTE
- Baseline TEST: PF 1.389, expectancy 1.1555, n 318, MDD -30.49
- Consistency: TRAIN PF 1.055; full years PF>1 5/7; rolling PF>1 19/29
- Conditioner notes: NO_FOMC = clean modest improvement: retention 96.86%, delta_pf 0.014, delta_exp 0.0413, MDD worsened -0.76; VOL_LOW = sample collapse: retention 45.91%, delta_pf 0.195, delta_exp -0.0508, delta_mdd 13.43
- Strengths: positive TEST PF/expectancy; TRAIN PF above 1; full-year PF>1 ratio 71.43%; rolling PF>1 ratio 65.52%; NO_FOMC improves cleanly
- Weaknesses: VOL_LOW improvement is sample-collapse sensitive
- Translation difficulty: MEDIUM
- Translation risk: Needs runner translation of fixed-time entry/exit, session clock, spread/commission, and optional FOMC calendar.

### US30 | US_OPEN | LONG
- Decision: PROMOTE WITH CAUTION
- Baseline TEST: PF 1.285, expectancy 2.0428, n 324, MDD -50.46
- Consistency: TRAIN PF 0.972; full years PF>1 4/7; rolling PF>1 15/29
- Conditioner notes: NO_FOMC = not helpful: retention 96.91%, delta_pf -0.024, delta_exp -0.1311; VOL_LOW = sample collapse: retention 43.52%, delta_pf -0.220, delta_exp -1.5921, delta_mdd -3.22
- Strengths: positive TEST PF/expectancy; full-year PF>1 ratio 57.14%; rolling PF>1 ratio 51.72%
- Weaknesses: TRAIN PF below 1; NO_FOMC does not help; VOL_LOW is not helpful and cuts sample below 50%
- Translation difficulty: MEDIUM
- Translation risk: Needs conservative cost/slippage review and train/test parity checks before operational translation.

### DE40 | US_OPEN | LONG
- Decision: PROMOTE WITH CAUTION
- Baseline TEST: PF 1.150, expectancy 0.4639, n 318, MDD -35.17
- Consistency: TRAIN PF 0.937; full years PF>1 4/7; rolling PF>1 13/29
- Conditioner notes: NO_FOMC = clean modest improvement: retention 96.86%, delta_pf 0.015, delta_exp 0.0461, MDD improved 0.72; VOL_LOW = sample collapse: retention 45.91%, delta_pf -0.286, delta_exp -0.8647, delta_mdd -9.90
- Strengths: full-year PF>1 ratio 57.14%; NO_FOMC improves cleanly
- Weaknesses: TRAIN PF below 1; weak rolling PF>1 ratio 44.83%; VOL_LOW is not helpful and cuts sample below 50%
- Translation difficulty: MEDIUM
- Translation risk: Needs proof that weak TRAIN behavior is not masked by TEST-only improvement.

### DE40 | EU_OPEN | SHORT
- Decision: WATCHLIST
- Baseline TEST: PF 1.157, expectancy 0.5856, n 318, MDD -52.29
- Consistency: TRAIN PF 0.997; full years PF>1 3/7; rolling PF>1 14/29
- Conditioner notes: NO_FOMC = not helpful: retention 96.86%, delta_pf -0.009, delta_exp -0.0183; VOL_LOW = sample collapse: retention 45.91%, delta_pf 0.053, delta_exp 0.0351, delta_mdd 26.68
- Strengths: positive TEST PF/expectancy
- Weaknesses: TRAIN PF below 1; weak full-year PF>1 ratio 42.86%; weak rolling PF>1 ratio 48.28%; NO_FOMC does not help; VOL_LOW improvement is sample-collapse sensitive
- Translation difficulty: HIGH
- Translation risk: EU-open execution is sensitive to opening volatility and conditioner sample collapse.

### USTEC | ASIA_EU_CONC | LONG
- Decision: WATCHLIST
- Baseline TEST: PF 1.138, expectancy 0.3651, n 324, MDD -20.92
- Consistency: TRAIN PF 0.952; full years PF>1 2/7; rolling PF>1 16/29
- Conditioner notes: NO_FOMC = not helpful: retention 96.91%, delta_pf -0.002, delta_exp 0.0039; VOL_LOW = sample collapse: retention 42.59%, delta_pf -0.407, delta_exp -0.9394, delta_mdd -17.98
- Strengths: rolling PF>1 ratio 55.17%
- Weaknesses: TRAIN PF below 1; weak full-year PF>1 ratio 28.57%; NO_FOMC does not help; VOL_LOW is not helpful and cuts sample below 50%
- Translation difficulty: HIGH
- Translation risk: Needs stronger clean baseline evidence before any runner translation should be considered.

## Cross-Asset Comparison
- DE40: best US_MID LONG, decision PROMOTE, TEST PF 1.389, promote 1, caution 1, watchlist 1.
- US30: best US_OPEN LONG, decision PROMOTE WITH CAUTION, TEST PF 1.285, promote 0, caution 1, watchlist 0.
- USTEC: best ASIA_EU_CONC LONG, decision WATCHLIST, TEST PF 1.138, promote 0, caution 0, watchlist 1.

## Final Phase 4 Translation Shortlist
1. DE40 | US_MID | LONG | PROMOTE | BASELINE + optional NO_FOMC review
2. US30 | US_OPEN | LONG | PROMOTE WITH CAUTION | BASELINE
3. DE40 | US_OPEN | LONG | PROMOTE WITH CAUTION | BASELINE + optional NO_FOMC review

## Phase 3 Doctrine Check
- Research baseline remains research baseline, not an operational runner.
- VOL_LOW sample-collapse improvements are not used as promotion basis.
- No new families, conditioners, thresholds, or parameter search were introduced.
