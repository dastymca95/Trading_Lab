# Phase 5 - Pre-Operational Evaluation

## Scope
- Three Phase 4 candidates only.
- Perturbations: cost inflation, 30-minute timing shifts, 30-minute hold changes, NO_FOMC where already approved.
- No new filters, families, directions, conditioners, or runtime integration.
- Retention compares each scenario to original BASELINE sample count; timing shifts can exceed 100% at data boundaries.

## Input
- Phase 4 source run: `C:\Users\Dasty\PycharmProjects\Trading_Lab\reports\research_phase4\run_20260409_182845`

## Readiness
### DE40_US_MID_LONG
- Classification: ROBUST
- Baseline TEST: PF 1.389, exp 1.1555; TRAIN PF 1.055
- Perturbation pass: cost 2/2, timing+hold 4/4, total 6/6
- Worst perturbation TEST: PF 1.359, exp 1.0277
- Full years PF>1 5/7; rolling PF>1 19/29
- Event sensitivity: NO_FOMC TEST PF 1.403, exp 1.1968, retention 96.86%

### US30_US_OPEN_LONG
- Classification: ACCEPTABLE
- Baseline TEST: PF 1.285, exp 2.0428; TRAIN PF 0.972
- Perturbation pass: cost 2/2, timing+hold 4/4, total 6/6
- Worst perturbation TEST: PF 1.070, exp 0.4778
- Full years PF>1 4/7; rolling PF>1 15/29
- Event sensitivity: not applicable

### DE40_US_OPEN_LONG
- Classification: FRAGILE
- Baseline TEST: PF 1.150, exp 0.4639; TRAIN PF 0.937
- Perturbation pass: cost 2/2, timing+hold 4/4, total 6/6
- Worst perturbation TEST: PF 1.116, exp 0.3104
- Full years PF>1 4/7; rolling PF>1 13/29
- Event sensitivity: NO_FOMC TEST PF 1.165, exp 0.5100, retention 96.86%

## Scenario Ranking
- DE40_US_MID_LONG | BASELINE: TEST PF 1.389, exp 1.1555, n 318, pass=True
- DE40_US_MID_LONG | COST_UP_1: TEST PF 1.384, exp 1.1430, n 318, pass=True
- DE40_US_MID_LONG | COST_UP_2: TEST PF 1.379, exp 1.1305, n 318, pass=True
- DE40_US_MID_LONG | NO_FOMC: TEST PF 1.403, exp 1.1968, n 308, pass=True
- DE40_US_MID_LONG | HOLD_LONGER_30: TEST PF 1.395, exp 1.2254, n 318, pass=True
- DE40_US_MID_LONG | HOLD_SHORTER_30: TEST PF 1.359, exp 1.0277, n 318, pass=True
- DE40_US_MID_LONG | TIME_SHIFT_EARLY_30: TEST PF 1.371, exp 1.1905, n 318, pass=True
- DE40_US_MID_LONG | TIME_SHIFT_LATE_30: TEST PF 1.370, exp 1.0670, n 318, pass=True
- DE40_US_OPEN_LONG | BASELINE: TEST PF 1.150, exp 0.4639, n 318, pass=True
- DE40_US_OPEN_LONG | COST_UP_1: TEST PF 1.146, exp 0.4514, n 318, pass=True
- DE40_US_OPEN_LONG | COST_UP_2: TEST PF 1.141, exp 0.4390, n 318, pass=True
- DE40_US_OPEN_LONG | NO_FOMC: TEST PF 1.165, exp 0.5100, n 308, pass=True
- DE40_US_OPEN_LONG | HOLD_LONGER_30: TEST PF 1.196, exp 0.6397, n 318, pass=True
- DE40_US_OPEN_LONG | HOLD_SHORTER_30: TEST PF 1.118, exp 0.3104, n 318, pass=True
- DE40_US_OPEN_LONG | TIME_SHIFT_EARLY_30: TEST PF 1.116, exp 0.3188, n 318, pass=True
- DE40_US_OPEN_LONG | TIME_SHIFT_LATE_30: TEST PF 1.201, exp 0.5554, n 318, pass=True
- US30_US_OPEN_LONG | BASELINE: TEST PF 1.285, exp 2.0428, n 324, pass=True
- US30_US_OPEN_LONG | COST_UP_1: TEST PF 1.282, exp 2.0206, n 324, pass=True
- US30_US_OPEN_LONG | COST_UP_2: TEST PF 1.278, exp 1.9984, n 324, pass=True
- US30_US_OPEN_LONG | HOLD_LONGER_30: TEST PF 1.207, exp 1.6120, n 324, pass=True
- US30_US_OPEN_LONG | HOLD_SHORTER_30: TEST PF 1.214, exp 1.4067, n 323, pass=True
- US30_US_OPEN_LONG | TIME_SHIFT_EARLY_30: TEST PF 1.195, exp 1.3111, n 325, pass=True
- US30_US_OPEN_LONG | TIME_SHIFT_LATE_30: TEST PF 1.070, exp 0.4778, n 323, pass=True
