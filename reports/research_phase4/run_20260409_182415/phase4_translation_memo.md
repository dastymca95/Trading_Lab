# Phase 4 - Session Candidate Runner V1

## Scope
- Fixed-time operational-candidate backtest only.
- No runtime, paper/live, breakout, ORB, stops, trailing, or new filters.
- Costs are explicit: spread at entry execution plus roundtrip commission.

## Inputs
- Candidate specs: `C:\Users\Dasty\PycharmProjects\Trading_Lab\reports\research_phase4\run_20260409_182415\phase4_candidate_specs.json`
- Phase 3 source run: `C:\Users\Dasty\PycharmProjects\Trading_Lab\reports\research_phase3\run_20260409_181737`

## Candidate Results
### DE40_US_MID_LONG | BASELINE
- TEST: PF 1.389, expectancy 1.1555, n 318, retention 100.00%, MDD -30.49
- TRAIN: PF 1.055, expectancy 0.1038, n 1778
- Full years PF>1: 5/7; rolling PF>1: 19/29

### DE40_US_MID_LONG | NO_FOMC
- TEST: PF 1.403, expectancy 1.1968, n 308, retention 96.86%, MDD -31.25
- TRAIN: PF 1.058, expectancy 0.1086, n 1724
- Full years PF>1: 5/7; rolling PF>1: 20/29

### DE40_US_OPEN_LONG | BASELINE
- TEST: PF 1.150, expectancy 0.4639, n 318, retention 100.00%, MDD -35.17
- TRAIN: PF 0.937, expectancy -0.1320, n 1778
- Full years PF>1: 4/7; rolling PF>1: 13/29

### DE40_US_OPEN_LONG | NO_FOMC
- TEST: PF 1.165, expectancy 0.5100, n 308, retention 96.86%, MDD -34.45
- TRAIN: PF 0.927, expectancy -0.1551, n 1724
- Full years PF>1: 4/7; rolling PF>1: 11/29

### US30_US_OPEN_LONG | BASELINE
- TEST: PF 1.285, expectancy 2.0428, n 324, retention 100.00%, MDD -50.46
- TRAIN: PF 0.972, expectancy -0.1496, n 1807
- Full years PF>1: 4/7; rolling PF>1: 15/29

## Runner-Like Coverage
- Aligned: deterministic entry/exit schedule, fixed direction, one candidate trade per eligible day, costs explicit, trade ledger emitted.
- Still simplified: no live order queue, no slippage model beyond spread, no holiday/calendar provider, no broker execution validation, no runtime state machine.

## Translation Guardrail
- This is candidate research infrastructure. It is not production, paper, live, or london_bot integration.
