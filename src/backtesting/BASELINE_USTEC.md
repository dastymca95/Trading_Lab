# USTEC — Operational Baseline Specification
## STATUS: LOCKED BASELINE

> This document is the authoritative record of the USTEC operational specification
> as validated and frozen at this checkpoint. Any future change to parameters or
> filters must be preceded by a new validation cycle and a replacement of this file.

---

## Specification

| Parameter        | Value                                    |
|------------------|------------------------------------------|
| Asset            | USTEC (Nasdaq 100 CFD)                   |
| Slot             | MT5 hour 18 only (UTC 16:00–17:59)       |
| Direction        | Forced LONG (`force_direction: 1`)       |
| SL               | 0.3% of price (`sl_pct: 0.003`)          |
| Trailing mult    | 3.0× ATR (`trail_mult: 3.0`)             |
| Breakeven mult   | 0.75× ATR (`be_atr_mult: 0.75`)          |
| ATR mult (entry) | 1.5 (`atr_mult: 1.5`)                    |
| LRR min          | 1.0 (`lrr_min: 1.0`)                     |
| Risk per trade   | 2% of capital (`risk_pct: 0.02`)         |
| Days of week     | Mon–Fri (`dow: [0,1,2,3,4]`)             |
| Commission       | 0.00 USD                                 |
| Spread proxy     | 1.0 point (`sp: 1.0`)                    |
| Digits           | 2                                        |

### Daily regime filter

| Filter            | Value                                    |
|-------------------|------------------------------------------|
| LOW_VOL percentile | 50th (`low_vol_pct: 50`)               |
| LOW_VOL window    | 90 days (`low_vol_win: 90`)              |
| roll_mfe N        | 20 LOW_VOL sessions (`roll_mfe_n: 20`)  |
| roll_mfe shift    | 1 (look-ahead free)                      |
| roll_mfe threshold| > 1.0 (`roll_mfe_min: 1.0`)             |

**Filter logic** (implemented in `backtest_runner._build_daily_filter`):

1. `daily_atr` = mean of bar-level ATR14 per calendar day
2. `rolling_thr` = rolling quantile(p=0.50, window=90) of `daily_atr`
3. LOW_VOL day: `daily_atr ≤ rolling_thr` (≥45 days warm-up)
4. Session MFE/MAE: hours 18+19 (MT5) = UTC 16–17. `mfe_mae = (high−open)/(open−low)`, MAE > 1 required
5. `mfe_roll` = rolling median over last 20 **LOW_VOL** sessions (shift=1)
6. Trade allowed: LOW_VOL day **AND** `mfe_roll > 1.0`

---

## Rationale

### Research alignment
- Research (fixed-window, always-LONG baseline) isolated a **US MID-session early LONG bias** on low-volatility days during UTC 16–17 (MT5 18–19).
- The roll_mfe gate selects regimes where that session consistently produces positive asymmetry (MFE > MAE).
- Research → engine parity corrections applied: session window matches (MT5 18+19), rolling domain restricted to LOW_VOL days only.

### Transfer gap closed
- Bidirectional breakout with `hours=[15,18]` (London + US) produced no edge because London contamination (MT5 15 = UTC 13) obscured the directional filter.
- Isolating `hours=[18]` + `force_direction=1` recovered the research edge in the operational engine.

### Portfolio context
- Asset pruning (5 combinations) concluded USTEC solo dominates all multi-asset combinations on both FULL history and TEST period metrics.
- US30 pruned (PF=0.87 FULL). DE40 pruned (PF=0.92 FULL). US500 in observation only.

---

## Validation results (at lock date)

> Engine: `backtest_runner.py` | Capital per asset: $250 | Risk: 2%

### FULL history

| Metric        | Value      |
|---------------|------------|
| Trades        | (see run)  |
| Win rate      | (see run)  |
| Profit factor | **4.06**   |
| Return        | (see run)  |
| Max drawdown  | (see run)  |

### TEST period (2025-01-01 → present)

| Metric        | Value      |
|---------------|------------|
| Profit factor | **4.95**   |
| (full metrics)| (see run)  |

> Run `python ustec_validation.py` for the complete 9-section validation report.
> Run `python portfolio_pruning.py` to reproduce the portfolio comparison.
> Run `python ustec_comparison.py` to reproduce BASE vs REFINED vs OPERATIONAL comparison.

---

## Lock declaration

```
Asset     : USTEC
Status    : LOCKED BASELINE
Locked by : Diego
Lock date : 2026-04-07
Config    : src/backtesting/backtest_config.py — OPTIONAL_PARAMS['USTEC']
Engine    : src/backtesting/backtest_runner.py — _build_daily_filter + force_direction
Rationale : Aligned with research (UTC 16-17 LONG bias on LOW_VOL+roll_mfe days).
            Validated in operational engine. Dominates all portfolio combinations tested.
```

Any modification to the USTEC entry in `backtest_config.py` must:
1. Re-run `ustec_validation.py` and confirm ≥10/12 criteria pass
2. Re-run `portfolio_pruning.py` and confirm USTEC solo remains competitive
3. Update this file with new metrics and a new lock date
4. Commit with message: `feat(USTEC): update operational baseline — <reason>`
