# USTEC Paper/Shadow Readiness Specification — v1
## Phase 10.1

**Status**: PRE-PAPER — two blocking gaps must be closed first (see §6)
**Locked by**: Diego
**Date**: 2026-04-08

---

## 1. Operational Specification

| Parameter               | Value                                              |
|-------------------------|----------------------------------------------------|
| Asset                   | USTEC (Nasdaq 100 CFD)                             |
| Timeframe               | M2 (2-minute candles)                              |
| Signal slot             | MT5 hour 18 only (UTC 16:00–17:59)                 |
| Direction model         | Forced LONG (`force_direction: 1`)                 |
| Daily filter            | LOW_VOL(p50,w90) + roll_mfe(N=20, shift=1) > 1.0  |
| Stop loss               | 0.3% of entry price (`sl_pct: 0.003`)              |
| Trailing stop mult      | 3.0× ATR (`trail_mult: 3.0`)                       |
| Breakeven mult          | 0.75× ATR (`be_atr_mult: 0.75`)                    |
| ATR mult (entry check)  | 1.5 (`atr_mult: 1.5`)                              |
| LRR minimum             | 1.0 (`lrr_min: 1.0`)                               |
| Risk per trade          | 2% of initial capital per asset                    |
| Capital per asset       | $250 (paper/shadow, not live)                      |
| Days of week            | Mon–Fri (`dow: [0,1,2,3,4]`)                       |
| Commission assumption   | 0.00 USD/lot (to be verified against broker)       |
| Spread source           | Live MT5 spread_px (dynamic, bar-level)            |
| Data source             | MT5 terminal, M2 candles, last 1800 bars           |
| Timezone                | MT5 server time = EET = UTC+2 fixed (no DST in MT5)|
| Max trades/day/asset    | 1 (single slot: hour 18 only)                      |

### Signal logic (order of operations, must match backtester exactly)

1. Check trading day allowed (dow filter)
2. Check daily regime filter: LOW_VOL(p50,w90) → allowed date set
3. Check roll_mfe gate: rolling median(mfe/mae, N=20, shift=1) over LOW_VOL sessions > 1.0
4. Fetch hour-18:00 candle close price
5. Check tick_volume ≥ volume_mean (global historical mean)
6. Compute London High/Low (MT5 hours 9–14, same day)
7. Compute LRR = (LH−LL) / mean(ATR14, London session)
8. Check LRR > 1.0
9. Generate signal: breakout above LH → LONG; breakout below LL → SHORT; large candle → counter
10. **Apply force_direction: override to LONG regardless of signal direction**
11. Compute SL = entry × (1 − 0.003), then SL = max(SL, London Low)
12. Size: lots = round(risk_pct × capital / sl_dist, step=0.1), floor at 0.1

---

## 2. Valid Operating Conditions

| Condition              | Requirement                                          |
|------------------------|------------------------------------------------------|
| MT5 connectivity       | Terminal running, symbol visible, tick data flowing  |
| Session timing         | Signal scan at MT5 18:00–18:03 only                  |
| Market state           | USTEC market open (not holiday, not weekend)         |
| Volume mean available  | Loaded from parquet or MT5 history at startup        |
| Daily filter state     | Pre-computed from historical data at startup         |
| Max open positions     | 0 open USTEC positions before scan                   |
| Max daily trades       | 0 trades already executed today for USTEC            |

### What invalidates a paper/shadow run

- Bot using `hours: [15, 18]` instead of `hours: [18]`
- `force_direction` not applied (bot takes SHORT signals)
- LOW_VOL + roll_mfe filter not running (bot trades on ALL days)
- Volume mean = 0 (filter disabled, all signals pass)
- LRR min = 0 (filter disabled)
- `execution_enabled: true` in paper environment (real orders sent)
- Capital per asset ≠ $250

---

## 3. Governance / Risk Limits

### Intra-session controls (per day)

| Rule                          | Limit        | Action                     |
|-------------------------------|--------------|----------------------------|
| Max trades per day (USTEC)    | 1            | Hard block after 1st trade |
| Max daily loss (USD)          | −$15         | Pause for day, log alert   |
| Single position open          | Yes          | Block second signal        |

### Rolling controls (weekly / multi-day)

| Rule                               | Limit     | Action                               |
|------------------------------------|-----------|--------------------------------------|
| Max weekly loss (USD)              | −$30      | Pause for remainder of week          |
| Max rolling drawdown (paper equity)| −15%      | Manual review required before resume |
| Max consecutive losses             | 5         | Manual review trigger                |
| Max consecutive no-signal days     | 10        | Verify filter is running, not broken |

### Signal volume controls

| Rule                                       | Limit        | Action                              |
|--------------------------------------------|--------------|-------------------------------------|
| Expected signals/month (approximate)       | 5–15         | Alert if 0 for 15 consecutive days  |
| Max deviation from expected signal count   | ±50% / month | Parity investigation trigger        |

### Kill switch conditions (immediate halt)

1. `execution_enabled` found True in paper environment
2. Bot sends SHORT order for USTEC (force_direction broken)
3. Bot trades on HIGH_VOL day (filter not running)
4. Any order sent to live account (wrong account alias)
5. Unhandled exception causing silent state corruption

---

## 4. Runtime Parity Checklist

Each item must be verified at paper/shadow start and after any code change.

| # | Item | Expected behavior | How to verify | Fail condition |
|---|------|-------------------|---------------|----------------|
| 1 | Signal slot | Only hour 18 evaluated | Check `signal_hours_mt5: [18]` in config | Any hour-15 scan observed in logs |
| 2 | Force direction | All USTEC entries are BUY | Check parity_log.csv `direction` column | Any `direction=-1` for USTEC |
| 3 | LOW_VOL filter | Only LOW_VOL days generate signals | Compare signal dates to backtest filter dates | Signal on HIGH_VOL day |
| 4 | roll_mfe gate | Only passes when roll_mfe > 1.0 | Compare signal dates to backtest allowed dates | Signal when roll_mfe ≤ 1.0 |
| 5 | LRR filter | LRR > 1.0 required | Check `lrr` column in parity_log.csv | Trade with lrr ≤ 1.0 |
| 6 | Volume filter | tick_volume ≥ volume_mean | Check `tv` vs `vm` in parity_log | Trade on low-volume candle |
| 7 | Lot sizing | risk=2%, floor=0.1, step=0.1 | Compare lots in parity fill vs manual calc | Lots < 0.1 or wrong rounding |
| 8 | SL placement | sl = max(ep×0.997, London Low) | Check `initial_sl` in parity fill vs signal | SL above entry or below London Low |
| 9 | Breakeven | BE triggers at 0.75×ATR above entry | Check `be_hit` + `be_level` in audit CSV | BE never triggers on winning trades |
| 10 | Trailing | Trail activates after BE, mult=3.0×ATR | Check SL movement in trade_manager logs | SL never moves on profitable trade |
| 11 | Spread source | Live MT5 spread_px used | Check `spread_signal` in parity_log | Spread = 0 or constant value |
| 12 | Timezone | Signal at MT5 18:00 = UTC 16:00 | Verify mt5_now offset in logs | Signal stamped outside UTC 16:xx |
| 13 | Capital base | $250/asset | Check `initial_capital_per_asset` in config | Any other value |
| 14 | Account type | Paper/demo account | Verify account alias in demo.yaml | Live account number in logs |
| 15 | execution_enabled | False in paper mode | Check config before start | True → real orders |

---

## 5. Pass / Fail Criteria for Paper/Shadow Phase

### Minimum observation period: 30 trading days (≈6 calendar weeks)

#### Signal generation checks (first 2 weeks)

| Check | Pass | Fail |
|---|---|---|
| Signals observed | ≥5 scan evaluations logged | 0 signals in 10 days |
| Direction | 100% BUY | Any SELL |
| Filter firing | ≥1 day blocked by LOW_VOL or roll_mfe | Filter never fires (0 blocks) |
| Parity log populated | Event per scan window | Missing entries |

#### Trade execution checks (after first trade)

| Check | Pass | Fail |
|---|---|---|
| Lot size matches manual calc | Within ±0.1 lots | > 0.1 lots deviation |
| SL placement matches expected | Within ±2 pts | > 5 pts deviation |
| Slippage at entry | ≤ 5 pts | > 10 pts consistently |
| Spread at signal | Matches MT5 live spread | Factor > 3x historical |

#### Performance checks (after 30 trading days)

| Metric | Pass | Fail / Review |
|---|---|---|
| PF (paper trades) | ≥ 1.5 | < 1.0 |
| Win rate | ≥ 45% | < 35% |
| Drawdown (paper equity) | < −15% | < −20% |
| Consecutive losses | ≤ 5 | > 7 |
| Signal count vs backtest rate | Within ±50% | < 25% or > 200% |
| Direction compliance | 100% LONG | Any SHORT |

#### Promote to demo/live-controlled when:

1. ≥30 trading days observed
2. PF ≥ 1.5 on paper equity
3. 0 parity violations (direction, filter, sizing)
4. Slippage within acceptable range
5. No kill switch events

#### Return to backtesting / halt when:

1. Any SELL trade executed (force_direction broken)
2. Drawdown exceeds −20% on paper equity
3. Signal rate < 25% of expected (filter likely broken)
4. ≥3 consecutive parity violations

---

## 6. Blocking Gaps — Must Close Before Paper Start

These are **hard blockers**. Paper/shadow cannot start until resolved.

### Gap 1 — `london_bot.yaml` USTEC config not updated (CRITICAL)

**Current:** `hours: [15, 18]`, no `force_direction`, no `low_vol_pct`, no `roll_mfe_n`
**Required:** `hours: [18]`, plus runtime support for `force_direction` and daily filter

**What to do:**
1. Update `london_bot.yaml` USTEC entry to `hours: [18]`
2. Add `force_direction: 1` key to USTEC entry
3. Add `low_vol_pct: 50`, `low_vol_win: 90`, `roll_mfe_n: 20`, `roll_mfe_min: 1.0`

### Gap 2 — Runtime signal logic missing `force_direction` and daily filter (CRITICAL)

**Current:** `signal_london.py` has no direction override, no LOW_VOL/roll_mfe computation
**Required:** Both mechanisms must mirror `backtest_runner._build_daily_filter` and the force_direction block exactly

**What to do:**
1. Add `force_direction` override to `signal_london.py` (after direction is set, same logic as backtest_runner)
2. Port `_build_daily_filter` (or equivalent) to the runtime data path in `london_levels.py`
3. Compute allowed-dates set at startup (not per-candle) and pass to `check_signal`

### Gap 3 — No `paper.yaml` environment config (MINOR, easily fixed)

**Current:** Only `demo.yaml`, `production.yaml`, `research.yaml`
**Required:** A `paper.yaml` with `execution_enabled: false`, `shadow_mode: true`, USTEC only

---

## 7. Minimum Files to Create/Change for Paper Start

| File | Change | Status |
|---|---|---|
| `config/environments/paper.yaml` | Create — execution_enabled=false, USTEC only | **TODO** |
| `config/london_bot.yaml` | Update USTEC hours, add force_direction + filter keys | **TODO** |
| `src/london_bot/signal_london.py` | Add force_direction override | **TODO** |
| `src/london_bot/london_levels.py` | Add daily filter computation (LOW_VOL + roll_mfe) | **TODO** |
| `src/backtesting/PAPER_SHADOW_SPEC_USTEC.md` | This file | **DONE** |

---

## 8. Monitoring During Paper/Shadow

### Minimum monitoring (no dashboard required)

- **parity_log.csv**: review daily — direction, lrr, spread_signal, exec_reason
- **slippage_audit.csv**: review per trade — fill vs signal price, be_hit, close_price
- **Bot logs**: check for any error/warning; confirm heartbeat every hour

### Alert conditions (manual review within 24h)

- Any `direction=-1` in parity_log for USTEC
- Any `exec_reason=execution disabled` when expected to trade
- `signal_found=False` for ≥10 consecutive days
- Any unhandled exception in bot logs

---

*Reference: `src/backtesting/BASELINE_USTEC.md` for locked parameter values*
*Reference: `src/backtesting/backtest_config.py` — OPTIONAL_PARAMS['USTEC']*
*Reference: `src/backtesting/backtest_runner.py` — `_build_daily_filter`, `force_direction` block*
