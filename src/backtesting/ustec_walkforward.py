#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USTEC 9.2 — WALK-FORWARD / TIME-STRESS TEST
=============================================
Validates temporal stability of the USTEC locked baseline via:

  A) Calendar year slices       — zero overlap, one row per year
  B) Rolling 12m / step 3m     — every consecutive 12-month stretch

No new filters. No engine changes. Audit only.

Run from: src/backtesting/
  python ustec_walkforward.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_data   import load_price_data
from backtest_runner import run_backtest
from backtest_specs  import resolve_asset_params
from backtest_stats  import calc_metrics

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(_REPO_ROOT, "data", "backtesting")
CAP        = 250.0
W          = 88

_LOCKED = dict(resolve_asset_params(DATA_DIR)['USTEC'])


# ─── Display helpers ──────────────────────────────────────────────────────────

def _hdr(lw=16):
    return (
        f"  {'window':<{lw}}  {'n':>4}  {'WR%':>5}  {'PF':>5}  "
        f"{'ret%':>8}  {'MDD%':>7}  {'Sharpe**':>8}  "
        f"{'avg_win':>7}  {'avg_loss':>8}  {'exp/trd':>8}"
    )


def _row(m, label, lw=16):
    if not m or m.get('n', 0) == 0:
        return f"  {label:<{lw}}  — no trades"
    sh = f"{m['sharpe']:>8.3f}" if pd.notna(m.get('sharpe', float('nan'))) else "       —"
    return (
        f"  {label:<{lw}}  {m['n']:>4}  {m['wr']:>4.1f}%  {m['pf']:>5.2f}  "
        f"{m['ret']:>+7.2f}%  {m['mdd']:>7.2f}%  {sh}  "
        f"{m['aw']:>+7.2f}  {m['al']:>+7.2f}  {m['exp']:>+8.4f}"
    )


def _title(t):
    print(f"\n{'═'*W}")
    print(f"  {t}")
    print(f"{'═'*W}")


def _sub(t):
    pad = W - 6 - len(t)
    print(f"\n  ── {t} {'─'*max(pad,2)}")


def _sep():
    print(f"  {'─'*84}")


# ─── Window builders ──────────────────────────────────────────────────────────

def year_windows(df):
    """One window per calendar year present in data."""
    years = sorted(df['time'].dt.year.unique())
    data_start = df['time'].min().normalize()
    data_end = df['time'].max().normalize()
    windows = []
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        end   = pd.Timestamp(f"{y+1}-01-01")
        is_partial = (
            (y == data_start.year and (data_start.month != 1 or data_start.day != 1)) or
            (y == data_end.year and (data_end.month != 12 or data_end.day != 31))
        )
        label = f"{y}{' [PARTIAL]' if is_partial else ''}"
        windows.append((label, start, end, is_partial))
    return windows


def rolling_windows(df, months_len=12, months_step=3):
    """Rolling windows of fixed length, stepping forward."""
    data_start = df['time'].min().normalize()
    data_end = df['time'].max().normalize()
    t_min = df['time'].min().to_period('M').to_timestamp()
    t_max = df['time'].max().to_period('M').to_timestamp() + pd.offsets.MonthEnd(1)
    windows = []
    start = t_min
    while True:
        end = start + pd.DateOffset(months=months_len)
        if end > t_max + pd.DateOffset(months=months_step):
            break
        requested_end = end - pd.Timedelta(days=1)
        is_partial = start < data_start or requested_end > data_end
        label = f"{start.strftime('%Y-%m')}→{requested_end.strftime('%Y-%m')}{' [PARTIAL]' if is_partial else ''}"
        windows.append((label, start, end, is_partial))
        start += pd.DateOffset(months=months_step)
    return windows


# ─── Summary stats ────────────────────────────────────────────────────────────

def _summary(metrics_list, labels):
    valid = [(lb, m) for lb, m in zip(labels, metrics_list) if m and m.get('n', 0) > 0]
    if not valid:
        print("  No data.")
        return

    n_pos    = sum(1 for _, m in valid if m['ret'] > 0)
    n_pf1    = sum(1 for _, m in valid if m['pf']  > 1.0)
    n_pf15   = sum(1 for _, m in valid if m['pf']  > 1.5)
    n_pf2    = sum(1 for _, m in valid if m['pf']  > 2.0)

    best_lb  = max(valid, key=lambda x: x[1]['pf'])
    worst_lb = min(valid, key=lambda x: x[1]['pf'])
    max_mdd  = min(valid, key=lambda x: x[1]['mdd'])

    pf_vals  = [m['pf']  for _, m in valid]
    ret_vals = [m['ret'] for _, m in valid]

    print(f"\n  Windows with data        : {len(valid)}")
    print(f"  Positive return (ret>0)  : {n_pos}/{len(valid)}")
    print(f"  PF > 1.0                 : {n_pf1}/{len(valid)}")
    print(f"  PF > 1.5                 : {n_pf15}/{len(valid)}")
    print(f"  PF > 2.0                 : {n_pf2}/{len(valid)}")
    print(f"  Median PF                : {np.median(pf_vals):.2f}")
    print(f"  Mean ret%                : {np.mean(ret_vals):+.2f}%")
    print(f"  Best window   (by PF)    : {best_lb[0]}  PF={best_lb[1]['pf']:.2f}  ret={best_lb[1]['ret']:+.2f}%")
    print(f"  Worst window  (by PF)    : {worst_lb[0]}  PF={worst_lb[1]['pf']:.2f}  ret={worst_lb[1]['ret']:+.2f}%")
    print(f"  Max drawdown window      : {max_mdd[0]}  MDD={max_mdd[1]['mdd']:.2f}%")

    # Stability grade
    pf_pass_rate = n_pf15 / len(valid)
    pos_rate     = n_pos  / len(valid)
    if pf_pass_rate >= 0.75 and pos_rate >= 0.80:
        grade = "STRONG   — consistent across time"
    elif pf_pass_rate >= 0.50 and pos_rate >= 0.65:
        grade = "MODERATE — some weak windows, not catastrophic"
    elif pf_pass_rate >= 0.33:
        grade = "MARGINAL — significant temporal instability"
    else:
        grade = "WEAK     — edge not stable across time"
    print(f"\n  Stability grade          : {grade}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("╔" + "═"*(W-2) + "╗")
    print("║   USTEC 9.2 — WALK-FORWARD / TIME-STRESS TEST" + " "*(W-48) + "║")
    print("╚" + "═"*(W-2) + "╝")

    df, lr, vm, qc = load_price_data('USTEC', DATA_DIR, _LOCKED['digits'])
    if df is None:
        print("  ERROR: USTEC data not found.")
        sys.exit(1)
    print(f"\n  Data : {df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()} | {len(df):,} bars")
    print(f"  Cap  : ${CAP:.0f}  |  Locked baseline: hours=[18], force=LONG, LOW_VOL+roll_mfe")

    def run_window(start, end):
        t, e = run_backtest('USTEC', df, lr, vm, _LOCKED, CAP, start, end, 'WF')
        return calc_metrics(t, e, CAP, sample_start=t.attrs.get('sample_start'), sample_end=t.attrs.get('sample_end'))

    # ═════════════════════════════════════════════════════════════════════════
    # A) CALENDAR YEAR SLICES
    # ═════════════════════════════════════════════════════════════════════════
    _title("A) CALENDAR YEAR SLICES")
    print("  Each calendar year tested independently. Zero overlap.")

    wins_yr = year_windows(df)
    print(f"\n  Years detected: {[w[0] for w in wins_yr]}")
    if any(w[3] for w in wins_yr):
        print("  Note: partial calendar years are shown for traceability and excluded from summary/stability scoring.")

    _sub("Results by year")
    print(_hdr(lw=6))
    _sep()

    yr_metrics = []
    yr_labels  = []
    for label, start, end, is_partial in wins_yr:
        m = run_window(start, end)
        yr_metrics.append(m)
        yr_labels.append(label)
        print(_row(m, label, lw=6))

    _sub("Summary — year slices")
    _summary([m for lb, m in zip(yr_labels, yr_metrics) if '[PARTIAL]' not in lb], [lb for lb in yr_labels if '[PARTIAL]' not in lb])

    # ═════════════════════════════════════════════════════════════════════════
    # B) ROLLING 12m / STEP 3m
    # ═════════════════════════════════════════════════════════════════════════
    _title("B) ROLLING 12-MONTH WINDOWS  (step = 3 months)")
    print("  Every consecutive 12-month stretch. Overlapping — not independent.")
    print("  Purpose: detect whether edge holds over any arbitrary 12-month period,")
    print("           not just calendar years.")

    wins_roll = rolling_windows(df, months_len=12, months_step=3)
    print(f"\n  Windows: {len(wins_roll)}")
    if any(w[3] for w in wins_roll):
        print("  Note: partial rolling windows are shown for traceability and excluded from summary/stability scoring.")

    _sub("Results — rolling windows")
    print(_hdr(lw=20))
    _sep()

    roll_metrics = []
    roll_labels  = []
    for label, start, end, is_partial in wins_roll:
        m = run_window(start, end)
        roll_metrics.append(m)
        roll_labels.append(label)
        print(_row(m, label, lw=20))

    _sub("Summary — rolling windows")
    _summary([m for lb, m in zip(roll_labels, roll_metrics) if '[PARTIAL]' not in lb], [lb for lb in roll_labels if '[PARTIAL]' not in lb])

    # ═════════════════════════════════════════════════════════════════════════
    # STABILITY VERDICT
    # ═════════════════════════════════════════════════════════════════════════
    _title("TEMPORAL STABILITY VERDICT")

    valid_yr   = [(lb, m) for lb, m in zip(yr_labels,   yr_metrics)   if '[PARTIAL]' not in lb and m and m.get('n', 0) > 0]
    valid_roll = [(lb, m) for lb, m in zip(roll_labels, roll_metrics) if '[PARTIAL]' not in lb and m and m.get('n', 0) > 0]

    if valid_yr:
        pf_yr  = [m['pf'] for _, m in valid_yr]
        min_pf = min(pf_yr)
        worst_yr = min(valid_yr, key=lambda x: x[1]['pf'])
        print(f"\n  Worst calendar year : {worst_yr[0]}  PF={worst_yr[1]['pf']:.2f}"
              f"  ret={worst_yr[1]['ret']:+.2f}%  n={worst_yr[1]['n']}")

    if valid_roll:
        pf_roll = [m['pf'] for _, m in valid_roll]
        worst_r = min(valid_roll, key=lambda x: x[1]['pf'])
        n_neg   = sum(1 for _, m in valid_roll if m['pf'] < 1.0)
        print(f"  Worst rolling 12m   : {worst_r[0]}  PF={worst_r[1]['pf']:.2f}"
              f"  ret={worst_r[1]['ret']:+.2f}%")
        print(f"  Rolling windows PF<1: {n_neg}/{len(valid_roll)}")

    print(f"""
  Interpretation guide:
    PF > 2.0 in all calendar years   → very strong temporal stability
    PF > 1.5 in ≥75% rolling windows → strong; acceptable weak patches
    Any year PF < 1.0                → structural weakness in that period
    >25% rolling windows PF < 1.0   → edge is not temporally stable

  ** Sharpe: monthly PnL / initial_cap, annualized sqrt(12). Not equity-weighted.
""")
    print(f"{'═'*W}\n")


if __name__ == '__main__':
    main()


