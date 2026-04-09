#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USTEC Institutional Comparison
================================
Compares 3 variants of the USTEC operational setup using the real backtest engine.

Variants
--------
  BASE              : no daily filter, no forced direction, hours=[15,18]
  REFINED filter    : LOW_VOL(p50,w90) + roll_mfe(N=20)>1.0, hours=[15,18], bidirectional
  OPERATIONAL       : LOW_VOL(p50,w90) + roll_mfe(N=20)>1.0, hours=[18], forced LONG

Run from: src/backtesting/
  python ustec_comparison.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_config import GLOBAL_RISK_PCT, TEST_START
from backtest_data   import load_price_data
from backtest_runner import run_backtest
from backtest_specs  import resolve_asset_params
from backtest_stats  import calc_metrics

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "backtesting")
CAP      = 250.0

# ─── Shared base params ───────────────────────────────────────────────────────
_BASE = {}

# ─── 3 variants ───────────────────────────────────────────────────────────────
VARIANTS = [
    (
        'BASE',
        'No filter, hours=[15,18], bidirectional',
        {**_BASE, 'hours': [15, 18]},
    ),
    (
        'REFINED filter only',
        'LOW_VOL+roll_mfe, hours=[15,18], bidirectional',
        {**_BASE, 'hours': [15, 18],
         'low_vol_pct': 50, 'low_vol_win': 90,
         'roll_mfe_n': 20, 'roll_mfe_min': 1.0},
    ),
    (
        'OPERATIONAL',
        'LOW_VOL+roll_mfe, hours=[18], forced LONG',
        {**_BASE, 'hours': [18], 'force_direction': 1,
         'low_vol_pct': 50, 'low_vol_win': 90,
         'roll_mfe_n': 20, 'roll_mfe_min': 1.0},
    ),
]


def _row(m):
    """Format a calc_metrics dict into a display tuple."""
    if not m:
        return None
    return (
        m['n'], m['wr'], m['pf'],
        m['aw'], m['al'], m['exp'],
        m['ret'], m['mdd'],
    )


def _print_table(rows, period):
    hdr = (
        f"\n  {'variant':<22}  {'n':>5}  {'WR%':>5}  {'PF':>5}  "
        f"{'avg_win':>7}  {'avg_loss':>8}  {'exp':>7}  {'ret%':>7}  {'MDD%':>7}"
    )
    sep = f"  {'-'*82}"
    print(f"\n{'═'*86}")
    print(f"  {period}")
    print(f"{'═'*86}")
    print(hdr)
    print(sep)
    for name, r in rows:
        if r is None:
            print(f"  {name:<22}  — no trades")
            continue
        n, wr, pf, aw, al, exp, ret, mdd = r
        print(
            f"  {name:<22}  {n:>5}  {wr:>4.1f}%  {pf:>5.2f}  "
            f"{aw:>+7.2f}  {al:>+8.2f}  {exp:>+7.4f}  "
            f"{ret:>+6.2f}%  {mdd:>7.2f}%"
        )


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   USTEC — INSTITUTIONAL COMPARISON                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  Variants:")
    for name, desc, _ in VARIANTS:
        print(f"    [{name}]  {desc}")
    print()

    df, lr, vm, qc = load_price_data('USTEC', DATA_DIR, digits=2)
    if df is None:
        print("ERROR: USTEC data not found in", DATA_DIR)
        sys.exit(1)

    print(
        f"  Data: {df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()} "
        f"| {len(df):,} bars\n"
    )

    base = dict(resolve_asset_params(DATA_DIR)['USTEC'])
    base['risk_pct'] = GLOBAL_RISK_PCT
    variants = [
        (
            'BASE',
            'No filter, hours=[15,18], bidirectional',
            {**base, 'hours': [15, 18], 'force_direction': None,
             'low_vol_pct': None, 'low_vol_win': None, 'roll_mfe_n': None, 'roll_mfe_min': None},
        ),
        (
            'REFINED filter only',
            'LOW_VOL+roll_mfe, hours=[15,18], bidirectional',
            {**base, 'hours': [15, 18], 'force_direction': None,
             'low_vol_pct': 50, 'low_vol_win': 90,
             'roll_mfe_n': 20, 'roll_mfe_min': 1.0},
        ),
        (
            'OPERATIONAL',
            'LOW_VOL+roll_mfe, hours=[18], forced LONG',
            {**base, 'hours': [18], 'force_direction': 1,
             'low_vol_pct': 50, 'low_vol_win': 90,
             'roll_mfe_n': 20, 'roll_mfe_min': 1.0},
        ),
    ]

    full_rows = []
    test_rows = []

    for name, _, p in variants:
        t_f, e_f = run_backtest('USTEC', df, lr, vm, p, CAP, None,        None,       'ALL')
        t_t, e_t = run_backtest('USTEC', df, lr, vm, p, CAP, TEST_START,  None,       'TEST')
        m_f = calc_metrics(t_f, e_f, CAP, sample_start=t_f.attrs.get('sample_start'), sample_end=t_f.attrs.get('sample_end'))
        m_t = calc_metrics(t_t, e_t, CAP, sample_start=t_t.attrs.get('sample_start'), sample_end=t_t.attrs.get('sample_end'))
        full_rows.append((name, _row(m_f)))
        test_rows.append((name, _row(m_t)))

    _print_table(full_rows, 'ALL   (all available history)')
    _print_table(test_rows, 'TEST  (2025-01-01 → present)')

    # ── Delta analysis ─────────────────────────────────────────────────────────
    print(f"\n{'═'*86}")
    print("  DELTA ANALYSIS  (vs BASE)")
    print(f"{'═'*86}")
    print(f"  {'period':<6}  {'variant':<22}  {'Δn':>6}  {'ΔPF':>6}  {'Δret%':>7}  {'ΔMDD%':>7}")
    print(f"  {'-'*60}")

    for period, rows in [('FULL', full_rows), ('TEST', test_rows)]:
        base = rows[0][1]
        if base is None:
            continue
        b_n, _, b_pf, _, _, _, b_ret, b_mdd = base
        for name, r in rows[1:]:
            if r is None:
                print(f"  {period:<6}  {name:<22}  — no trades")
                continue
            n, _, pf, _, _, _, ret, mdd = r
            print(
                f"  {period:<6}  {name:<22}  "
                f"{n-b_n:>+6}  {pf-b_pf:>+6.2f}  "
                f"{ret-b_ret:>+6.2f}%  {mdd-b_mdd:>+7.2f}%"
            )
        print()

    print("  Done.")


if __name__ == '__main__':
    main()

