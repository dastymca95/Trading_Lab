#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Candidate Selection / Asset Pruning
===============================================
Compares portfolio combinations using the current operational params.
Each asset is backtested once; results are aggregated per combination.

Combinations evaluated
----------------------
  1. USTEC only
  2. USTEC + US500
  3. USTEC + US30
  4. USTEC + DE40
  5. All 4

Run from: src/backtesting/
  python portfolio_pruning.py
"""

import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _THIS_DIR)

from backtest_config import (
    GLOBAL_RISK_PCT, INITIAL_PER_ASSET, TEST_START,
    ASSET_PARAMS_BASE, OPTIONAL_PARAMS, OPTIONAL_ASSETS,
)
from backtest_data   import load_price_data
from backtest_runner import run_backtest
from backtest_stats  import calc_metrics

DATA_DIR = os.path.join(_REPO_ROOT, "data", "backtesting")
CAP_PER  = INITIAL_PER_ASSET

# ─── Operational params (mirrors backtest_runner without live MT5 call) ───────
# Commissions for US-index assets are 0 — live spec makes no difference here.
_PARAMS = {}
for k, v in ASSET_PARAMS_BASE.items():
    _PARAMS[k] = {**v, 'risk_pct': GLOBAL_RISK_PCT}
for k in OPTIONAL_ASSETS:
    if k in OPTIONAL_PARAMS:
        _PARAMS[k] = {**OPTIONAL_PARAMS[k], 'risk_pct': GLOBAL_RISK_PCT}

ASSETS_TO_TEST = [a for a in ['USTEC', 'US500', 'US30', 'DE40'] if a in _PARAMS]

COMBINATIONS = [
    ('USTEC only',    ['USTEC']),
    ('USTEC + US500', ['USTEC', 'US500']),
    ('USTEC + US30',  ['USTEC', 'US30']),
    ('USTEC + DE40',  ['USTEC', 'DE40']),
    ('All 4',         ['USTEC', 'US500', 'US30', 'DE40']),
]


# ─── Portfolio aggregation helper ─────────────────────────────────────────────

def _agg_metrics(trade_dfs, cap0):
    """Combine trade DataFrames into portfolio equity + metrics."""
    dfs = [d for d in trade_dfs if d is not None and len(d) > 0]
    if not dfs:
        return {}
    all_t = pd.concat(dfs).sort_values('Fecha Apertura').reset_index(drop=True)
    cap = cap0
    eq  = [cap]
    for pnl in all_t['PnL Neto USD'].values:
        cap = max(cap + pnl, 0.01)
        eq.append(cap)
    return calc_metrics(all_t, np.array(eq), cap0)


# ─── Display helpers ──────────────────────────────────────────────────────────

def _mrow(m, label, width=22):
    if not m:
        return f"  {label:<{width}}  — no trades"
    sh = f"{m['sharpe']:>6.3f}" if pd.notna(m.get('sharpe', float('nan'))) else "     —"
    return (
        f"  {label:<{width}}  {m['n']:>5}  {m['wr']:>5.1f}%  {m['pf']:>5.2f}  "
        f"{m['ret']:>+7.2f}%  {m['mdd']:>7.2f}%  "
        f"{m['aw']:>+7.2f}  {m['al']:>+7.2f}  {m['exp']:>+8.4f}  {sh}"
    )


def _hdr(width=22):
    return (
        f"  {'combination':<{width}}  {'n':>5}  {'WR%':>6}  {'PF':>5}  "
        f"{'ret%':>8}  {'MDD%':>7}  "
        f"{'avg_win':>7}  {'avg_loss':>8}  {'exp/trd':>8}  {'Sharpe':>6}"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    W = 80
    print("╔" + "═"*(W-2) + "╗")
    print("║   PORTFOLIO CANDIDATE SELECTION — ASSET PRUNING" + " "*(W-50) + "║")
    print("╚" + "═"*(W-2) + "╝")
    print(f"\n  Data dir   : {DATA_DIR}")
    print(f"  Risk/asset : {GLOBAL_RISK_PCT:.2%}  |  Cap/asset : ${CAP_PER:.0f}")
    print(f"  Test from  : {TEST_START.date()}")
    print(f"\n  Combinations: {len(COMBINATIONS)}")
    for name, assets in COMBINATIONS:
        print(f"    {name:<16}  {assets}")

    # ── Step 1: load price data ───────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  DATA LOADING")
    print(f"{'─'*W}")

    asset_data = {}
    for asset in ASSETS_TO_TEST:
        p = _PARAMS[asset]
        df, lr, vm, qc = load_price_data(asset, DATA_DIR, p['digits'])
        if df is None:
            print(f"  ⚠  {asset}: datos no encontrados — excluido")
        else:
            asset_data[asset] = (df, lr, vm)

    if not asset_data:
        print("  ❌ Sin datos.")
        return

    # ── Step 2: run backtest once per asset ───────────────────────────────────
    print(f"\n{'─'*W}")
    print("  RUNNING BACKTESTS")
    print(f"{'─'*W}")

    trades_full = {}   # asset → DataFrame (all history)
    trades_test = {}   # asset → DataFrame (TEST period)

    for asset, (df, lr, vm) in asset_data.items():
        p = _PARAMS[asset]
        print(f"  {asset}...", end=' ', flush=True)
        t_f, _ = run_backtest(asset, df, lr, vm, p, CAP_PER, None,       None,  'FULL')
        t_t, _ = run_backtest(asset, df, lr, vm, p, CAP_PER, TEST_START, None,  'TEST')
        trades_full[asset] = t_f if len(t_f) > 0 else None
        trades_test[asset] = t_t if len(t_t) > 0 else None
        n_f = len(t_f); n_t = len(t_t)
        print(f"FULL n={n_f}  TEST n={n_t}")

    # ── Step 3: aggregate per combination ────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  RESULTS — FULL HISTORY")
    print(f"{'─'*W}")
    print(_hdr())
    print(f"  {'─'*76}")

    full_results = {}
    for name, assets in COMBINATIONS:
        available = [a for a in assets if a in asset_data]
        n_assets  = len(available)
        cap0      = CAP_PER * n_assets
        dfs       = [trades_full.get(a) for a in available]
        m         = _agg_metrics(dfs, cap0)
        full_results[name] = m
        print(_mrow(m, name))

    print(f"\n{'─'*W}")
    print("  RESULTS — TEST PERIOD")
    print(f"{'─'*W}")
    print(_hdr())
    print(f"  {'─'*76}")

    test_results = {}
    for name, assets in COMBINATIONS:
        available = [a for a in assets if a in asset_data]
        n_assets  = len(available)
        cap0      = CAP_PER * n_assets
        dfs       = [trades_test.get(a) for a in available]
        m         = _agg_metrics(dfs, cap0)
        test_results[name] = m
        print(_mrow(m, name))

    # ── Step 4: delta vs USTEC solo ───────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  DELTA vs 'USTEC only'  (positive = improvement)")
    print(f"{'─'*W}")
    print(f"  {'combination':<22}  {'per':<5}  {'Δn':>6}  {'ΔPF':>6}  {'Δret%':>8}  {'ΔMDD%':>7}  {'ΔSharpe':>8}")
    print(f"  {'─'*68}")

    base_name = 'USTEC only'
    for period, results in [('FULL', full_results), ('TEST', test_results)]:
        base = results.get(base_name, {})
        if not base:
            continue
        for name, _ in COMBINATIONS[1:]:   # skip USTEC only itself
            m = results.get(name, {})
            if not m:
                print(f"  {name:<22}  {period:<5}  — no trades")
                continue
            dn     = m['n']   - base['n']
            dpf    = m['pf']  - base['pf']
            dret   = m['ret'] - base['ret']
            dmdd   = m['mdd'] - base['mdd']
            bsh    = base.get('sharpe', float('nan'))
            msh    = m.get('sharpe',   float('nan'))
            dsh    = (msh - bsh) if (pd.notna(msh) and pd.notna(bsh)) else float('nan')
            dsh_s  = f"{dsh:>+8.3f}" if pd.notna(dsh) else "       —"
            print(
                f"  {name:<22}  {period:<5}  {dn:>+6}  {dpf:>+6.2f}  "
                f"{dret:>+7.2f}%  {dmdd:>+7.2f}%  {dsh_s}"
            )
        print()

    print(f"{'═'*W}")
    print("  Done.")
    print(f"{'═'*W}\n")


if __name__ == '__main__':
    main()
