#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USTEC PRE-PRODUCTION ROBUSTNESS — COST & RISK SENSITIVITY
==========================================================
Phase 9.1 — tests USTEC locked baseline against:
  A) realistic cost scenarios (spread / commission)
  B) risk % sweep (0.5% → 6.0%)

No new filters. No engine changes. Audit only.

Run from: src/backtesting/
  python ustec_robustness.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_config import TEST_START
from backtest_data   import load_price_data
from backtest_runner import run_backtest
from backtest_specs  import resolve_asset_params
from backtest_stats  import calc_metrics

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(_REPO_ROOT, "data", "backtesting")
CAP        = 250.0
W          = 86

# ─── USTEC locked baseline ────────────────────────────────────────────────────
# risk_pct / sp / comm are swept below; all other keys stay locked.
_LOCKED = dict(resolve_asset_params(DATA_DIR)['USTEC'])

# ─── Cost scenarios ───────────────────────────────────────────────────────────
# stress_spread_mult applies to real spread_px from the MT5 parquet.
# comm is always applied as lots * comm regardless of spread source.
# Each scenario: (label, stress_spread_mult, comm)
COST_SCENARIOS = [
    ('BASE          x1.0, c=0.00',  1.0,  0.00),
    ('2x spread     x2.0, c=0.00',  2.0,  0.00),
    ('3x spread     x3.0, c=0.00',  3.0,  0.00),
    ('+commission   x1.0, c=2.00',  1.0,  2.00),
    ('Conserv.      x2.0, c=2.00',  2.0,  2.00),
    ('Adversarial   x3.0, c=5.00',  3.0,  5.00),
]

# ─── Risk % sweep ─────────────────────────────────────────────────────────────
RISK_LEVELS = [r / 200.0 for r in range(1, 13)]   # 0.5%, 1.0%, ..., 6.0%


# ─── Display helpers ──────────────────────────────────────────────────────────

def _hdr(lw=28):
    return (
        f"  {'scenario':<{lw}}  {'n':>5}  {'WR%':>6}  {'PF':>5}  "
        f"{'ret%':>9}  {'MDD%':>7}  {'Sharpe**':>8}  "
        f"{'avg_win':>7}  {'avg_loss':>8}  {'exp/trd':>8}"
    )


def _row(m, label, lw=28):
    if not m:
        return f"  {label:<{lw}}  — no trades"
    sh = f"{m['sharpe']:>8.3f}" if pd.notna(m.get('sharpe', float('nan'))) else "       —"
    return (
        f"  {label:<{lw}}  {m['n']:>5}  {m['wr']:>5.1f}%  {m['pf']:>5.2f}  "
        f"{m['ret']:>+8.2f}%  {m['mdd']:>7.2f}%  {sh}  "
        f"{m['aw']:>+7.2f}  {m['al']:>+7.2f}  {m['exp']:>+8.4f}"
    )


def _title(t):
    print(f"\n{'═'*W}")
    print(f"  {t}")
    print(f"{'═'*W}")


def _sub(t):
    print(f"\n  ── {t} {'─'*(W-6-len(t))}")


def _sep():
    print(f"  {'─'*82}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("╔" + "═"*(W-2) + "╗")
    print("║   USTEC — PRE-PRODUCTION ROBUSTNESS: COST & RISK SENSITIVITY" + " "*(W-63) + "║")
    print("╚" + "═"*(W-2) + "╝")

    # ── Load data once ────────────────────────────────────────────────────────
    df, lr, vm, qc = load_price_data('USTEC', DATA_DIR, _LOCKED['digits'])
    if df is None:
        print("  ERROR: USTEC data not found.")
        sys.exit(1)
    print(f"\n  Data: {df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()} | {len(df):,} bars")
    print(f"  Cap : ${CAP:.0f}  |  ml=0.1 lots  |  step=0.1  |  SL~0.3% price")

    def run(p, period_start):
        t, e = run_backtest('USTEC', df, lr, vm, p, CAP,
                            period_start, None,
                            'TEST' if period_start else 'FULL')
        return calc_metrics(t, e, CAP)

    # ═════════════════════════════════════════════════════════════════════════
    # A) COST SENSITIVITY
    # ═════════════════════════════════════════════════════════════════════════
    _title("A) COST SENSITIVITY")
    print(f"  spread_cost per trade = spread_px(MT5) × stress_mult × lots.")
    print(f"  Real MT5 spread_px is used — stress_spread_mult widens it proportionally.")
    print(f"  comm is always applied as lots × comm (separate from spread).")

    for period_label, period_start in [
        ('FULL HISTORY',              None),
        ('TEST (2025-01-01→present)', TEST_START),
    ]:
        _sub(period_label)
        print(_hdr())
        _sep()
        for label, mult, comm in COST_SCENARIOS:
            p = {**_LOCKED, 'stress_spread_mult': mult, 'comm': comm}
            print(_row(run(p, period_start), label))

    # Delta vs BASE — FULL only
    _sub("DELTA vs BASE  (FULL)")
    print(f"  {'scenario':<28}  {'Δn':>4}  {'ΔPF':>6}  {'Δret%':>8}  {'ΔMDD%':>7}  {'ΔSharpe':>8}  {'Δexp':>8}")
    _sep()
    bm = run({**_LOCKED, 'stress_spread_mult': COST_SCENARIOS[0][1], 'comm': COST_SCENARIOS[0][2]}, None)
    for label, mult, comm in COST_SCENARIOS[1:]:
        m = run({**_LOCKED, 'stress_spread_mult': mult, 'comm': comm}, None)
        if not m:
            print(f"  {label:<28}  — no trades"); continue
        dsh = (
            m['sharpe'] - bm['sharpe']
            if pd.notna(m.get('sharpe')) and pd.notna(bm.get('sharpe'))
            else float('nan')
        )
        dsh_s = f"{dsh:>+8.3f}" if pd.notna(dsh) else "       —"
        print(
            f"  {label:<28}  {m['n']-bm['n']:>+4}  {m['pf']-bm['pf']:>+6.2f}  "
            f"{m['ret']-bm['ret']:>+7.2f}%  {m['mdd']-bm['mdd']:>+7.2f}%  "
            f"{dsh_s}  {m['exp']-bm['exp']:>+8.4f}"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # B) RISK % SENSITIVITY
    # ═════════════════════════════════════════════════════════════════════════
    _title("B) RISK % SENSITIVITY")
    print(f"  Base cost: sp=1.0, comm=0.00 (locked baseline)")

    # Lot floor table
    SL_APPROX = 54.0   # 0.3% × 18000 avg USTEC price
    print(f"\n  Starting lot size at cap=${CAP:.0f}, SL≈{SL_APPROX:.0f}pts:")
    print(f"  {'risk%':>6}  {'req_lots':>9}  {'actual':>8}  {'floor?':>7}")
    print(f"  {'─'*35}")
    for rp in RISK_LEVELS:
        req = rp * CAP / SL_APPROX
        actual = max(0.1, round(round(req / 0.1) * 0.1, 1))
        print(f"  {rp*100:>5.1f}%  {req:>9.4f}  {actual:>8.1f}  {'<-- floor' if req < 0.1 else '':>7}")

    for period_label, period_start in [
        ('FULL HISTORY',              None),
        ('TEST (2025-01-01→present)', TEST_START),
    ]:
        _sub(period_label)
        print(_hdr(lw=10))
        _sep()
        for rp in RISK_LEVELS:
            p = {**_LOCKED, 'risk_pct': rp}
            m = run(p, period_start)
            print(_row(m, f"risk={rp*100:.1f}%", lw=10))

    # ═════════════════════════════════════════════════════════════════════════
    # VERDICT
    # ═════════════════════════════════════════════════════════════════════════
    _title("VERDICT")
    print("""
  COST ROBUSTNESS
    Interpret the delta table:
      - USTEC's edge is directional (forced LONG + regime filter), not
        spread-marginal. Expect PF to degrade slowly with higher spread.
      - Green threshold: PF stays > 2.0 under 2x spread.
      - Yellow threshold: PF stays > 1.5 under adversarial (3x sp + c=5).
      - If PF < 1.5 under adversarial, edge is still present but thin.

  RISK SENSITIVITY
    Lot floor insight:
      - Below ~4% risk at $250 cap: min lot = 0.1 dominates.
        PF / WR / exp per trade are identical in this band.
        Only ret% / MDD% differ (proportional to lots × compounding).
      - Above ~4%: lot steps up, compounding amplifies both sides.
      - MDD% grows faster than ret% at high risk due to compounding.

    Institutional range:
      - Identify the row where MDD% first crosses -15% (FULL) or -20% (TEST).
      - Identify the row where Sharpe starts to degrade.
      - Recommended operating range: risk where MDD < 15% and PF > 2.0.
      - Current locked value (2%) is almost certainly inside the safe band.

  NEXT SUBFASE
    9.2 — Walk-forward / time-stress test.
""")
    print(f"  ** Sharpe: monthly PnL / initial_cap, annualized sqrt(12). Not equity-weighted.")
    print(f"{'═'*W}\n")


if __name__ == '__main__':
    main()
