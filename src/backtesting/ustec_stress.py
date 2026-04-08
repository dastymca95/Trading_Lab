#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USTEC 9.3 — HARDER STRESS / DEGRADATION ANALYSIS
==================================================
Subjects the USTEC locked baseline to progressively harder stress:

  Dimensions:
    stress_spread_mult  — widens real MT5 spread_px proportionally
    comm                — round-trip commission per lot (USD)
    entry_slippage_pts  — worse fill: ep += slippage * direction (LONG = higher entry)

  Scenarios (7):
    BASE         x1.0  c=0.00  slip=0     locked baseline, no stress
    Hard spread  x5.0  c=0.00  slip=0     pure spread stress
    Hard cost    x3.0  c=5.00  slip=0     realistic high-friction broker
    Slip 2pt     x1.0  c=0.00  slip=2     mild execution degradation
    Slip 5pt     x1.0  c=0.00  slip=5     hard execution degradation (~9% of SL)
    Combined     x3.0  c=5.00  slip=2     cost + execution together
    Extreme      x5.0  c=10.0  slip=5     institutional stress limit

  Context: SL ≈ 54 pts (0.3% × ~18000). 1pt slippage ≈ 1.9% of SL width.

No new filters. No engine changes beyond entry_slippage_pts hook. Audit only.

Run from: src/backtesting/
  python ustec_stress.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_config import TEST_START, OPTIONAL_PARAMS
from backtest_data   import load_price_data
from backtest_runner import run_backtest
from backtest_stats  import calc_metrics

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(_REPO_ROOT, "data", "backtesting")
CAP        = 250.0
W          = 88

_LOCKED = dict(OPTIONAL_PARAMS['USTEC'])

# ─── Stress scenarios ─────────────────────────────────────────────────────────
# (label, stress_spread_mult, comm, entry_slippage_pts)
SCENARIOS = [
    ('BASE',          1.0,  0.00, 0.0),
    ('Hard spread x5',5.0,  0.00, 0.0),
    ('Hard cost',     3.0,  5.00, 0.0),
    ('Slip 2pt',      1.0,  0.00, 2.0),
    ('Slip 5pt',      1.0,  0.00, 5.0),
    ('Combined',      3.0,  5.00, 2.0),
    ('Extreme',       5.0, 10.00, 5.0),
]

# ─── Verdict thresholds ───────────────────────────────────────────────────────
PF_SURVIVES  = 2.0   # PF ≥ 2.0 → SURVIVES
PF_DEGRADES  = 1.3   # 1.3 ≤ PF < 2.0 → DEGRADES BUT ACCEPTABLE
               # PF < 1.3 → BREAKS


# ─── Display helpers ──────────────────────────────────────────────────────────

def _hdr(lw=18):
    return (
        f"  {'scenario':<{lw}}  {'n':>4}  {'WR%':>5}  {'PF':>5}  "
        f"{'ret%':>8}  {'MDD%':>7}  {'Sharpe**':>8}  "
        f"{'avg_win':>7}  {'avg_loss':>8}  {'exp/trd':>8}"
    )


def _row(m, label, lw=18):
    if not m or m.get('n', 0) == 0:
        return f"  {label:<{lw}}  — no trades"
    sh = f"{m['sharpe']:>8.3f}" if pd.notna(m.get('sharpe', float('nan'))) else "       —"
    return (
        f"  {label:<{lw}}  {m['n']:>4}  {m['wr']:>4.1f}%  {m['pf']:>5.2f}  "
        f"{m['ret']:>+7.2f}%  {m['mdd']:>7.2f}%  {sh}  "
        f"{m['aw']:>+7.2f}  {m['al']:>+7.2f}  {m['exp']:>+8.4f}"
    )


def _verdict(pf):
    if pf >= PF_SURVIVES:  return 'SURVIVES'
    if pf >= PF_DEGRADES:  return 'DEGRADES (acceptable)'
    return 'BREAKS'


def _title(t):
    print(f"\n{'═'*W}")
    print(f"  {t}")
    print(f"{'═'*W}")


def _sub(t):
    pad = W - 6 - len(t)
    print(f"\n  ── {t} {'─'*max(pad, 2)}")


def _sep():
    print(f"  {'─'*84}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("╔" + "═"*(W-2) + "╗")
    print("║   USTEC 9.3 — HARDER STRESS / DEGRADATION ANALYSIS" + " "*(W-54) + "║")
    print("╚" + "═"*(W-2) + "╝")

    df, lr, vm, qc = load_price_data('USTEC', DATA_DIR, _LOCKED['digits'])
    if df is None:
        print("  ERROR: USTEC data not found.")
        sys.exit(1)
    print(f"\n  Data : {df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()} | {len(df):,} bars")
    print(f"  Cap  : ${CAP:.0f}  |  SL ≈ 54 pts (0.3% × ~18000)")
    print(f"\n  Stress parameter key:")
    print(f"    stress_spread_mult  — multiplies real MT5 spread_px")
    print(f"    comm                — round-trip USD per lot")
    print(f"    entry_slippage_pts  — ep += slip × direction (worse fill for LONG)")
    print(f"\n  {'label':<18}  {'mult':>5}  {'comm':>5}  {'slip':>5}")
    print(f"  {'─'*36}")
    for label, mult, comm, slip in SCENARIOS:
        print(f"  {label:<18}  {mult:>5.1f}  {comm:>5.2f}  {slip:>5.1f}")

    def run(p, period_start):
        t, e = run_backtest('USTEC', df, lr, vm, p, CAP,
                            period_start, None,
                            'TEST' if period_start else 'FULL')
        return calc_metrics(t, e, CAP)

    def build_p(mult, comm, slip):
        return {**_LOCKED,
                'stress_spread_mult':  mult,
                'comm':                comm,
                'entry_slippage_pts':  slip}

    # ─── Pre-compute all metrics ──────────────────────────────────────────────
    results_full = []
    results_test = []
    for label, mult, comm, slip in SCENARIOS:
        p = build_p(mult, comm, slip)
        results_full.append(run(p, None))
        results_test.append(run(p, TEST_START))

    base_f = results_full[0]
    base_t = results_test[0]

    # ═════════════════════════════════════════════════════════════════════════
    # RESULTS TABLE
    # ═════════════════════════════════════════════════════════════════════════

    for period_label, results, base_m in [
        ('FULL HISTORY',              results_full, base_f),
        ('TEST (2025-01-01→present)', results_test, base_t),
    ]:
        _title(f"RESULTS — {period_label}")
        print(_hdr())
        _sep()
        for (label, _, _, _), m in zip(SCENARIOS, results):
            print(_row(m, label))

        # Delta vs BASE
        _sub("DELTA vs BASE")
        print(
            f"  {'scenario':<18}  {'Δn':>4}  {'ΔPF':>6}  "
            f"{'Δret%':>8}  {'ΔMDD%':>7}  {'Δexp':>8}  {'verdict':>22}"
        )
        _sep()
        for (label, _, _, _), m in zip(SCENARIOS[1:], results[1:]):
            if not m or m.get('n', 0) == 0:
                print(f"  {label:<18}  — no trades"); continue
            dsh = (
                f"{m['sharpe']-base_m['sharpe']:>+8.3f}"
                if pd.notna(m.get('sharpe')) and pd.notna(base_m.get('sharpe'))
                else "       —"
            )
            verd = _verdict(m['pf'])
            print(
                f"  {label:<18}  {m['n']-base_m['n']:>+4}  "
                f"{m['pf']-base_m['pf']:>+6.2f}  "
                f"{m['ret']-base_m['ret']:>+7.2f}%  "
                f"{m['mdd']-base_m['mdd']:>+7.2f}%  "
                f"{m['exp']-base_m['exp']:>+8.4f}  "
                f"{verd:>22}"
            )

    # ═════════════════════════════════════════════════════════════════════════
    # VERDICT MATRIX
    # ═════════════════════════════════════════════════════════════════════════
    _title("VERDICT MATRIX")
    print(f"\n  {'scenario':<18}  {'PF_FULL':>8}  {'verdict_FULL':>22}  {'PF_TEST':>8}  {'verdict_TEST':>22}")
    _sep()
    for (label, _, _, _), mf, mt in zip(SCENARIOS, results_full, results_test):
        pf_f = mf.get('pf', 0) if mf else 0
        pf_t = mt.get('pf', 0) if mt else 0
        print(
            f"  {label:<18}  {pf_f:>8.2f}  {_verdict(pf_f):>22}  "
            f"{pf_t:>8.2f}  {_verdict(pf_t):>22}"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═════════════════════════════════════════════════════════════════════════
    _title("FINAL VERDICT")

    combined_f = results_full[5]   # 'Combined'
    extreme_f  = results_full[6]   # 'Extreme'
    combined_t = results_test[5]
    extreme_t  = results_test[6]

    def _grade(mf, mt, name):
        pf_f = mf.get('pf', 0) if mf else 0
        pf_t = mt.get('pf', 0) if mt else 0
        print(f"\n  {name}")
        print(f"    FULL : PF={pf_f:.2f}  → {_verdict(pf_f)}")
        print(f"    TEST : PF={pf_t:.2f}  → {_verdict(pf_t)}")

    _grade(results_full[1], results_test[1], "Hard spread x5")
    _grade(results_full[2], results_test[2], "Hard cost (x3 + c=5)")
    _grade(results_full[4], results_test[4], "Slip 5pt")
    _grade(combined_f, combined_t,           "Combined (x3 + c=5 + slip=2)")
    _grade(extreme_f,  extreme_t,            "Extreme  (x5 + c=10 + slip=5)")

    print(f"""
  Thresholds used:
    PF ≥ {PF_SURVIVES:.1f}  →  SURVIVES
    PF ≥ {PF_DEGRADES:.1f}  →  DEGRADES (acceptable — edge present, friction cuts margin)
    PF <  {PF_DEGRADES:.1f}  →  BREAKS   (edge not sufficient to absorb cost level)

  If EXTREME is still DEGRADES or better: phase 9 can close.
  If COMBINED is SURVIVES: edge is institutionally durable under realistic friction.
  If BASE TEST PF is ≥4 and COMBINED TEST PF is ≥2: proceed to phase 10.

  ** Sharpe: monthly PnL / initial_cap, annualized sqrt(12). Not equity-weighted.
""")
    print(f"{'═'*W}\n")


if __name__ == '__main__':
    main()
