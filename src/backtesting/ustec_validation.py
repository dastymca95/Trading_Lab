#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USTEC Advanced Validation — Solo Candidate
===========================================
Comprehensive robustness audit for USTEC operational variant.

Sections
--------
  1. Core metrics       — FULL vs TEST side by side
  2. Year-by-year       — temporal stability
  3. Statistical tests  — t-test, sign test, bootstrap CI
  4. Monte Carlo DD     — stress test under random permutation
  5. Drawdown profile   — depth and duration analysis
  6. Monthly returns    — calendar heatmap
  7. Return distribution — daily return shape
  8. Concentration      — dependence on top trades
  9. Verdict

Run from: src/backtesting/
  python ustec_validation.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _THIS_DIR)

from backtest_config import (
    GLOBAL_RISK_PCT, INITIAL_PER_ASSET, TEST_START,
)
from backtest_data   import load_price_data
from backtest_runner import run_backtest
from backtest_specs  import resolve_asset_params
from backtest_stats  import (
    calc_metrics, daily_equity, monthly_heatmap, return_distribution,
)

DATA_DIR = os.path.join(_REPO_ROOT, "data", "backtesting")
CAP      = INITIAL_PER_ASSET
ASSET    = 'USTEC'
P        = dict(resolve_asset_params(DATA_DIR)[ASSET])
P['risk_pct'] = GLOBAL_RISK_PCT

W = 74   # console width


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _sep(c='─'): return c * W

def _h(title):
    print(f"\n{_sep()}")
    print(f"  {title}")
    print(_sep())

def _yn(cond):   return "✓" if cond else "✗"
def _pv(p):
    stars = "***" if p < 0.001 else ("** " if p < 0.01 else ("*  " if p < 0.05 else "   "))
    return f"{p:.4f} {stars}"


# ─── Core summary row ─────────────────────────────────────────────────────────

def _core_row(m, label):
    if not m:
        return f"  {label:<6}  — no trades"
    sh = f"{m['sharpe']:>6.3f}" if pd.notna(m.get('sharpe', float('nan'))) else "     —"
    so = f"{m['sortino']:>7.3f}" if pd.notna(m.get('sortino', float('nan'))) else "      —"
    return (
        f"  {label:<6}  n={m['n']:>4}  WR={m['wr']:>5.1f}%  PF={m['pf']:>5.2f}  "
        f"ret={m['ret']:>+7.2f}%  MDD={m['mdd']:>6.2f}%  "
        f"Sh={sh}  So={so}  Calmar={m['calmar']:>6.3f}"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("╔" + "═"*(W-2) + "╗")
    print("║   USTEC ADVANCED VALIDATION — SOLO CANDIDATE" + " "*(W-47) + "║")
    print("╚" + "═"*(W-2) + "╝")
    print(f"\n  Config : hours={P['hours']}  force_dir={P.get('force_direction','—')}"
          f"  LOW_VOL(p{P['low_vol_pct']},w{P['low_vol_win']})"
          f"  roll_mfe>{P['roll_mfe_min']}")
    print(f"  Risk   : {GLOBAL_RISK_PCT:.2%}/trade  |  Cap : ${CAP:.0f}")
    print(f"  Test   : {TEST_START.date()} → present")

    # ── Load + run ────────────────────────────────────────────────────────────
    print(f"\n  Loading data...", end=' ', flush=True)
    df, lr, vm, _ = load_price_data(ASSET, DATA_DIR, P['digits'])
    if df is None:
        print(f"\n  ❌ Data not found in {DATA_DIR}"); return
    print(f"ok  ({len(df):,} bars  {df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()})")

    print(f"  Running backtest...", end=' ', flush=True)
    t_f, e_f = run_backtest(ASSET, df, lr, vm, P, CAP, None,       None,  'FULL')
    t_t, e_t = run_backtest(ASSET, df, lr, vm, P, CAP, TEST_START, None,  'TEST')
    m_f = calc_metrics(t_f, e_f, CAP)
    m_t = calc_metrics(t_t, e_t, CAP)
    print(f"done  (FULL n={m_f.get('n',0)}  TEST n={m_t.get('n',0)})")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1 — CORE METRICS
    # ═════════════════════════════════════════════════════════════════════════
    _h("1. CORE METRICS — FULL vs TEST")
    print(_core_row(m_f, 'FULL'))
    print(_core_row(m_t, 'TEST'))
    print()
    # Consistency check: TEST should not be dramatically worse than FULL
    for label, m in [('FULL', m_f), ('TEST', m_t)]:
        if not m: continue
        print(f"  {label:<4}  n_win={m['n_win']}  n_loss={m['n_loss']}  n_be={m['n_be']}"
              f"  max_consec_loss={m['ms']}  avg_win=${m['aw']:+.2f}  avg_loss=${m['al']:+.2f}"
              f"  RR={m['rr']:.3f}  exp/trade=${m['exp']:+.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2 — YEAR-BY-YEAR
    # ═════════════════════════════════════════════════════════════════════════
    _h("2. YEAR-BY-YEAR  (temporal stability)")
    yhdr = (f"  {'year':<6}  {'n':>4}  {'WR%':>5}  {'PF':>5}  "
            f"{'ret%':>8}  {'MDD%':>7}  {'aw':>7}  {'al':>7}  {'exp':>8}")
    ysep = f"  {'─'*70}"
    print(yhdr); print(ysep)

    data_start = pd.Timestamp(df['time'].iloc[0]).normalize()
    data_end = pd.Timestamp(df['time'].iloc[-1]).normalize()
    yrs_pos = 0; yrs_neg = 0
    full_years = 0
    seen_years = sorted(t_f['Año'].unique())
    for yr in seen_years:
        t_y = t_f[t_f['Año'] == yr].reset_index(drop=True)
        if t_y.empty: continue
        pnl = t_y['PnL Neto USD'].values
        cap_y = CAP
        eq_y = [cap_y]
        for p in pnl:
            cap_y = max(cap_y + p, 0.01); eq_y.append(cap_y)
        year_start = pd.Timestamp(f"{yr}-01-01")
        year_end = pd.Timestamp(f"{yr}-12-31")
        is_partial = (
            (yr == data_start.year and (data_start.month != 1 or data_start.day != 1)) or
            (yr == data_end.year and (data_end.month != 12 or data_end.day != 31))
        )
        m_y = calc_metrics(
            t_y,
            np.array(eq_y),
            CAP,
            sample_start=max(year_start, data_start),
            sample_end=min(year_end, data_end),
        )
        if not m_y: continue
        pf_s  = f"{m_y['pf']:>5.2f}" if m_y['pf'] < 99 else "  ≫99"
        tag   = "  ◄ TEST" if yr >= TEST_START.year else ""
        tag   = "  ◄ TEST START" if yr == TEST_START.year else tag
        if is_partial:
            tag += "  [PARTIAL]"
        else:
            full_years += 1
            if m_y['ret'] > 0: yrs_pos += 1
            else:              yrs_neg += 1
        print(f"  {yr:<6}  {m_y['n']:>4}  {m_y['wr']:>4.1f}%  {pf_s}  "
              f"{m_y['ret']:>+7.2f}%  {m_y['mdd']:>7.2f}%  "
              f"{m_y['aw']:>+7.2f}  {m_y['al']:>+7.2f}  {m_y['exp']:>+8.4f}{tag}")
    print(ysep)
    total_yrs = yrs_pos + yrs_neg
    if total_yrs > 0:
        print(f"  Profitable full years: {yrs_pos}/{total_yrs}  ({yrs_pos/total_yrs*100:.0f}%)")
    if full_years != len(seen_years):
        print("  Partial years are shown for traceability but excluded from the profitable-years summary.")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3 — STATISTICAL TESTS
    # ═════════════════════════════════════════════════════════════════════════
    _h("3. STATISTICAL TESTS")
    for label, m in [('FULL', m_f), ('TEST', m_t)]:
        if not m: continue
        print(f"  {label}")
        print(f"    t-test (daily PnL ≠ 0)  p={_pv(m['p_val'])}"
              f"  → {'significant' if m['p_val'] < 0.05 else 'NOT significant'}")
        print(f"    sign test               p={_pv(m['sign_p'])}"
              f"  → {'significant' if m['sign_p'] < 0.05 else 'NOT significant'}")
        print(f"    bootstrap CI 95%        [{m['boot_lo']:+.4f}, {m['boot_hi']:+.4f}]"
              f"  → {'excludes 0 ✓' if m['boot_lo'] > 0 else 'includes 0 ✗'}")
        print()

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 4 — MONTE CARLO STRESS TEST
    # ═════════════════════════════════════════════════════════════════════════
    _h("4. MONTE CARLO STRESS TEST  (2000 random permutations)")
    for label, m in [('FULL', m_f), ('TEST', m_t)]:
        if not m: continue
        p95 = m['mc_dd_p95']
        print(f"  {label}  DD p5={m['mc_dd_p5']:>7.2f}%  "
              f"p50={m['mc_dd_p50']:>7.2f}%  p95={p95:>7.2f}%  "
              f"final_p50=${m['mc_final_p50']:.2f}"
              f"  {'⚠ p95 DD > -20%' if p95 < -20 else '✓ p95 DD contained'}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5 — DRAWDOWN PROFILE
    # ═════════════════════════════════════════════════════════════════════════
    _h("5. DRAWDOWN PROFILE")
    for label, t, cap0 in [('FULL', t_f, CAP), ('TEST', t_t, CAP)]:
        if t.empty: continue
        deq = daily_equity(t, cap0)
        dds = deq['DrawdownPct'].values
        in_dd = dds < -0.01   # days with meaningful drawdown
        # Count drawdown periods (transitions into drawdown)
        transitions = np.diff(in_dd.astype(int))
        n_periods = int((transitions == 1).sum())
        dd_days = int(in_dd.sum())
        worst = float(dds.min())
        mean_dd = float(dds[in_dd].mean()) if dd_days > 0 else 0.0
        pct_in_dd = dd_days / len(dds) * 100 if len(dds) > 0 else 0.0
        print(f"  {label}  worst={worst:.2f}%  mean_in_dd={mean_dd:.2f}%  "
              f"dd_periods={n_periods}  days_in_dd={dd_days} ({pct_in_dd:.0f}%)  "
              f"worst_day=${t['PnL Neto USD'].groupby(t['Date']).sum().min():.2f}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 6 — MONTHLY RETURNS
    # ═════════════════════════════════════════════════════════════════════════
    _h("6. MONTHLY RETURNS  (net PnL in USD, FULL history)")
    heat = monthly_heatmap(t_f)
    if not heat.empty:
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        avail  = [m for m in months if m in heat.columns]
        hdr    = f"  {'year':<6}  " + "  ".join(f"{m:>6}" for m in avail) + "  {'tot':>7}"
        print(hdr)
        print(f"  {'─'*70}")
        for _, row in heat.iterrows():
            yr  = int(row['Year'])
            vals = []
            tot = 0.0
            for m in avail:
                v = row.get(m, 0.0)
                tot += v
                vals.append(f"{v:>+6.1f}" if v != 0 else f"{'—':>6}")
            tag = "  ◄ test" if yr >= TEST_START.year else ""
            print(f"  {yr:<6}  {'  '.join(vals)}  {tot:>+7.1f}{tag}")
        print()
        # Month win rate across years
        print(f"  {'month':<6}  " + "  ".join(f"{m:>6}" for m in avail))
        print(f"  {'─'*70}")
        wr_row = []
        for m in avail:
            col = heat[m].values if m in heat.columns else np.array([])
            pct = (col > 0).mean() * 100 if len(col) > 0 else float('nan')
            wr_row.append(f"{pct:>5.0f}%" if not np.isnan(pct) else "    —")
        print(f"  {'WR%':<6}  {'  '.join(wr_row)}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 7 — RETURN DISTRIBUTION
    # ═════════════════════════════════════════════════════════════════════════
    _h("7. RETURN DISTRIBUTION  (daily returns, FULL)")
    rd = return_distribution(t_f, CAP)
    if not rd.empty:
        rd_d = dict(zip(rd['Metric'], rd['Value']))
        print(f"  Days    : {int(rd_d.get('CountDays',0))}")
        print(f"  Mean    : {rd_d.get('MeanDailyPct',0):>+.4f}%/day")
        print(f"  Median  : {rd_d.get('MedianDailyPct',0):>+.4f}%/day")
        print(f"  Std     : {rd_d.get('StdDailyPct',0):>.4f}%")
        print(f"  Skew    : {rd_d.get('SkewDaily',0):>+.3f}"
              f"  {'(positive = right tail, wins > losses in size)' if rd_d.get('SkewDaily',0) > 0 else '(negative = left tail)'}")
        print(f"  Kurtosis: {rd_d.get('KurtosisDaily',0):>+.3f}"
              f"  {'(fat tails)' if abs(rd_d.get('KurtosisDaily',0)) > 1 else '(near-normal)'}")
        print(f"  p05/p50/p95: {rd_d.get('Pct05',0):>+.4f}% / "
              f"{rd_d.get('Pct50',0):>+.4f}% / "
              f"{rd_d.get('Pct95',0):>+.4f}%")
        print(f"  Worst day : {rd_d.get('WorstDayPct',0):>+.4f}%"
              f"  Best day : {rd_d.get('BestDayPct',0):>+.4f}%")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 8 — CONCENTRATION
    # ═════════════════════════════════════════════════════════════════════════
    _h("8. CONCENTRATION CHECK  (dependence on top trades)")
    for label, t in [('FULL', t_f), ('TEST', t_t)]:
        if t.empty: continue
        total_pnl = t['PnL Neto USD'].sum()
        top5  = t.nlargest(5,  'PnL Neto USD')['PnL Neto USD']
        bot5  = t.nsmallest(5, 'PnL Neto USD')['PnL Neto USD']
        top5_pct = top5.sum() / total_pnl * 100 if total_pnl > 0 else float('nan')
        bot5_pct = bot5.sum() / total_pnl * 100 if total_pnl > 0 else float('nan')
        print(f"  {label}  total_net=${total_pnl:.2f}")
        print(f"         top-5 wins  : ${top5.sum():.2f}  ({top5_pct:.1f}% of total PnL)"
              f"  {'⚠ concentrated' if top5_pct > 50 else '✓ distributed'}")
        print(f"         top-5 losses: ${bot5.sum():.2f}  ({bot5_pct:.1f}% of total PnL)")
        print(f"         max consec loss: {m_f['ms'] if label=='FULL' else m_t['ms']}")
        print()

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 9 — VERDICT
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*W}")
    print("  VERDICT")
    print(f"{'═'*W}")

    checks = [
        ("PF > 2.0 FULL",           m_f.get('pf', 0) > 2.0),
        ("PF > 2.0 TEST",           m_t.get('pf', 0) > 2.0),
        ("MDD < -15% FULL",         m_f.get('mdd', -99) > -15.0),
        ("MDD < -15% TEST",         m_t.get('mdd', -99) > -15.0),
        ("Sharpe > 1.5 FULL",       m_f.get('sharpe', 0) > 1.5   if pd.notna(m_f.get('sharpe')) else False),
        ("Sharpe > 1.5 TEST",       m_t.get('sharpe', 0) > 1.5   if pd.notna(m_t.get('sharpe')) else False),
        ("t-test p < 0.05 FULL",    m_f.get('p_val', 1) < 0.05),
        ("sign test p < 0.05 FULL", m_f.get('sign_p', 1) < 0.05),
        ("bootstrap CI > 0 FULL",   m_f.get('boot_lo', 0) > 0),
        ("bootstrap CI > 0 TEST",   m_t.get('boot_lo', 0) > 0),
        ("MC p95 DD > -20%",        m_f.get('mc_dd_p95', -99) > -20.0),
        ("profitable years ≥ 75%",  yrs_pos / max(total_yrs, 1) >= 0.75),
    ]

    passed = sum(1 for _, v in checks if v)
    total  = len(checks)

    for label, v in checks:
        print(f"  {_yn(v)}  {label}")

    print(f"\n  Score: {passed}/{total}  ({passed/total*100:.0f}%)")
    print()

    if passed >= 10:
        verdict = "READY"
        reason  = ("All critical statistical tests pass. "
                   "PF, Sharpe, and MDD are strong in both FULL and TEST. "
                   "Edge is statistically credible across years.")
    elif passed >= 7:
        verdict = "CONDITIONAL"
        reason  = ("Most checks pass but some statistical or stability conditions "
                   "need monitoring. Viable candidate with active oversight.")
    else:
        verdict = "NOT READY"
        reason  = ("Multiple critical checks fail. "
                   "Edge credibility is not yet established.")

    print(f"  ┌{'─'*68}┐")
    print(f"  │  {'VERDICT: ' + verdict:<66}│")
    print(f"  │  {reason[:66]:<66}│")
    if len(reason) > 66:
        print(f"  │  {reason[66:132]:<66}│")
    print(f"  └{'─'*68}┘")
    print(f"\n{'═'*W}\n")


if __name__ == '__main__':
    main()

