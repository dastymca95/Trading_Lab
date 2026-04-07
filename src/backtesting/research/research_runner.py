"""
research_runner.py
==================
Research skeleton — session-based baseline analysis.

Purpose
-------
NOT a strategy.  Provides the minimum infrastructure to:
  1. Load price data (reuses backtest_data.load_price_data).
  2. Label every bar with its UTC session (via session_clock).
  3. Compute per-session directional/return statistics as a baseline.
  4. Print a clean summary: know the raw session bias BEFORE building any logic.

This is the ground-truth audit that must pass before any session hypothesis
(EU-open drift, US-open momentum, etc.) is tested with real entry/exit logic.

Assets configured
-----------------
  RESEARCH_ASSETS = ['US500', 'USTEC']   — primary cluster (low commission)
  DE40 can be added later as separate track.
  US30 is hard mode — add after the cluster is understood.

Usage (from src/backtesting/)
------------------------------
  python research/research_runner.py

To add an asset temporarily:
  Edit RESEARCH_ASSETS below — no other file needs changing.

Next steps (NOT implemented here)
-----------------------------------
  - fixed_time_entry(): enter at session open, exit N bars later or at close
  - session_filter_baseline(): add a single filter (e.g. LRR > threshold)
  - This file is the skeleton; each hypothesis gets its own function here.
"""

import os
import sys

import numpy as np
import pandas as pd

# ─── Path setup: allow imports from src/backtesting/ ─────────────────────────
# This lets us reuse backtest_data without duplicating data loading logic.
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))          # .../research/
_PARENT_DIR  = os.path.dirname(_THIS_DIR)                           # .../backtesting/
_REPO_ROOT   = os.path.dirname(os.path.dirname(_PARENT_DIR))       # .../Trading_Lab/
sys.path.insert(0, _PARENT_DIR)

from backtest_data import load_price_data                           # noqa: E402
from research.session_clock import (                                # noqa: E402
    label_sessions, session_distribution, MT5_TZ_DEFAULT,
)

# ─── Research config ─────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(_REPO_ROOT, 'data', 'backtesting')
RESEARCH_ASSETS = ['US500', 'USTEC']    # primary cluster; add 'DE40', 'US30' later
MT5_TZ          = MT5_TZ_DEFAULT        # override here if broker uses DST

ASSET_DIGITS = {
    'US30': 2, 'USTEC': 2, 'US500': 2, 'DE40': 2, 'XAUUSD': 2,
}

# Sessions to analyse in the descriptive baseline report
SESSIONS_OF_INTEREST = ['EU_PRE', 'EU_OPEN', 'EU_MID', 'US_PRE', 'US_OPEN', 'US_MID']

# ─── Fixed-window baseline config ────────────────────────────────────────────
# Cost params per asset (mirrors backtest_config; update here if specs change).
# sp = spread in price points, cs = contract size, ml = minimum lot.
ASSET_COST_PARAMS = {
    'US500': {'sp': 0.5, 'cs': 1, 'ml': 0.1},
    'USTEC': {'sp': 1.0, 'cs': 1, 'ml': 0.1},
}

# Date split — mirrors backtest_config for consistent FULL/TEST labeling.
_TRAIN_END  = pd.Timestamp('2025-01-01')
_TEST_START = pd.Timestamp('2025-01-01')

# Baselines to run.  Add rows here to test new hypotheses.
BASELINES = [
    {'name': 'EU_OPEN_LONG', 'session': 'EU_OPEN', 'direction': 1},
    {'name': 'US_MID_LONG',  'session': 'US_MID',  'direction': 1},
]

# ─── Session return baseline ──────────────────────────────────────────────────

def session_return_stats(df: pd.DataFrame, session: str) -> pd.DataFrame:
    """
    For each calendar day, compute the session's net price movement:
      open-of-first-session-bar  →  close-of-last-session-bar

    Returns a DataFrame (one row per date):
      date, open_px, close_px, ret_pts, ret_pct, direction (1/-1/0)

    This is the raw directional bias of the session — no entry logic yet.
    A bull% far from 50% is the signal that a session *might* have edge.
    """
    mask = df['session_label'] == session
    sub  = df[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for date, grp in sub.groupby('date'):
        grp = grp.sort_values('time_utc')
        o   = float(grp['open'].iloc[0])
        c   = float(grp['close'].iloc[-1])
        ret_pts = c - o
        ret_pct = (ret_pts / o * 100) if o else np.nan
        rows.append({
            'date':      date,
            'open_px':   o,
            'close_px':  c,
            'ret_pts':   round(ret_pts, 4),
            'ret_pct':   round(ret_pct, 6),
            'direction': 1 if ret_pts > 0 else (-1 if ret_pts < 0 else 0),
        })
    return pd.DataFrame(rows)


def _print_session_stats(stats: pd.DataFrame, asset: str, session: str) -> None:
    if stats.empty:
        print(f"  [{asset}] {session:<10s}  — no data")
        return
    n    = len(stats)
    bull = (stats['direction'] == 1).mean() * 100
    avg  = stats['ret_pts'].mean()
    med  = stats['ret_pts'].median()
    std  = stats['ret_pts'].std()
    print(
        f"  [{asset}] {session:<10s}  "
        f"n={n:4d}  bull%={bull:5.1f}  "
        f"avg={avg:+8.3f}pts  med={med:+8.3f}pts  std={std:7.3f}pts"
    )


# ─── Fixed-window baseline engine ────────────────────────────────────────────

def run_fixed_window_baseline(
    df: pd.DataFrame,
    session: str,
    direction: int,
    sp: float,
    lots: float,
    cs: float,
    cap_start: float = 250.0,
    date_start: pd.Timestamp = None,
    date_end: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Simulate a fixed-window session trade: enter at session open, exit at
    session close.  One trade per day.  No trailing, no BE, no filters.

    Spread cost applied once at entry (same convention as backtest_runner).
    Commission = 0 (US500/USTEC have no RT commission at this broker).

    Parameters
    ----------
    df        : labeled DataFrame (must have session_label + time_utc columns).
    session   : session label to trade (e.g. 'EU_OPEN').
    direction : 1 = LONG, -1 = SHORT.
    sp        : spread in price points.
    lots      : lot size (use ml = minimum lot for conservative baseline).
    cs        : contract size (value per point per lot).
    cap_start : starting capital for this asset slice.
    date_start, date_end : optional date range filter (inclusive / exclusive).

    Returns
    -------
    DataFrame of trades:
      date, entry, exit, gross, cost, net, result ('WIN'/'LOSS'/'BE'), cap
    """
    daily = session_return_stats(df, session)
    if daily.empty:
        return pd.DataFrame()

    if date_start is not None:
        daily = daily[pd.to_datetime(daily['date']) >= date_start]
    if date_end is not None:
        daily = daily[pd.to_datetime(daily['date']) < date_end]
    daily = daily.reset_index(drop=True)

    cap  = cap_start
    rows = []
    for _, row in daily.iterrows():
        ep    = row['open_px']
        xp    = row['close_px']
        gross = (xp - ep) * direction * lots * cs
        cost  = sp * lots * cs          # spread cost at entry
        net   = gross - cost
        result = 'WIN' if net > 1e-4 else ('LOSS' if net < -1e-4 else 'BE')
        cap   += net
        rows.append({
            'date':   row['date'],
            'entry':  round(ep, 4),
            'exit':   round(xp, 4),
            'gross':  round(gross, 4),
            'cost':   round(cost, 4),
            'net':    round(net, 4),
            'result': result,
            'cap':    round(cap, 4),
        })
    return pd.DataFrame(rows)


def _baseline_metrics(trades: pd.DataFrame, cap_start: float) -> dict:
    """Compute summary metrics from a fixed-window trade list."""
    if trades.empty:
        return {}
    n      = len(trades)
    wins   = trades.loc[trades['result'] == 'WIN',  'net']
    losses = trades.loc[trades['result'] == 'LOSS', 'net']
    n_win  = len(wins)
    n_loss = len(losses)

    wr     = n_win / n * 100 if n else 0.0
    aw     = float(wins.mean())   if n_win  else 0.0
    al     = float(losses.mean()) if n_loss else 0.0

    gross_w = float(wins.sum())
    gross_l = abs(float(losses.sum()))
    pf      = gross_w / gross_l if gross_l > 0 else float('inf')
    exp     = float(trades['net'].mean())
    ret_pct = trades['net'].sum() / cap_start * 100

    caps    = np.array([cap_start] + list(trades['cap']))
    peak    = np.maximum.accumulate(caps)
    dd      = (caps - peak) / np.where(peak > 0, peak, 1.0)
    mdd     = float(dd.min()) * 100

    return dict(
        n=n, n_win=n_win, n_loss=n_loss,
        wr=round(wr, 1),
        aw=round(aw, 4), al=round(al, 4),
        pf=round(pf, 3) if pf != float('inf') else float('inf'),
        exp=round(exp, 4),
        ret_pct=round(float(ret_pct), 2),
        mdd=round(mdd, 2),
    )


def _print_baseline_report(m: dict, label: str) -> None:
    """Print a one-block summary for a single baseline + period combination."""
    if not m:
        print(f"  {label}: no trades")
        return
    pf_str = f"{m['pf']:.3f}" if m['pf'] != float('inf') else "inf"
    print(
        f"  {label}\n"
        f"    n={m['n']}  WR={m['wr']}%  "
        f"aw={m['aw']:+.4f}  al={m['al']:+.4f}  PF={pf_str}\n"
        f"    exp/trade={m['exp']:+.4f}  "
        f"return={m['ret_pct']:+.2f}%  MDD={m['mdd']:.2f}%"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   RESEARCH RUNNER — session baseline analysis           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  MT5 timezone : {MT5_TZ}")
    print(f"  Assets       : {RESEARCH_ASSETS}")
    print(f"  Data dir     : {DATA_DIR}\n")

    for asset in RESEARCH_ASSETS:
        digits = ASSET_DIGITS.get(asset, 2)
        result = load_price_data(asset, DATA_DIR, digits)
        if result[0] is None:
            print(f"\n  ⚠️  {asset}: datos no encontrados en {DATA_DIR}")
            continue

        df, _lr, _vm, _qc = result

        # ── Session labeling ─────────────────────────────────────────────
        df = label_sessions(df, MT5_TZ)

        # ── Distribution check ───────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  {asset} — session distribution (UTC-anchored)")
        print(f"{'─'*60}")
        dist = session_distribution(df)
        print(dist.to_string())

        # ── Per-session return baseline ──────────────────────────────────
        print(f"\n  {asset} — directional baseline per session")
        print(f"  {'session':<10s}  {'n':>4s}  {'bull%':>6s}  "
              f"{'avg pts':>9s}  {'median pts':>10s}  {'std pts':>8s}")
        print(f"  {'-'*65}")
        for sess in SESSIONS_OF_INTEREST:
            stats = session_return_stats(df, sess)
            _print_session_stats(stats, asset, sess)

        # ── Fixed-window baselines ────────────────────────────────────────
        ap = ASSET_COST_PARAMS.get(asset)
        if ap is None:
            print(f"\n  ⚠️  {asset}: sin cost params en ASSET_COST_PARAMS, skipping baselines")
            continue

        print(f"\n  {asset} — fixed-window baselines  "
              f"(lots={ap['ml']}, sp={ap['sp']}, cs={ap['cs']}, comm=0)")
        print(f"  {'─'*65}")

        for bl in BASELINES:
            for period, ds, de in [
                ('FULL', None,        _TRAIN_END),
                ('TEST', _TEST_START, None),
            ]:
                trades = run_fixed_window_baseline(
                    df,
                    session    = bl['session'],
                    direction  = bl['direction'],
                    sp         = ap['sp'],
                    lots       = ap['ml'],
                    cs         = ap['cs'],
                    cap_start  = 250.0,
                    date_start = ds,
                    date_end   = de,
                )
                m = _baseline_metrics(trades, 250.0)
                _print_baseline_report(m, f"{bl['name']} | {period}")
            print()

    print("✅  Research baseline completo.")


if __name__ == '__main__':
    main()
