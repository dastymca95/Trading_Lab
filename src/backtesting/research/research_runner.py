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

# Sessions to analyse in the baseline report
SESSIONS_OF_INTEREST = ['EU_PRE', 'EU_OPEN', 'EU_MID', 'US_PRE', 'US_OPEN', 'US_MID']

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

    print("\n✅  Research baseline completo.")
    print(
        "\nNext step: pick the session with strongest bull% bias"
        " and build a fixed_time_entry() hypothesis in this file."
    )


if __name__ == '__main__':
    main()
