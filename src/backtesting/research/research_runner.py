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

import datetime
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

# Primary research filter — consolidated baseline (LOW_VOL p50 w90).
_PRIMARY_PCT = 50
_PRIMARY_WIN = 90

# Baselines to run.  Add rows here to test new hypotheses.
BASELINES = [
    {'name': 'EU_OPEN_LONG', 'session': 'EU_OPEN', 'direction': 1},
    {'name': 'US_MID_LONG',  'session': 'US_MID',  'direction': 1},
]

# ─── Asset-specific primary hypothesis config ─────────────────────────────────
# After timing refinement we no longer use a shared US_MID window for all assets.
# US500 → edge is distributed across full US_MID (16-19h UTC)  — use full window.
# USTEC → edge is concentrated in the early sub-window (16-17h UTC) — restrict.
# LOW_VOL filter applied to both: p50, w90 (primary filter from robustness sweep).
ASSET_PRIMARY_CONFIG = {
    'US500': {
        'utc_hour_from': None,  # full US_MID 16-19h UTC
        'utc_hour_to':   None,
        'window_label':  'US_MID full  16-19h UTC',
    },
    'USTEC': {
        'utc_hour_from': 16,    # early US_MID 16-17h UTC
        'utc_hour_to':   17,
        'window_label':  'US_MID early 16-17h UTC',
    },
}

# ─── FOMC decision dates ─────────────────────────────────────────────────────
# FOMC rate decisions typically land at 14:00 ET = 19:00 UTC (winter) /
# 18:00 UTC (summer) — inside the US_MID window (16:00–19:59 UTC).
# NFP/CPI/ISM fall at 13:30–15:00 UTC (US_PRE / US_OPEN) — outside US_MID.
# This makes FOMC the only major scheduled catalyst within US_MID.
_FOMC_DATE_STRS = [
    # 2018
    '2018-01-31','2018-03-21','2018-05-02','2018-06-13',
    '2018-08-01','2018-09-26','2018-11-08','2018-12-19',
    # 2019
    '2019-01-30','2019-03-20','2019-05-01','2019-06-19',
    '2019-07-31','2019-09-18','2019-10-30','2019-12-11',
    # 2020 (includes Mar emergency cuts)
    '2020-01-29','2020-03-03','2020-03-15','2020-04-29',
    '2020-06-10','2020-07-29','2020-09-16','2020-11-05','2020-12-16',
    # 2021
    '2021-01-27','2021-03-17','2021-04-28','2021-06-16',
    '2021-07-28','2021-09-22','2021-11-03','2021-12-15',
    # 2022
    '2022-01-26','2022-03-16','2022-05-04','2022-06-15',
    '2022-07-27','2022-09-21','2022-11-02','2022-12-14',
    # 2023
    '2023-02-01','2023-03-22','2023-05-03','2023-06-14',
    '2023-07-26','2023-09-20','2023-11-01','2023-12-13',
    # 2024
    '2024-01-31','2024-03-20','2024-05-01','2024-06-12',
    '2024-07-31','2024-09-18','2024-11-07','2024-12-18',
    # 2025
    '2025-01-29','2025-03-19','2025-05-07','2025-06-18',
    '2025-07-30','2025-09-17','2025-10-29','2025-12-10',
    # 2026 (confirmed through dataset end)
    '2026-01-28','2026-03-18',
]
_FOMC_DATES = frozenset(
    datetime.date.fromisoformat(d) for d in _FOMC_DATE_STRS
)


# ─── Volatility regime ────────────────────────────────────────────────────────

def compute_daily_vol_regime(
    df: pd.DataFrame, window: int = 60, percentile: int = 50
) -> pd.Series:
    """
    Classify each trading date as HIGH_VOL or LOW_VOL.

    Method: daily ATR = mean of bar-level atr14 values per day.
    Threshold = rolling `percentile`-th quantile over `window` days.
    LOW_VOL  = daily_atr <= threshold  (bottom `percentile`% of days)
    HIGH_VOL = daily_atr >  threshold

    Parameters
    ----------
    window      : rolling lookback in trading days. Default 60.
    percentile  : threshold quantile (0-100). Default 50 = median split,
                  identical to prior behavior.  25 = strict (calmest 25%),
                  75 = lax (bottom 75%).

    Uses atr14 already computed by load_price_data() — no extra data needed.

    Returns
    -------
    pd.Series indexed by datetime.date, values 'HIGH_VOL' or 'LOW_VOL'.
    """
    daily_atr    = df.groupby('date')['atr14'].mean()
    rolling_thr  = daily_atr.rolling(window, min_periods=window // 2).quantile(
        percentile / 100.0
    )
    regime = pd.Series('LOW_VOL', index=daily_atr.index, dtype=object)
    regime[daily_atr > rolling_thr] = 'HIGH_VOL'
    return regime


# ─── Session return baseline ──────────────────────────────────────────────────

def session_return_stats(
    df: pd.DataFrame,
    session: str,
    utc_hour_from: int = None,
    utc_hour_to: int = None,
) -> pd.DataFrame:
    """
    For each calendar day, compute the session's net price movement:
      open-of-first-session-bar  →  close-of-last-session-bar

    Returns a DataFrame (one row per date):
      date, open_px, close_px, ret_pts, ret_pct, direction (1/-1/0)

    This is the raw directional bias of the session — no entry logic yet.
    A bull% far from 50% is the signal that a session *might* have edge.

    Parameters
    ----------
    utc_hour_from : int or None. If set, only bars with time_utc.hour >= value.
    utc_hour_to   : int or None. If set, only bars with time_utc.hour <= value.
    Both default to None (full session, unchanged behaviour).
    """
    mask = df['session_label'] == session
    sub  = df[mask].copy()
    if utc_hour_from is not None:
        sub = sub[sub['time_utc'].dt.hour >= utc_hour_from]
    if utc_hour_to is not None:
        sub = sub[sub['time_utc'].dt.hour <= utc_hour_to]
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
    include_dates=None,
    exclude_dates=None,
    utc_hour_from: int = None,
    utc_hour_to: int = None,
) -> pd.DataFrame:
    """
    Simulate a fixed-window session trade: enter at session open, exit at
    session close.  One trade per day.  No trailing, no BE, no filters.

    Spread cost applied once at entry (same convention as backtest_runner).
    Commission = 0 (US500/USTEC have no RT commission at this broker).

    Parameters
    ----------
    df            : labeled DataFrame (must have session_label + time_utc columns).
    session       : session label to trade (e.g. 'EU_OPEN').
    direction     : 1 = LONG, -1 = SHORT.
    sp            : spread in price points.
    lots          : lot size (use ml = minimum lot for conservative baseline).
    cs            : contract size (value per point per lot).
    cap_start     : starting capital for this asset slice.
    date_start, date_end : optional date range filter (inclusive / exclusive).
    include_dates : optional set/frozenset of datetime.date — keep only these dates.
    exclude_dates : optional set/frozenset of datetime.date — skip these dates.
    utc_hour_from, utc_hour_to : optional intra-session UTC hour bounds (inclusive).

    Returns
    -------
    DataFrame of trades:
      date, entry, exit, gross, cost, net, result ('WIN'/'LOSS'/'BE'), cap
    """
    daily = session_return_stats(df, session,
                                 utc_hour_from=utc_hour_from,
                                 utc_hour_to=utc_hour_to)
    if daily.empty:
        return pd.DataFrame()

    if date_start is not None:
        daily = daily[pd.to_datetime(daily['date']) >= date_start]
    if date_end is not None:
        daily = daily[pd.to_datetime(daily['date']) < date_end]
    if include_dates is not None:
        daily = daily[daily['date'].isin(include_dates)]
    if exclude_dates is not None:
        daily = daily[~daily['date'].isin(exclude_dates)]
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

    print("\n✅  Research baseline completo.")

    # ── US_MID_LONG conditioning ──────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  US_MID_LONG — conditioning (TEST only)")
    print("═"*65)
    print("  Variants: baseline | HIGH_VOL | LOW_VOL | NO_FOMC | HIGH_VOL+NO_FOMC")
    print(f"  FOMC dates in set: {len(_FOMC_DATES)}")
    print(f"  Vol-regime window: 60 trading days (atr14 daily mean)\n")

    for asset in RESEARCH_ASSETS:
        ap = ASSET_COST_PARAMS.get(asset)
        if ap is None:
            continue
        digits = ASSET_DIGITS.get(asset, 2)
        result = load_price_data(asset, DATA_DIR, digits)
        if result[0] is None:
            continue
        df, _lr, _vm, _qc = result
        df = label_sessions(df, MT5_TZ)

        # Compute vol regime for this asset (full history for stable rolling median)
        regime = compute_daily_vol_regime(df, window=60)
        high_dates = frozenset(regime[regime == 'HIGH_VOL'].index)
        low_dates  = frozenset(regime[regime == 'LOW_VOL'].index)

        print(f"── {asset}  (lots={ap['ml']}, sp={ap['sp']}) ──────────────────")

        variants = [
            ('baseline',           None,       None),
            ('HIGH_VOL',           high_dates, None),
            ('LOW_VOL',            low_dates,  None),
            ('NO_FOMC',            None,       _FOMC_DATES),
            ('HIGH_VOL+NO_FOMC',   high_dates, _FOMC_DATES),
        ]

        for vname, inc, exc in variants:
            trades = run_fixed_window_baseline(
                df,
                session       = 'US_MID',
                direction     = 1,
                sp            = ap['sp'],
                lots          = ap['ml'],
                cs            = ap['cs'],
                cap_start     = 250.0,
                date_start    = _TEST_START,
                date_end      = None,
                include_dates = inc,
                exclude_dates = exc,
            )
            m = _baseline_metrics(trades, 250.0)
            _print_baseline_report(m, f"US_MID_LONG | {vname:<20s} | TEST")
        print()

        # ── LOW_VOL robustness sweep ──────────────────────────────────────
        # Perturbations around the current LOW_VOL definition (p50, w60).
        # Varies percentile threshold and rolling window independently.
        # All variants run US_MID_LONG LONG on TEST period, same costs.
        # Compact tabular format: one line per variant for easy comparison.
        print(f"── {asset}  LOW_VOL robustness sweep (TEST) {'─'*20}")
        hdr = f"  {'variant':<22}  {'n':>4}  {'WR%':>5}  {'PF':>6}  {'exp':>7}  {'ret%':>7}  {'MDD%':>7}"
        print(hdr)
        print(f"  {'-'*67}")

        rob_variants = [
            # (label,            percentile, window)
            ('baseline (no filt)',  None,  None),
            ('p50 w60  [current]',  50,    60),
            ('p25 w60  [strict]',   25,    60),
            ('p75 w60  [lax]',      75,    60),
            ('p50 w30  [s-window]', 50,    30),
            ('p50 w90  [l-window]', 50,    90),
        ]

        for vname, pct, win in rob_variants:
            if pct is None:
                inc = None
            else:
                reg = compute_daily_vol_regime(df, window=win, percentile=pct)
                inc = frozenset(reg[reg == 'LOW_VOL'].index)
            trades = run_fixed_window_baseline(
                df,
                session       = 'US_MID',
                direction     = 1,
                sp            = ap['sp'],
                lots          = ap['ml'],
                cs            = ap['cs'],
                cap_start     = 250.0,
                date_start    = _TEST_START,
                date_end      = None,
                include_dates = inc,
            )
            m = _baseline_metrics(trades, 250.0)
            if m:
                pf_s = f"{m['pf']:.3f}" if m['pf'] != float('inf') else "  inf"
                print(
                    f"  {vname:<22}  {m['n']:>4}  {m['wr']:>4.1f}%  "
                    f"{pf_s:>6}  {m['exp']:>+7.4f}  "
                    f"{m['ret_pct']:>+7.2f}%  {m['mdd']:>7.2f}%"
                )
            else:
                print(f"  {vname:<22}  — no trades")
        print()

        # ── Consolidation: baseline vs LOW_VOL(_PRIMARY_PCT, _PRIMARY_WIN) ──
        # This is the primary research filter going forward.
        # Shows FULL/TEST summary then year-by-year to check temporal stability.
        lv_primary = frozenset(
            compute_daily_vol_regime(df, window=_PRIMARY_WIN, percentile=_PRIMARY_PCT)
            .pipe(lambda s: s[s == 'LOW_VOL'].index)
        )

        print(f"── {asset}  CONSOLIDATION  baseline vs LOW_VOL(p{_PRIMARY_PCT},w{_PRIMARY_WIN}) ──")

        # FULL / TEST summary — compact table, both variants
        chdr = (f"  {'variant':<18}  {'per':<5}  {'n':>4}  {'WR%':>5}  "
                f"{'PF':>6}  {'exp':>7}  {'ret%':>7}  {'MDD%':>7}")
        print(chdr)
        print(f"  {'-'*72}")
        for v_lbl, inc in [('baseline', None), (f'LOW_VOL p{_PRIMARY_PCT}w{_PRIMARY_WIN}', lv_primary)]:
            for per_lbl, ds, de in [('FULL', None, _TRAIN_END), ('TEST', _TEST_START, None)]:
                t = run_fixed_window_baseline(
                    df, session='US_MID', direction=1,
                    sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                    cap_start=250.0, date_start=ds, date_end=de,
                    include_dates=inc,
                )
                m = _baseline_metrics(t, 250.0)
                if m:
                    pf_s = f"{m['pf']:.3f}" if m['pf'] != float('inf') else "  inf"
                    print(
                        f"  {v_lbl:<18}  {per_lbl:<5}  {m['n']:>4}  {m['wr']:>4.1f}%  "
                        f"{pf_s:>6}  {m['exp']:>+7.4f}  "
                        f"{m['ret_pct']:>+7.2f}%  {m['mdd']:>7.2f}%"
                    )
                else:
                    print(f"  {v_lbl:<18}  {per_lbl:<5}  — no trades")
        print()

        # Year-by-year breakdown — baseline vs LOW_VOL side by side
        # Answers: is the improvement consistent or concentrated in one period?
        print(f"  {'year':<6}  "
              f"{'│':1}  {'base_n':>6}  {'base_PF':>7}  {'base_ret%':>9}  "
              f"{'│':1}  {'lv_n':>4}  {'lv_PF':>7}  {'lv_ret%':>8}  {'lv_MDD%':>8}")
        print(f"  {'-'*73}")
        for yr in range(2018, 2027):
            ds_y = pd.Timestamp(f'{yr}-01-01')
            de_y = pd.Timestamp(f'{yr+1}-01-01')
            t_base = run_fixed_window_baseline(
                df, session='US_MID', direction=1,
                sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                cap_start=250.0, date_start=ds_y, date_end=de_y,
            )
            t_lv = run_fixed_window_baseline(
                df, session='US_MID', direction=1,
                sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                cap_start=250.0, date_start=ds_y, date_end=de_y,
                include_dates=lv_primary,
            )
            mb = _baseline_metrics(t_base, 250.0)
            ml = _baseline_metrics(t_lv,   250.0)
            if not mb and not ml:
                continue    # year has no data
            pf_b  = f"{mb['pf']:.3f}"        if mb and mb['pf'] != float('inf') else '  inf'
            ret_b = f"{mb['ret_pct']:>+8.2f}%" if mb else '       -'
            n_b   = mb['n']                   if mb else 0
            pf_l  = f"{ml['pf']:.3f}"        if ml and ml['pf'] != float('inf') else '  inf'
            ret_l = f"{ml['ret_pct']:>+7.2f}%" if ml else '      -'
            mdd_l = f"{ml['mdd']:>7.2f}%"    if ml else '     -'
            n_l   = ml['n']                   if ml else 0
            print(
                f"  {yr:<6}  │  {n_b:>6}  {pf_b:>7}  {ret_b:>9}  "
                f"│  {n_l:>4}  {pf_l:>7}  {ret_l:>8}  {mdd_l:>8}"
            )
        print()

        # ── US_MID timing refinement ─────────────────────────────────────
        # US_MID = 16:00–19:59 UTC.  Split into two equal halves:
        #   early (16-17h UTC) = 12:00-13:59 ET  — US midday
        #   late  (18-19h UTC) = 14:00-15:59 ET  — US afternoon / power hour
        # Both variants × baseline and LOW_VOL(p50,w90) × TEST only.
        print(f"── {asset}  US_MID timing refinement (TEST) {'─'*20}")
        thdr = (f"  {'variant':<20}  {'window':<16}  {'n':>4}  {'WR%':>5}  "
                f"{'PF':>6}  {'exp':>7}  {'ret%':>7}  {'MDD%':>7}")
        print(thdr)
        print(f"  {'-'*76}")

        time_windows = [
            ('full   16-19h', None, None),
            ('early  16-17h', 16,   17),
            ('late   18-19h', 18,   19),
        ]
        for v_lbl, inc in [('baseline', None), (f'LOW_VOL p{_PRIMARY_PCT}w{_PRIMARY_WIN}', lv_primary)]:
            for win_lbl, hfrom, hto in time_windows:
                t = run_fixed_window_baseline(
                    df, session='US_MID', direction=1,
                    sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                    cap_start=250.0, date_start=_TEST_START, date_end=None,
                    include_dates=inc,
                    utc_hour_from=hfrom, utc_hour_to=hto,
                )
                m = _baseline_metrics(t, 250.0)
                if m:
                    pf_s = f"{m['pf']:.3f}" if m['pf'] != float('inf') else "  inf"
                    print(
                        f"  {v_lbl:<20}  {win_lbl:<16}  {m['n']:>4}  {m['wr']:>4.1f}%  "
                        f"{pf_s:>6}  {m['exp']:>+7.4f}  "
                        f"{m['ret_pct']:>+7.2f}%  {m['mdd']:>7.2f}%"
                    )
                else:
                    print(f"  {v_lbl:<20}  {win_lbl:<16}  — no trades")
            print()

        # ── Asset-specific primary hypothesis ─────────────────────────────
        # Each asset now has its own independent primary baseline.
        # This section makes the divergence explicit and is the reference
        # for any future live hypothesis testing.
        #
        #   US500 → US_MID full (16-19h UTC) + LOW_VOL(p50,w90)
        #   USTEC → US_MID early (16-17h UTC) + LOW_VOL(p50,w90)
        #
        # Comparison: [A] shared US_MID full baseline (no filter)
        #             [B] asset-specific window (no filter)   ← only differs for USTEC
        #             [C] asset-specific window + LOW_VOL     ← primary hypothesis
        pcfg = ASSET_PRIMARY_CONFIG.get(asset)
        if pcfg is None:
            continue

        hfrom = pcfg['utc_hour_from']
        hto   = pcfg['utc_hour_to']
        wlbl  = pcfg['window_label']

        print(f"── {asset}  ASSET-SPECIFIC PRIMARY HYPOTHESIS {'─'*18}")
        print(f"  Primary window : {wlbl}")
        print(f"  Filter         : LOW_VOL (p{_PRIMARY_PCT}, w{_PRIMARY_WIN})")
        print(f"  Reference      : US_MID full baseline (no filter, no window restriction)")
        print()

        # FULL + TEST summary table
        ahdr = (f"  {'variant':<32}  {'per':<5}  {'n':>4}  {'WR%':>5}  "
                f"{'PF':>6}  {'exp':>7}  {'ret%':>7}  {'MDD%':>7}")
        print(ahdr)
        print(f"  {'-'*80}")

        asset_variants = [
            # (label,                       include,    hfrom, hto)
            ('US_MID full  [shared ref]',   None,       None,  None),
            (f'{wlbl} [no filt]',           None,       hfrom, hto),
            (f'{wlbl} + LOW_VOL [PRIMARY]', lv_primary, hfrom, hto),
        ]

        for per_lbl, ds, de in [('FULL', None, _TRAIN_END), ('TEST', _TEST_START, None)]:
            for av_lbl, inc, ahf, aht in asset_variants:
                t = run_fixed_window_baseline(
                    df, session='US_MID', direction=1,
                    sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                    cap_start=250.0, date_start=ds, date_end=de,
                    include_dates=inc,
                    utc_hour_from=ahf, utc_hour_to=aht,
                )
                m = _baseline_metrics(t, 250.0)
                if m:
                    pf_s = f"{m['pf']:.3f}" if m['pf'] != float('inf') else "  inf"
                    print(
                        f"  {av_lbl:<32}  {per_lbl:<5}  {m['n']:>4}  {m['wr']:>4.1f}%  "
                        f"{pf_s:>6}  {m['exp']:>+7.4f}  "
                        f"{m['ret_pct']:>+7.2f}%  {m['mdd']:>7.2f}%"
                    )
                else:
                    print(f"  {av_lbl:<32}  {per_lbl:<5}  — no trades")
            print()

        # Year-by-year breakdown — primary hypothesis only (asset-specific window + LOW_VOL)
        # Answers: is the combined filter + window restriction stable across years?
        print(f"  Year-by-year — PRIMARY hypothesis ({wlbl} + LOW_VOL p{_PRIMARY_PCT}w{_PRIMARY_WIN})")
        print(f"  {'year':<6}  {'n':>4}  {'WR%':>5}  {'PF':>6}  {'exp':>7}  {'ret%':>7}  {'MDD%':>7}")
        print(f"  {'-'*52}")
        for yr in range(2018, 2027):
            ds_y = pd.Timestamp(f'{yr}-01-01')
            de_y = pd.Timestamp(f'{yr+1}-01-01')
            t_prim = run_fixed_window_baseline(
                df, session='US_MID', direction=1,
                sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                cap_start=250.0, date_start=ds_y, date_end=de_y,
                include_dates=lv_primary,
                utc_hour_from=hfrom, utc_hour_to=hto,
            )
            mp = _baseline_metrics(t_prim, 250.0)
            if not mp:
                continue
            pf_s = f"{mp['pf']:.3f}" if mp['pf'] != float('inf') else "  inf"
            print(
                f"  {yr:<6}  {mp['n']:>4}  {mp['wr']:>4.1f}%  "
                f"{pf_s:>6}  {mp['exp']:>+7.4f}  "
                f"{mp['ret_pct']:>+7.2f}%  {mp['mdd']:>7.2f}%"
            )
        print()

        # ── Regime diagnostic — 2022 vs rest ─────────────────────────────
        # Structural audit: what distinguishes the problem regime from good years?
        #
        # Regime quality columns (from df, no trades needed):
        #   n_lv     : LOW_VOL days available in year (filter selectivity)
        #   lv%      : LOW_VOL days / all trading days in year
        #   atr_lv   : mean daily ATR on selected LOW_VOL days
        #              → key question: was "calm" in 2022 actually calm vs other years?
        #   atr_all  : mean daily ATR across all days in year
        #              → overall regime level, independent of filter
        #
        # Trade performance columns (from trades on primary hypothesis):
        #   n        : trades taken (session had data on that LOW_VOL day)
        #   WR%      : win rate
        #   aw       : average win (signed +)
        #   al       : average loss (signed −)
        #   PF       : profit factor
        #   ret%     : period return on $250 starting capital
        #
        # std_net in aggregate table: per-trade net P&L standard deviation
        #   → was 2022 more erratic (high variance) or just directionally wrong?
        print(f"── {asset}  REGIME DIAGNOSTIC — {wlbl} + LOW_VOL(p{_PRIMARY_PCT},w{_PRIMARY_WIN}) ──")
        print(f"  Regime quality vs trade performance per year.  ◄ = problem year 2022")
        print()

        daily_atr_s  = df.groupby('date')['atr14'].mean()   # date → mean ATR
        all_dates_set = frozenset(daily_atr_s.index)

        # Run full-history primary trades once (no date boundary, just the primary filter)
        trades_diag = run_fixed_window_baseline(
            df, session='US_MID', direction=1,
            sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
            cap_start=250.0,
            include_dates=lv_primary,
            utc_hour_from=hfrom, utc_hour_to=hto,
        )

        # ── Table 1: year-by-year regime + performance ────────────────────
        dhdr = (
            f"  {'year':<6}  │  {'n_lv':>4}  {'lv%':>5}  {'atr_lv':>7}  {'atr_all':>7}  "
            f"│  {'n':>4}  {'WR%':>5}  {'aw':>7}  {'al':>7}  {'PF':>6}  {'ret%':>7}"
        )
        print(dhdr)
        print(f"  {'-'*85}")

        for yr in range(2018, 2027):
            yr_dates  = {d for d in all_dates_set if d.year == yr}
            lv_yr     = {d for d in lv_primary     if d.year == yr}
            n_all_yr  = len(yr_dates)
            n_lv_yr   = len(lv_yr)
            lv_pct_yr = n_lv_yr / n_all_yr * 100 if n_all_yr else float('nan')

            atr_lv_yr  = (float(daily_atr_s.loc[list(lv_yr)].mean())
                          if lv_yr else float('nan'))
            atr_all_yr = (float(daily_atr_s.loc[list(yr_dates)].mean())
                          if yr_dates else float('nan'))

            if not trades_diag.empty:
                t_yr = trades_diag[
                    trades_diag['date'].apply(lambda d: d.year) == yr
                ].reset_index(drop=True)
            else:
                t_yr = pd.DataFrame()

            atr_lv_s  = f"{atr_lv_yr:>7.2f}"  if not np.isnan(atr_lv_yr)  else "      —"
            atr_all_s = f"{atr_all_yr:>7.2f}" if not np.isnan(atr_all_yr) else "      —"
            marker    = "  ◄" if yr == 2022 else ""

            if t_yr.empty:
                print(
                    f"  {yr:<6}  │  {n_lv_yr:>4}  {lv_pct_yr:>4.0f}%  "
                    f"{atr_lv_s}  {atr_all_s}  │  "
                    f"{'—':>4}  {'—':>5}  {'—':>7}  {'—':>7}  {'—':>6}  {'—':>7}{marker}"
                )
                continue

            m    = _baseline_metrics(t_yr, 250.0)
            pf_s = f"{m['pf']:.3f}" if m['pf'] != float('inf') else "  inf"
            print(
                f"  {yr:<6}  │  {n_lv_yr:>4}  {lv_pct_yr:>4.0f}%  "
                f"{atr_lv_s}  {atr_all_s}  │  "
                f"{m['n']:>4}  {m['wr']:>4.1f}%  "
                f"{m['aw']:>+7.4f}  {m['al']:>+7.4f}  "
                f"{pf_s:>6}  {m['ret_pct']:>+6.2f}%{marker}"
            )
        print()

        # ── Table 2: 2022 vs rest aggregate ───────────────────────────────
        print(f"  Aggregate: rest vs 2022")
        ahdr = (
            f"  {'period':<8}  {'n_lv':>4}  {'lv%':>5}  {'atr_lv':>7}  "
            f"{'n':>4}  {'WR%':>5}  {'aw':>7}  {'al':>7}  "
            f"{'PF':>6}  {'exp':>7}  {'std_net':>8}"
        )
        print(ahdr)
        print(f"  {'-'*79}")

        for period_lbl, is_2022 in [('rest', False), ('2022', True)]:
            p_dates   = {d for d in all_dates_set if (d.year == 2022) == is_2022}
            lv_p      = {d for d in lv_primary     if (d.year == 2022) == is_2022}
            n_all_p   = len(p_dates)
            n_lv_p    = len(lv_p)
            lv_pct_p  = n_lv_p / n_all_p * 100 if n_all_p else float('nan')
            atr_lv_p  = (float(daily_atr_s.loc[list(lv_p)].mean())
                         if lv_p else float('nan'))

            if not trades_diag.empty:
                t_p = trades_diag[
                    trades_diag['date'].apply(lambda d: (d.year == 2022) == is_2022)
                ].reset_index(drop=True)
            else:
                t_p = pd.DataFrame()

            atr_lv_ps = f"{atr_lv_p:>7.2f}" if not np.isnan(atr_lv_p) else "      —"

            if t_p.empty:
                print(f"  {period_lbl:<8}  — no trades")
                continue

            m_p     = _baseline_metrics(t_p, 250.0)
            std_net = float(t_p['net'].std()) if len(t_p) > 1 else float('nan')
            pf_s_p  = f"{m_p['pf']:.3f}" if m_p['pf'] != float('inf') else "  inf"
            std_s   = f"{std_net:>8.4f}" if not np.isnan(std_net) else "       —"
            print(
                f"  {period_lbl:<8}  {n_lv_p:>4}  {lv_pct_p:>4.0f}%  {atr_lv_ps}  "
                f"{m_p['n']:>4}  {m_p['wr']:>4.1f}%  "
                f"{m_p['aw']:>+7.4f}  {m_p['al']:>+7.4f}  "
                f"{pf_s_p:>6}  {m_p['exp']:>+7.4f}  {std_s}"
            )
        print()

        # ── LOW_VOL comparability audit ───────────────────────────────────
        # Question: does LOW_VOL(p50,w90) select "relatively calm within the year"
        # (local, adaptive) or "absolutely calm across all years" (global, stable)?
        #
        # We expose the rolling threshold used internally by compute_daily_vol_regime
        # and compare it year-by-year against the actual ATR of selected LOW_VOL days.
        #
        # Columns:
        #   thr_mean : mean rolling-p50 threshold across the year's trading days
        #              → if much higher in 2022, the filter adapted upward to match
        #                the inflated vol environment, selecting days that are locally
        #                "calm" but globally elevated
        #   atr_lv   : mean daily ATR on selected LOW_VOL days
        #   ratio    : atr_lv / thr_mean  (always ≤ 1.0 by construction)
        #              → close to 1.0 = selected days are near the threshold = not very calm
        #   p25/p50/p75 : ATR percentiles of the selected LOW_VOL days
        #              → compare p50 across years: if 2022 p50 > rest p75, the filter
        #                lost cross-year comparability
        print(f"── {asset}  LOW_VOL COMPARABILITY AUDIT  (p{_PRIMARY_PCT}, w{_PRIMARY_WIN}) ───────")
        print(f"  Is LOW_VOL selecting 'locally calm' or 'globally comparable' days?")
        print()

        # One new computation: the rolling threshold series (same formula as
        # compute_daily_vol_regime, but we keep the threshold values, not just labels).
        # daily_atr_s and all_dates_set are already in scope from the REGIME DIAGNOSTIC block.
        rolling_thr = daily_atr_s.rolling(_PRIMARY_WIN, min_periods=_PRIMARY_WIN // 2).quantile(
            _PRIMARY_PCT / 100.0
        )

        # Format helpers (local, diagnostic-only)
        def _fw(v, w=7):
            return f"{v:{w}.2f}" if not np.isnan(v) else (' ' * (w - 1) + '—')
        def _fr(v):
            return f"{v:5.3f}" if not np.isnan(v) else '    —'

        cahdr = (
            f"  {'year':<6}  {'n_lv':>4}  {'lv%':>5}  "
            f"{'thr_mean':>8}  {'atr_lv':>7}  {'ratio':>6}  "
            f"{'p25':>7}  {'p50':>7}  {'p75':>7}"
        )
        print(cahdr)
        print(f"  {'-'*72}")

        for yr in range(2018, 2027):
            yr_dates  = {d for d in all_dates_set if d.year == yr}
            lv_yr     = {d for d in lv_primary     if d.year == yr}
            n_all_yr  = len(yr_dates)
            n_lv_yr   = len(lv_yr)
            lv_pct_yr = n_lv_yr / n_all_yr * 100 if n_all_yr else float('nan')

            thr_vals = rolling_thr.loc[list(yr_dates)].dropna()
            thr_mean = float(thr_vals.mean()) if len(thr_vals) else float('nan')

            if lv_yr:
                atr_lv_vals = daily_atr_s.loc[list(lv_yr)]
                atr_lv_m    = float(atr_lv_vals.mean())
                p25         = float(atr_lv_vals.quantile(0.25))
                p50         = float(atr_lv_vals.quantile(0.50))
                p75         = float(atr_lv_vals.quantile(0.75))
                ratio       = (atr_lv_m / thr_mean
                               if not np.isnan(thr_mean) and thr_mean > 0
                               else float('nan'))
            else:
                atr_lv_m = p25 = p50 = p75 = ratio = float('nan')

            marker = "  ◄" if yr == 2022 else ""
            print(
                f"  {yr:<6}  {n_lv_yr:>4}  {lv_pct_yr:>4.0f}%  "
                f"{_fw(thr_mean, 8)}  {_fw(atr_lv_m)}  {_fr(ratio)}  "
                f"{_fw(p25)}  {_fw(p50)}  {_fw(p75)}{marker}"
            )
        print()

        # Aggregate: rest vs 2022 — pool all LOW_VOL days for each group
        # This directly answers: "is 2022 p50 above rest p75?"
        print(f"  Aggregate: rest vs 2022  (pooled LOW_VOL days)")
        aagg_hdr = (
            f"  {'period':<8}  {'n_lv':>4}  {'lv%':>5}  "
            f"{'thr_mean':>8}  {'atr_lv':>7}  {'ratio':>6}  "
            f"{'p25':>7}  {'p50':>7}  {'p75':>7}"
        )
        print(aagg_hdr)
        print(f"  {'-'*72}")

        for period_lbl, is_2022 in [('rest', False), ('2022', True)]:
            p_dates  = {d for d in all_dates_set if (d.year == 2022) == is_2022}
            lv_p     = {d for d in lv_primary     if (d.year == 2022) == is_2022}
            n_all_p  = len(p_dates)
            n_lv_p   = len(lv_p)
            lv_pct_p = n_lv_p / n_all_p * 100 if n_all_p else float('nan')

            thr_p_vals = rolling_thr.loc[list(p_dates)].dropna()
            thr_p_mean = float(thr_p_vals.mean()) if len(thr_p_vals) else float('nan')

            if lv_p:
                atr_p_vals = daily_atr_s.loc[list(lv_p)]
                atr_p_m    = float(atr_p_vals.mean())
                pp25       = float(atr_p_vals.quantile(0.25))
                pp50       = float(atr_p_vals.quantile(0.50))
                pp75       = float(atr_p_vals.quantile(0.75))
                ratio_p    = (atr_p_m / thr_p_mean
                              if not np.isnan(thr_p_mean) and thr_p_mean > 0
                              else float('nan'))
            else:
                atr_p_m = pp25 = pp50 = pp75 = ratio_p = float('nan')

            print(
                f"  {period_lbl:<8}  {n_lv_p:>4}  {lv_pct_p:>4.0f}%  "
                f"{_fw(thr_p_mean, 8)}  {_fw(atr_p_m)}  {_fr(ratio_p)}  "
                f"{_fw(pp25)}  {_fw(pp50)}  {_fw(pp75)}"
            )
        print()

        # ── Absolute-vol guardrail test ───────────────────────────────────
        # Causal test: does a simple hard ATR cap on top of LOW_VOL(p50,w90)
        # recover cross-year comparability without destroying good years?
        #
        # Cap thresholds are derived directly from the comparability audit above
        # — no new parameter search, no arbitrary tuning:
        #
        #   cap_lenient = p75 of ATR on LOW_VOL days from non-2022 years
        #   cap_strict  = p50 of ATR on LOW_VOL days from non-2022 years
        #
        # These are the exact percentiles already visible in the audit aggregate
        # table (rest row).  Days in lv_primary whose ATR exceeds the cap are
        # dropped.  The question is whether 2022 falls outside this range while
        # good years are preserved.
        lv_rest     = frozenset(d for d in lv_primary if d.year != 2022)
        atr_rest_s  = daily_atr_s.loc[list(lv_rest)]
        cap_lenient = float(atr_rest_s.quantile(0.75))
        cap_strict  = float(atr_rest_s.quantile(0.50))

        lv_cap_len = frozenset(d for d in lv_primary if daily_atr_s.loc[d] <= cap_lenient)
        lv_cap_str = frozenset(d for d in lv_primary if daily_atr_s.loc[d] <= cap_strict)

        print(f"── {asset}  ABSOLUTE-VOL GUARDRAIL TEST ─────────────────────────────")
        print(f"  Base filter : LOW_VOL (p{_PRIMARY_PCT}, w{_PRIMARY_WIN})  +  {wlbl}")
        print(f"  cap_lenient : ATR ≤ {cap_lenient:.2f}  [p75 of non-2022 LOW_VOL days]"
              f"  n={len(lv_cap_len)}")
        print(f"  cap_strict  : ATR ≤ {cap_strict:.2f}  [p50 of non-2022 LOW_VOL days]"
              f"  n={len(lv_cap_str)}")
        print()

        gvars = [
            ('PRIMARY only',                    lv_primary),
            (f'+ cap_len {cap_lenient:.2f}',    lv_cap_len),
            (f'+ cap_str {cap_strict:.2f}',     lv_cap_str),
        ]

        # ── FULL + TEST summary ───────────────────────────────────────────
        ghdr = (f"  {'variant':<22}  {'per':<5}  {'n':>4}  {'WR%':>5}  "
                f"{'PF':>6}  {'exp':>7}  {'ret%':>7}  {'MDD%':>7}")
        print(ghdr)
        print(f"  {'-'*73}")

        for gv_lbl, inc in gvars:
            for per_lbl, ds, de in [('FULL', None, _TRAIN_END), ('TEST', _TEST_START, None)]:
                t = run_fixed_window_baseline(
                    df, session='US_MID', direction=1,
                    sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                    cap_start=250.0, date_start=ds, date_end=de,
                    include_dates=inc, utc_hour_from=hfrom, utc_hour_to=hto,
                )
                m = _baseline_metrics(t, 250.0)
                if m:
                    pf_s = f"{m['pf']:.3f}" if m['pf'] != float('inf') else "  inf"
                    print(
                        f"  {gv_lbl:<22}  {per_lbl:<5}  {m['n']:>4}  {m['wr']:>4.1f}%  "
                        f"{pf_s:>6}  {m['exp']:>+7.4f}  "
                        f"{m['ret_pct']:>+7.2f}%  {m['mdd']:>7.2f}%"
                    )
                else:
                    print(f"  {gv_lbl:<22}  {per_lbl:<5}  — no trades")
            print()

        # ── Year-by-year compact: 3 variants side by side ─────────────────
        print(f"  Year-by-year  (n | PF | ret%  per variant)")
        yhdr = (
            f"  {'year':<6}  "
            f"│ {'prim_n':>6} {'prim_PF':>7} {'prim_ret%':>9}  "
            f"│ {'len_n':>5} {'len_PF':>7} {'len_ret%':>9}  "
            f"│ {'str_n':>5} {'str_PF':>7} {'str_ret%':>8}"
        )
        print(yhdr)
        print(f"  {'-'*84}")

        for yr in range(2018, 2027):
            ds_y = pd.Timestamp(f'{yr}-01-01')
            de_y = pd.Timestamp(f'{yr+1}-01-01')
            cols = []
            for inc in [lv_primary, lv_cap_len, lv_cap_str]:
                t_y = run_fixed_window_baseline(
                    df, session='US_MID', direction=1,
                    sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                    cap_start=250.0, date_start=ds_y, date_end=de_y,
                    include_dates=inc, utc_hour_from=hfrom, utc_hour_to=hto,
                )
                m_y = _baseline_metrics(t_y, 250.0)
                if m_y:
                    pf_y = f"{m_y['pf']:.3f}" if m_y['pf'] != float('inf') else "  inf"
                    cols.append(f" {m_y['n']:>5} {pf_y:>7} {m_y['ret_pct']:>+8.2f}%")
                else:
                    cols.append(f" {'—':>5} {'—':>7} {'—':>9}")
            marker = "  ◄" if yr == 2022 else ""
            print(f"  {yr:<6}  │{'  │'.join(cols)}{marker}")
        print()

        # ── Aggregate rest vs 2022 ────────────────────────────────────────
        print(f"  Aggregate: rest vs 2022")
        aghdr = (f"  {'period':<6}  {'variant':<22}  {'n':>4}  {'WR%':>5}  "
                 f"{'PF':>6}  {'exp':>7}  {'ret%':>7}")
        print(aghdr)
        print(f"  {'-'*63}")

        for period_lbl, is_2022 in [('rest', False), ('2022', True)]:
            yr_dates_p = {d for d in all_dates_set if (d.year == 2022) == is_2022}
            for gv_lbl, inc in gvars:
                eff_inc = frozenset(d for d in inc if d in yr_dates_p)
                t_agg = run_fixed_window_baseline(
                    df, session='US_MID', direction=1,
                    sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                    cap_start=250.0, include_dates=eff_inc,
                    utc_hour_from=hfrom, utc_hour_to=hto,
                )
                m_agg = _baseline_metrics(t_agg, 250.0)
                if m_agg:
                    pf_s = f"{m_agg['pf']:.3f}" if m_agg['pf'] != float('inf') else "  inf"
                    print(
                        f"  {period_lbl:<6}  {gv_lbl:<22}  {m_agg['n']:>4}  "
                        f"{m_agg['wr']:>4.1f}%  {pf_s:>6}  "
                        f"{m_agg['exp']:>+7.4f}  {m_agg['ret_pct']:>+7.2f}%"
                    )
                else:
                    print(f"  {period_lbl:<6}  {gv_lbl:<22}  — no trades")
            print()

        # ── USTEC regime discrimination: 2022 vs high-ATR good years ──────
        # Both 2022 (atr_lv=8.61, PF=0.78) and 2025 (atr_lv=8.29, PF=1.62)
        # have similar ATR levels yet opposite outcomes.
        # This block isolates what else differs beyond raw ATR level.
        #
        # Key discriminating metric added: aw/|al| (win/loss asymmetry ratio)
        #   good years: aw ≥ |al|  → ratio ≥ 1.0  (wins at least as large as losses)
        #   2022       : aw < |al|  → ratio < 1.0  (losses disproportionately large)
        #
        # Selected years: control=2020,2023 | bad-high-ATR=2022 | good-high-ATR=2025,2026
        # 2026 flagged: small sample (~11 trades), interpret with caution.
        if asset == 'USTEC':
            print(f"── USTEC  REGIME DISCRIMINATION — 2022 vs high-ATR good years ─────")
            print(f"  Hypothesis : {wlbl} + LOW_VOL(p{_PRIMARY_PCT},w{_PRIMARY_WIN})")
            print(f"  Question   : does 2025 belong to the same regime as 2022 or not?")
            print(f"  Note       : 2026 small sample (~11 trades) — treat as indicative only.")
            print()

            disc_years = [2020, 2022, 2023, 2025, 2026]

            dhdr2 = (
                f"  {'year':<6}  {'n':>4}  {'WR%':>5}  {'aw':>7}  {'al':>7}  "
                f"{'asym':>6}  {'PF':>6}  {'exp':>7}  {'ret%':>7}  {'MDD%':>6}  "
                f"{'std':>6}  {'atr_lv':>6}  {'thr_mn':>6}"
            )
            print(dhdr2)
            print(f"  {'-'*98}")

            for yr in disc_years:
                ds_y = pd.Timestamp(f'{yr}-01-01')
                de_y = pd.Timestamp(f'{yr+1}-01-01')
                t_y = run_fixed_window_baseline(
                    df, session='US_MID', direction=1,
                    sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                    cap_start=250.0, date_start=ds_y, date_end=de_y,
                    include_dates=lv_primary,
                    utc_hour_from=hfrom, utc_hour_to=hto,
                )
                m_y = _baseline_metrics(t_y, 250.0)
                if not m_y:
                    print(f"  {yr:<6}  — no trades")
                    continue

                std_y  = float(t_y['net'].std()) if len(t_y) > 1 else float('nan')
                al_abs = abs(m_y['al'])
                asym   = m_y['aw'] / al_abs if al_abs > 0 else float('nan')

                lv_yr   = {d for d in lv_primary     if d.year == yr}
                yr_all  = {d for d in all_dates_set   if d.year == yr}
                atr_lv_y = float(daily_atr_s.loc[list(lv_yr)].mean()) if lv_yr  else float('nan')
                thr_yr   = rolling_thr.loc[list(yr_all)].dropna()
                thr_mn_y = float(thr_yr.mean())                        if len(thr_yr) else float('nan')

                pf_s   = f"{m_y['pf']:.3f}"  if m_y['pf'] != float('inf') else "  inf"
                asym_s = f"{asym:>6.3f}"     if not np.isnan(asym)    else "     —"
                std_s  = f"{std_y:>6.2f}"    if not np.isnan(std_y)   else "     —"
                atr_s  = f"{atr_lv_y:>6.2f}" if not np.isnan(atr_lv_y) else "     —"
                thr_s  = f"{thr_mn_y:>6.2f}" if not np.isnan(thr_mn_y) else "     —"
                marker = "  ◄ BAD" if yr == 2022 else (
                         "  (small n)" if yr == 2026 else "")

                print(
                    f"  {yr:<6}  {m_y['n']:>4}  {m_y['wr']:>4.1f}%  "
                    f"{m_y['aw']:>+7.4f}  {m_y['al']:>+7.4f}  "
                    f"{asym_s}  {pf_s:>6}  {m_y['exp']:>+7.4f}  "
                    f"{m_y['ret_pct']:>+6.2f}%  {m_y['mdd']:>6.2f}%  "
                    f"{std_s}  {atr_s}  {thr_s}{marker}"
                )
            print()

            # Bucket summary: pool years into ATR-regime buckets
            # control = lower-ATR good years (2020, 2023)
            # high-ATR bad = 2022
            # high-ATR good = 2025 + 2026 (pooled to compensate small 2026 n)
            print(f"  Bucket summary  (years pooled by ATR regime + outcome)")
            bhdr = (
                f"  {'bucket':<24}  {'years':<10}  {'n':>4}  {'WR%':>5}  "
                f"{'asym':>6}  {'PF':>6}  {'exp':>7}  {'std':>6}"
            )
            print(bhdr)
            print(f"  {'-'*72}")

            buckets = [
                ('control  (low ATR)',   [2020, 2023]),
                ('high-ATR bad',         [2022]),
                ('high-ATR good',        [2025, 2026]),
            ]
            for bkt_lbl, bkt_years in buckets:
                bkt_inc = frozenset(d for d in lv_primary if d.year in set(bkt_years))
                t_bkt   = run_fixed_window_baseline(
                    df, session='US_MID', direction=1,
                    sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                    cap_start=250.0, include_dates=bkt_inc,
                    utc_hour_from=hfrom, utc_hour_to=hto,
                )
                m_bkt = _baseline_metrics(t_bkt, 250.0)
                if not m_bkt:
                    print(f"  {bkt_lbl:<24}  — no trades")
                    continue
                std_b  = float(t_bkt['net'].std()) if len(t_bkt) > 1 else float('nan')
                al_b   = abs(m_bkt['al'])
                asym_b = m_bkt['aw'] / al_b if al_b > 0 else float('nan')
                pf_sb  = f"{m_bkt['pf']:.3f}" if m_bkt['pf'] != float('inf') else "  inf"
                asym_sb = f"{asym_b:>6.3f}"   if not np.isnan(asym_b) else "     —"
                std_sb  = f"{std_b:>6.2f}"    if not np.isnan(std_b)  else "     —"
                yr_str  = '+'.join(str(y) for y in bkt_years)
                print(
                    f"  {bkt_lbl:<24}  {yr_str:<10}  {m_bkt['n']:>4}  "
                    f"{m_bkt['wr']:>4.1f}%  {asym_sb}  {pf_sb:>6}  "
                    f"{m_bkt['exp']:>+7.4f}  {std_sb}"
                )
            print()

        # ── USTEC trade-quality discriminator audit ───────────────────────
        # From regime discrimination: 2022 and 2025 both have atr_lv ~8-9,
        # but opposite outcomes.  ATR is not the discriminant.
        #
        # Here we look at the *directional structure* of the session window
        # on LOW_VOL days — derived from raw bars, no new functions:
        #
        #   efficiency = (close - open) / (high - low)
        #     Ranges from -1 to +1.  High positive value → clean directional
        #     move within the window.  Near-zero or negative → whipsaw /
        #     reversal: the market moved then came back.
        #
        #   mfe_mae = (high - open) / (open - low)    [LONG perspective]
        #     > 1.0 → market went further in our favor than against us
        #     < 1.0 → market went further against us (explains al > aw in 2022)
        #     Median used (not mean) to reduce sensitivity to outlier days where
        #     mae → 0 (perfect days with no adverse excursion).
        #
        # If 2022 has lower efficiency and lower mfe_mae vs 2025, the session
        # window in 2022 was structurally worse — not just higher volatility.
        if asset == 'USTEC':
            print(f"── USTEC  TRADE-QUALITY DISCRIMINATOR AUDIT ─────────────────────────")
            print(f"  Window     : {wlbl}")
            print(f"  Filter     : LOW_VOL (p{_PRIMARY_PCT}, w{_PRIMARY_WIN})")
            print(f"  efficiency : (close-open)/(high-low)  — directional purity [-1,+1]")
            print(f"  mfe_mae    : (high-open)/(open-low)   — favorable vs adverse excursion (median)")
            print()

            # Build per-day quality metrics — inline groupby on raw bars
            mask_q = df['session_label'] == 'US_MID'
            sub_q  = df[mask_q].copy()
            if hfrom is not None:
                sub_q = sub_q[sub_q['time_utc'].dt.hour >= hfrom]
            if hto is not None:
                sub_q = sub_q[sub_q['time_utc'].dt.hour <= hto]
            sub_q = sub_q[sub_q['date'].isin(lv_primary)]

            q_rows = []
            for d_q, g_q in sub_q.groupby('date'):
                g_q = g_q.sort_values('time_utc')
                o_q  = float(g_q['open'].iloc[0])
                c_q  = float(g_q['close'].iloc[-1])
                h_q  = float(g_q['high'].max())
                l_q  = float(g_q['low'].min())
                rng  = h_q - l_q
                mfe  = max(h_q - o_q, 0.0)
                mae  = max(o_q - l_q, 0.0)
                q_rows.append({
                    'date':       d_q,
                    'year':       d_q.year,
                    'efficiency': (c_q - o_q) / rng if rng > 1e-9 else float('nan'),
                    'mfe_mae':    mfe / mae           if mae > 1.0  else float('nan'),
                })
            qual_df = pd.DataFrame(q_rows)

            disc_years = [2020, 2022, 2023, 2025, 2026]
            qhdr = (
                f"  {'year':<6}  {'n':>4}  {'WR%':>5}  {'PF':>6}  {'exp':>7}  "
                f"{'efficiency':>10}  {'mfe_mae':>8}"
            )
            print(qhdr)
            print(f"  {'-'*62}")

            for yr in disc_years:
                ds_y = pd.Timestamp(f'{yr}-01-01')
                de_y = pd.Timestamp(f'{yr+1}-01-01')
                t_y  = run_fixed_window_baseline(
                    df, session='US_MID', direction=1,
                    sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                    cap_start=250.0, date_start=ds_y, date_end=de_y,
                    include_dates=lv_primary,
                    utc_hour_from=hfrom, utc_hour_to=hto,
                )
                m_y  = _baseline_metrics(t_y, 250.0)
                yr_q = qual_df[qual_df['year'] == yr]
                eff_m = float(yr_q['efficiency'].mean())   if not yr_q.empty else float('nan')
                mm_m  = float(yr_q['mfe_mae'].median())    if not yr_q.empty else float('nan')

                pf_s  = f"{m_y['pf']:.3f}"  if m_y and m_y['pf'] != float('inf') else "    —"
                eff_s = f"{eff_m:>10.4f}"   if not np.isnan(eff_m) else "         —"
                mm_s  = f"{mm_m:>8.3f}"     if not np.isnan(mm_m)  else "       —"
                wr_s  = f"{m_y['wr']:>4.1f}%" if m_y else "    —"
                exp_s = f"{m_y['exp']:>+7.4f}" if m_y else "      —"
                marker = "  ◄ BAD" if yr == 2022 else ("  (small n)" if yr == 2026 else "")

                print(
                    f"  {yr:<6}  {m_y['n'] if m_y else 0:>4}  {wr_s}  {pf_s:>6}  "
                    f"{exp_s}  {eff_s}  {mm_s}{marker}"
                )
            print()

            # Bucket quality summary — same buckets as discrimination section
            print(f"  Bucket quality summary")
            bqhdr = (
                f"  {'bucket':<24}  {'years':<10}  {'n':>4}  "
                f"{'efficiency':>10}  {'mfe_mae':>8}  {'WR%':>5}  {'PF':>6}"
            )
            print(bqhdr)
            print(f"  {'-'*72}")

            for bkt_lbl, bkt_years in [
                ('control  (low ATR)',  [2020, 2023]),
                ('high-ATR bad',        [2022]),
                ('high-ATR good',       [2025, 2026]),
            ]:
                bkt_q   = qual_df[qual_df['year'].isin(bkt_years)]
                bkt_inc = frozenset(d for d in lv_primary if d.year in set(bkt_years))
                t_bkt   = run_fixed_window_baseline(
                    df, session='US_MID', direction=1,
                    sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                    cap_start=250.0, include_dates=bkt_inc,
                    utc_hour_from=hfrom, utc_hour_to=hto,
                )
                m_bkt  = _baseline_metrics(t_bkt, 250.0)
                eff_b  = float(bkt_q['efficiency'].mean())  if not bkt_q.empty else float('nan')
                mm_b   = float(bkt_q['mfe_mae'].median())   if not bkt_q.empty else float('nan')
                pf_sb  = f"{m_bkt['pf']:.3f}"  if m_bkt and m_bkt['pf'] != float('inf') else "    —"
                eff_sb = f"{eff_b:>10.4f}"     if not np.isnan(eff_b) else "         —"
                mm_sb  = f"{mm_b:>8.3f}"       if not np.isnan(mm_b)  else "       —"
                wr_sb  = f"{m_bkt['wr']:>4.1f}%" if m_bkt else "    —"
                yr_str = '+'.join(str(y) for y in bkt_years)
                print(
                    f"  {bkt_lbl:<24}  {yr_str:<10}  {len(bkt_q):>4}  "
                    f"{eff_sb}  {mm_sb}  {wr_sb}  {pf_sb:>6}"
                )
            print()

            # ── USTEC rolling trade-quality filter test ───────────────────────
            # Causal test: does a look-ahead-free rolling quality guard improve
            # USTEC primary without destroying 2025?
            #
            # Signal construction — no look-ahead:
            #   qual_df already has per-day efficiency and mfe_mae.
            #   shift(1) ensures day t uses only days 0..t-1.
            #   min_periods = N//2 requires at least half the window before
            #   the filter activates (early dates excluded conservatively).
            #
            # N_ROLL = 20 eligible LOW_VOL sessions ≈ 4-5 calendar weeks.
            #   Short enough to detect within-year regime shifts.
            #   Long enough to avoid reacting to single noisy days.
            #
            # Thresholds — natural break-even points from observed data:
            #   eff_roll  > 0.0  : net directional move was positive on average
            #   mfe_roll  > 1.0  : favorable excursion exceeded adverse excursion
            # Neither threshold is tuned to any specific year.
            N_ROLL = 20
            qual_s = qual_df.sort_values('date').reset_index(drop=True)
            qual_s['eff_roll'] = (
                qual_s['efficiency']
                .shift(1)
                .rolling(N_ROLL, min_periods=N_ROLL // 2)
                .mean()
            )
            qual_s['mfe_roll'] = (
                qual_s['mfe_mae']
                .shift(1)
                .rolling(N_ROLL, min_periods=N_ROLL // 2)
                .median()
            )

            eff_pass = frozenset(qual_s.loc[qual_s['eff_roll'] > 0.0, 'date'])
            mfe_pass = frozenset(qual_s.loc[qual_s['mfe_roll'] > 1.0, 'date'])
            lv_eff   = frozenset(d for d in lv_primary if d in eff_pass)
            lv_mfe   = frozenset(d for d in lv_primary if d in mfe_pass)

            print(f"── USTEC  ROLLING TRADE-QUALITY FILTER TEST ────────────────────────")
            print(f"  Rolling N={N_ROLL} LOW_VOL sessions | shift(1) | no look-ahead")
            print(f"  eff_roll  > 0.0  (rolling mean efficiency)   "
                  f"n={len(lv_eff):4d} / {len(lv_primary)}")
            print(f"  mfe_roll  > 1.0  (rolling median mfe_mae)    "
                  f"n={len(lv_mfe):4d} / {len(lv_primary)}")
            print()

            rvars = [
                ('PRIMARY only',      lv_primary),
                ('+ roll_eff  > 0.0', lv_eff),
                ('+ roll_mfe  > 1.0', lv_mfe),
            ]

            # ── FULL + TEST summary ───────────────────────────────────────────
            rhdr = (f"  {'variant':<22}  {'per':<5}  {'n':>4}  {'WR%':>5}  "
                    f"{'PF':>6}  {'exp':>7}  {'ret%':>7}  {'MDD%':>7}")
            print(rhdr)
            print(f"  {'-'*72}")

            for rv_lbl, inc in rvars:
                for per_lbl, ds, de in [('FULL', None, _TRAIN_END), ('TEST', _TEST_START, None)]:
                    t = run_fixed_window_baseline(
                        df, session='US_MID', direction=1,
                        sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                        cap_start=250.0, date_start=ds, date_end=de,
                        include_dates=inc, utc_hour_from=hfrom, utc_hour_to=hto,
                    )
                    m = _baseline_metrics(t, 250.0)
                    if m:
                        pf_s = f"{m['pf']:.3f}" if m['pf'] != float('inf') else "  inf"
                        print(
                            f"  {rv_lbl:<22}  {per_lbl:<5}  {m['n']:>4}  {m['wr']:>4.1f}%  "
                            f"{pf_s:>6}  {m['exp']:>+7.4f}  "
                            f"{m['ret_pct']:>+7.2f}%  {m['mdd']:>7.2f}%"
                        )
                    else:
                        print(f"  {rv_lbl:<22}  {per_lbl:<5}  — no trades")
                print()

            # ── Selected years: focus on 2022 vs 2025 ────────────────────────
            print(f"  Selected years  (n | PF | ret%  per variant)")
            syhdr = (
                f"  {'year':<6}  "
                f"│ {'prim_n':>6} {'prim_PF':>7} {'prim_ret%':>9}  "
                f"│ {'eff_n':>5} {'eff_PF':>7} {'eff_ret%':>9}  "
                f"│ {'mfe_n':>5} {'mfe_PF':>7} {'mfe_ret%':>8}"
            )
            print(syhdr)
            print(f"  {'-'*84}")

            for yr in [2020, 2022, 2023, 2025, 2026]:
                ds_y = pd.Timestamp(f'{yr}-01-01')
                de_y = pd.Timestamp(f'{yr+1}-01-01')
                cols = []
                for inc in [lv_primary, lv_eff, lv_mfe]:
                    t_y = run_fixed_window_baseline(
                        df, session='US_MID', direction=1,
                        sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                        cap_start=250.0, date_start=ds_y, date_end=de_y,
                        include_dates=inc, utc_hour_from=hfrom, utc_hour_to=hto,
                    )
                    m_y = _baseline_metrics(t_y, 250.0)
                    if m_y:
                        pf_y = f"{m_y['pf']:.3f}" if m_y['pf'] != float('inf') else "  inf"
                        cols.append(f" {m_y['n']:>5} {pf_y:>7} {m_y['ret_pct']:>+8.2f}%")
                    else:
                        cols.append(f" {'—':>5} {'—':>7} {'—':>9}")
                marker = "  ◄ BAD" if yr == 2022 else ("  (small n)" if yr == 2026 else "")
                print(f"  {yr:<6}  │{'  │'.join(cols)}{marker}")
            print()

            # ── roll_mfe threshold sensitivity test ───────────────────────────
            # qual_s['mfe_roll'] already computed above (N=20, shift(1), no look-ahead).
            # Test thresholds: 1.0 (reference), 1.1, 1.2.
            # Question: does raising the bar further improve 2022 without
            # materially destroying 2025 or collapsing TEST n?
            thr_variants = [
                ('PRIMARY only',        lv_primary),
                ('roll_mfe > 1.0 [ref]', frozenset(d for d in lv_primary
                    if d in frozenset(qual_s.loc[qual_s['mfe_roll'] > 1.0, 'date']))),
                ('roll_mfe > 1.1',       frozenset(d for d in lv_primary
                    if d in frozenset(qual_s.loc[qual_s['mfe_roll'] > 1.1, 'date']))),
                ('roll_mfe > 1.2',       frozenset(d for d in lv_primary
                    if d in frozenset(qual_s.loc[qual_s['mfe_roll'] > 1.2, 'date']))),
            ]

            print(f"── USTEC  roll_mfe THRESHOLD SENSITIVITY  (N={N_ROLL}, shift=1) ────────")
            for tv_lbl, tv_inc in thr_variants:
                print(f"  {tv_lbl:<26}  total eligible n={len(tv_inc)}")
            print()

            # FULL + TEST summary — compact single-line per variant
            shdr = (
                f"  {'variant':<26}  "
                f"{'FL_n':>5}  {'FL_PF':>6}  {'FL_ret%':>7}  {'FL_MDD%':>7}  "
                f"{'TS_n':>5}  {'TS_PF':>6}  {'TS_ret%':>7}"
            )
            print(shdr)
            print(f"  {'-'*82}")

            for tv_lbl, tv_inc in thr_variants:
                row = []
                for ds, de in [(None, _TRAIN_END), (_TEST_START, None)]:
                    t = run_fixed_window_baseline(
                        df, session='US_MID', direction=1,
                        sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                        cap_start=250.0, date_start=ds, date_end=de,
                        include_dates=tv_inc, utc_hour_from=hfrom, utc_hour_to=hto,
                    )
                    m = _baseline_metrics(t, 250.0)
                    if m:
                        pf_s = f"{m['pf']:.3f}" if m['pf'] != float('inf') else "  inf"
                        if ds is None:   # FULL
                            row.append(f"{m['n']:>5}  {pf_s:>6}  {m['ret_pct']:>+6.2f}%  {m['mdd']:>7.2f}%")
                        else:            # TEST
                            row.append(f"{m['n']:>5}  {pf_s:>6}  {m['ret_pct']:>+6.2f}%")
                    else:
                        row.append("    —      —        —       —" if ds is None else "    —      —        —")
                print(f"  {tv_lbl:<26}  {'  '.join(row)}")
            print()

            # Selected years: 2022 / 2023 (control) / 2025
            print(f"  Year breakdown  (n | PF | ret%)")
            yshdr = (
                f"  {'year':<6}  "
                f"│ {'prim_n':>6} {'prim_PF':>7} {'prim_ret%':>9}  "
                f"│ {'1.0_n':>5} {'1.0_PF':>7} {'1.0_ret%':>9}  "
                f"│ {'1.1_n':>5} {'1.1_PF':>7} {'1.1_ret%':>9}  "
                f"│ {'1.2_n':>5} {'1.2_PF':>7} {'1.2_ret%':>8}"
            )
            print(yshdr)
            print(f"  {'-'*106}")

            for yr in [2022, 2023, 2025]:
                ds_y = pd.Timestamp(f'{yr}-01-01')
                de_y = pd.Timestamp(f'{yr+1}-01-01')
                cols = []
                for _, tv_inc in thr_variants:
                    t_y = run_fixed_window_baseline(
                        df, session='US_MID', direction=1,
                        sp=ap['sp'], lots=ap['ml'], cs=ap['cs'],
                        cap_start=250.0, date_start=ds_y, date_end=de_y,
                        include_dates=tv_inc, utc_hour_from=hfrom, utc_hour_to=hto,
                    )
                    m_y = _baseline_metrics(t_y, 250.0)
                    if m_y:
                        pf_y = f"{m_y['pf']:.3f}" if m_y['pf'] != float('inf') else "  inf"
                        cols.append(f" {m_y['n']:>5} {pf_y:>7} {m_y['ret_pct']:>+8.2f}%")
                    else:
                        cols.append(f" {'—':>5} {'—':>7} {'—':>9}")
                marker = "  ◄ BAD" if yr == 2022 else ""
                print(f"  {yr:<6}  │{'  │'.join(cols)}{marker}")
            print()


if __name__ == '__main__':
    main()
