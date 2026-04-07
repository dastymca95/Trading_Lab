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


if __name__ == '__main__':
    main()
