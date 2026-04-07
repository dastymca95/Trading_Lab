"""
session_clock.py
================
DST-aware session labeler for MT5 price data.

Problem this solves
-------------------
MT5 data timestamps are naive (no timezone).  The broker server clock is
typically EET (UTC+2 winter / UTC+3 summer) or a fixed UTC+2 offset.
Without explicit conversion, a fixed MT5 hour drifts by ±1 h relative to
actual exchange opens during DST transitions — making session-based research
unreliable.

Design
------
1.  Accept naive MT5 timestamps.
2.  Localize to the broker's IANA timezone (configurable).
3.  Convert to UTC.
4.  Assign a session label based on UTC hour.

Session boundaries (UTC, exchange-anchored)
-------------------------------------------
  ASIA      00:00 – 06:59  Asia / off-hours / overnight
  EU_PRE    07:00 – 07:59  Pre-Xetra (DE futures light session)
  EU_OPEN   08:00 – 09:59  Xetra + LSE open  (high-volatility window)
  EU_MID    10:00 – 13:29  EU mid-session    (lower volatility, range)
  US_PRE    13:30 – 14:29  Pre-US cash       (30-min setup window)
  US_OPEN   14:30 – 15:59  US cash open      (CME equity, peak volume)
  US_MID    16:00 – 19:59  US mid-session    (momentum / mean-reversion)
  US_LATE   20:00 – 21:59  US late / close   (hedging, position squaring)
  ASIA      22:00 – 23:59  rollover to next Asia

Broker timezone notes
---------------------
  'Etc/GMT-2'       — UTC+2 fixed, no DST  (common: IC Markets raw feed)
  'Europe/Athens'   — UTC+2/+3 with DST    (EET/EEST, some EU brokers)
  'Europe/Helsinki' — UTC+2/+3 with DST    (equivalent to Athens)

Usage
-----
    from research.session_clock import label_sessions, session_distribution

    df = label_sessions(df, mt5_tz='Etc/GMT-2')
    print(session_distribution(df))
"""

from zoneinfo import ZoneInfo
import pandas as pd

# ─── Default MT5 server timezone ─────────────────────────────────────────────
# Override this if your broker observes DST (use 'Europe/Athens' or similar).
MT5_TZ_DEFAULT = 'Etc/GMT-2'

# ─── Session boundary table (UTC) ────────────────────────────────────────────
# Each tuple: (start_hour_utc, start_minute_utc, session_label)
# Sorted ascending.  Last matching entry wins for any given bar.
_SESSIONS_UTC = [
    ( 0,  0, 'ASIA'),
    ( 7,  0, 'EU_PRE'),
    ( 8,  0, 'EU_OPEN'),
    (10,  0, 'EU_MID'),
    (13, 30, 'US_PRE'),
    (14, 30, 'US_OPEN'),
    (16,  0, 'US_MID'),
    (20,  0, 'US_LATE'),
    (22,  0, 'ASIA'),       # end-of-day rollover
]

# Pre-compute boundaries in minutes-since-midnight for fast vectorized lookup
_BOUNDARIES_MIN = [(h * 60 + m, lbl) for h, m, lbl in _SESSIONS_UTC]

# Canonical display order
SESSION_ORDER = ['ASIA', 'EU_PRE', 'EU_OPEN', 'EU_MID',
                 'US_PRE', 'US_OPEN', 'US_MID', 'US_LATE']


# ─── Public API ──────────────────────────────────────────────────────────────

def label_sessions(df: pd.DataFrame, mt5_tz: str = MT5_TZ_DEFAULT) -> pd.DataFrame:
    """
    Add time_utc and session_label columns to a price DataFrame.

    Parameters
    ----------
    df      : DataFrame with df['time'] as naive timestamps in MT5 server time.
    mt5_tz  : IANA timezone string matching the MT5 server clock.
              Examples: 'Etc/GMT-2', 'Europe/Athens', 'Europe/Helsinki'.

    Returns
    -------
    df copy with two new columns:
      time_utc      — naive UTC timestamps (tzinfo stripped for clean usage)
      session_label — one of SESSION_ORDER labels (str)

    Notes
    -----
    - ambiguous='infer'        handles fall-back ambiguous hours automatically.
    - nonexistent='shift_forward' handles spring-forward gaps.
    - Both only matter when mt5_tz observes DST (e.g. 'Europe/Athens').
      For 'Etc/GMT-2' (fixed offset) they are harmless no-ops.
    """
    df = df.copy()
    tz = ZoneInfo(mt5_tz)

    df['time_utc'] = (
        df['time']
        .dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
        .dt.tz_convert('UTC')
        .dt.tz_localize(None)       # strip tzinfo — keeps downstream code simple
    )

    hm = df['time_utc'].dt.hour * 60 + df['time_utc'].dt.minute

    # Vectorized label assignment: iterate boundaries ascending,
    # later (larger) start_min entries overwrite earlier ones.
    labels = pd.Series('ASIA', index=df.index, dtype=object)
    for start_min, lbl in _BOUNDARIES_MIN:
        # .where(cond, other): keep label where cond is True, else set other
        labels = labels.where(hm < start_min, lbl)

    df['session_label'] = labels
    return df


def session_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return bar count and percentage per session label.

    Useful as a sanity-check: verifies the labeling is correct
    and shows relative weight of each session in the dataset.

    Raises ValueError if label_sessions() has not been called first.
    """
    if 'session_label' not in df.columns:
        raise ValueError("Run label_sessions(df) before calling session_distribution().")

    counts = df['session_label'].value_counts().rename('bars')
    pct    = (counts / counts.sum() * 100).round(2).rename('pct')
    idx    = [s for s in SESSION_ORDER if s in counts.index]
    return pd.concat([counts, pct], axis=1).reindex(idx)
