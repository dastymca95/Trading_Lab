import os

import numpy as np
import pandas as pd

from backtest_config import TIMEFRAME_MINUTES


# ═══════════════════════════════════════════════════════════════════════
# CARGA DE DATOS + QC
# ═══════════════════════════════════════════════════════════════════════

def validate_price_data(df, asset):
    qc = {'Asset': asset, 'Rows Initial': int(len(df)),
          'Duplicates Removed': 0, 'NaN Rows Removed': 0,
          'Corrupt Rows Removed': 0, 'Gap Count': 0, 'Gap %': 0.0,
          'Rows Final': 0, 'Has Spread Column': 'spread' in df.columns}

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    for c in ['open','high','low','close','tick_volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'spread' in df.columns:
        df['spread'] = pd.to_numeric(df['spread'], errors='coerce')
    else:
        df['spread'] = np.nan

    dup = int(df.duplicated(subset=['time']).sum())
    if dup:
        df = df.drop_duplicates(subset=['time'], keep='last').reset_index(drop=True)
    qc['Duplicates Removed'] = dup

    nan_r = int(df[['time','open','high','low','close','tick_volume']].isna().any(axis=1).sum())
    if nan_r:
        df = df.dropna(subset=['time','open','high','low','close','tick_volume']).reset_index(drop=True)
    qc['NaN Rows Removed'] = nan_r

    corrupt = (
        (df['high'] < df[['open','close','low']].max(axis=1)) |
        (df['low']  > df[['open','close','high']].min(axis=1)) |
        (df['low']  > df['high'])
    )
    n_c = int(corrupt.sum())
    if n_c:
        df = df.loc[~corrupt].reset_index(drop=True)
    qc['Corrupt Rows Removed'] = n_c

    df = df.sort_values('time').reset_index(drop=True)
    delta = df['time'].diff().dt.total_seconds().div(60)
    gaps  = int((delta.dropna() != TIMEFRAME_MINUTES).sum())
    qc['Gap Count'] = gaps
    qc['Gap %']     = round(gaps / max(len(df), 1) * 100, 4)
    qc['Rows Final'] = int(len(df))
    return df, qc


def load_price_data(asset, data_dir, digits):
    path_parquet = os.path.join(data_dir, f"{asset}_Data.parquet")
    path_excel   = os.path.join(data_dir, f"{asset}_Data.xlsx")

    if os.path.exists(path_parquet):
        df  = pd.read_parquet(path_parquet, engine='pyarrow')
        fmt = 'parquet'
    elif os.path.exists(path_excel):
        df  = pd.read_excel(path_excel)
        fmt = 'xlsx'
    else:
        print(f"  ⚠️  No encontrado: {asset}_Data.parquet ni {asset}_Data.xlsx")
        return None, None, None, None

    df, qc = validate_price_data(df, asset)
    df = df.sort_values('time').reset_index(drop=True)

    df['date']  = df['time'].dt.date
    df['dow']   = df['time'].dt.dayofweek
    df['idx']   = np.arange(len(df))
    df['pc']    = df['close'].shift(1)
    df['tr']    = np.maximum(df['high']-df['low'],
                  np.maximum(abs(df['high']-df['pc']),
                             abs(df['low']-df['pc'])))
    df['atr14'] = df['tr'].rolling(14).mean()

    point = 10 ** (-digits) if digits > 0 else 1.0
    if 'spread' in df.columns and df['spread'].notna().any():
        df['spread_px'] = df['spread'].ffill().bfill().fillna(0) * point
        spread_src = 'dynamic_mt5'
    else:
        df['spread_px'] = np.nan
        spread_src = 'fallback_hardcoded'

    lon = df[(df['time'].dt.hour >= 9) & (df['time'].dt.hour < 15)]
    lr  = lon.groupby('date').agg(
        lh=('high','max'), ll=('low','min'), lam=('atr14','mean')
    ).reset_index()
    lr['lrr'] = (lr['lh'] - lr['ll']) / lr['lam'].replace(0, np.nan)

    vm = df['tick_volume'].mean()
    print(f"  ✓ {asset} [{fmt}]: {len(df):,} velas | "
          f"{df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()} | "
          f"spread={spread_src} | vm={vm:.1f}")
    return df, lr, vm, qc
