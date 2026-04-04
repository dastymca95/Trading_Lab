#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BACKTEST HÍBRIDO — v2 ULTRA ALIGNED
Alineado con el bot live MT5-only:

Mejoras principales:
  + GLOBAL_RISK_PCT: una sola línea controla el riesgo de TODOS los activos
  + USE_LIVE_SPECS: configurable fácilmente
      - True  -> specs live desde MT5
      - False -> specs desde JSON / base
  + Comisión con misma jerarquía conceptual del extractor/bot:
      1) historial real de deals
      2) order_check()
      3) fallback manual/base
  + Mantiene precios históricos desde .parquet por defecto, con fallback a .xlsx
  + Mantiene spread dinámico de la vela si existe
  + Mantiene QC, bootstrap, sign test, Monte Carlo, rolling Sharpe, heatmap, etc.

USO:
    pip install pandas numpy scipy openpyxl pyarrow MetaTrader5
    python backtest_hibrido_v2_ultra_aligned.py
"""

import os
import glob
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

SEED = 42
np.random.seed(SEED)

INITIAL_PER_ASSET  = 250.0
GLOBAL_RISK_PCT    = 0.02      # ← CAMBIA SOLO ESTA LÍNEA
USE_LIVE_SPECS     = True      # ← True = MT5 live specs | False = JSON/base
AUTO_ENABLE_SYMBOL = True
COMMISSION_LOOKBACK_DAYS = 90

TRAIN_END          = pd.Timestamp('2025-01-01')
TEST_START         = pd.Timestamp('2025-01-01')
MAX_BARS           = 240
ROLLING_SHARPE_WIN = 20
MONTE_CARLO_RUNS   = 2000
BOOTSTRAP_RUNS     = 2000
TIMEFRAME_MINUTES  = 2

# Parámetros estratégicos base
ASSET_PARAMS_BASE = {
    'XAUUSD': {
        'sl_pct':0.003, 'trail_mult':1.0,  'risk_pct':GLOBAL_RISK_PCT, 'lrr_min':1.5,
        'hours':[15,18], 'dow':[1,2,3,4],   'atr_mult':1.5,
        'comm':7.00, 'cs':100,    'ml':0.01, 'step':0.01, 'sp':0.30,  'jpy':False, 'digits':2,
    },
    'US30': {
        'sl_pct':0.003, 'trail_mult':3.0,  'risk_pct':GLOBAL_RISK_PCT, 'lrr_min':1.0,
        'hours':[15,18], 'dow':[0,1,2,3,4], 'atr_mult':1.5,
        'comm':0.00, 'cs':1,      'ml':0.1,  'step':0.1,  'sp':3.0,   'jpy':False, 'digits':2,
    },
}

OPTIONAL_ASSETS = ['USTEC', 'US500', 'DE40']

OPTIONAL_PARAMS = {
    'USTEC': {'sl_pct':0.003,'trail_mult':3.0,'risk_pct':GLOBAL_RISK_PCT,'lrr_min':1.0,
              'hours':[15,18],'dow':[0,1,2,3,4],'atr_mult':1.5,
              'comm':0.00,'cs':1,'ml':0.1,'step':0.1,'sp':1.0,'jpy':False,'digits':2},
    'US500': {'sl_pct':0.003,'trail_mult':3.0,'risk_pct':GLOBAL_RISK_PCT,'lrr_min':1.0,
              'hours':[15,18],'dow':[0,1,2,3,4],'atr_mult':1.5,
              'comm':0.00,'cs':1,'ml':0.1,'step':0.1,'sp':0.5,'jpy':False,'digits':2},
    'DE40':  {'sl_pct':0.003,'trail_mult':3.0,'risk_pct':GLOBAL_RISK_PCT,'lrr_min':1.0,
              'hours':[15,18],'dow':[0,1,2,3,4],'atr_mult':1.5,
              'comm':0.00,'cs':1,'ml':0.1,'step':0.1,'sp':1.0,'jpy':False,'digits':2},
}

# Fallback manual tipo extractor
COMMISSION_MANUAL_RT = {
    'XAUUSD': 7.00,
    'US30':   0.00,
    'USTEC':  0.00,
    'US500':  0.00,
    'DE40':   0.00,
}


# ═══════════════════════════════════════════════════════════════════════
# UTILIDADES MT5 / SPECS
# ═══════════════════════════════════════════════════════════════════════

def connect_mt5() -> bool:
    if mt5 is None:
        print("  ⚠️  MetaTrader5 no está instalado.")
        return False
    if not mt5.initialize():
        print(f"  ❌ Error initialize(): {mt5.last_error()}")
        return False
    info = mt5.account_info()
    if info is None:
        print("  ❌ No se pudo leer account_info()")
        return False
    print(f"  ✓ Conectado MT5 | Cuenta: {info.login} | Broker: {info.company}")
    return True


def safe_symbol(symbol: str) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"  ⚠️  {symbol}: no existe en este broker")
        return False
    if info.visible:
        return True
    if AUTO_ENABLE_SYMBOL and mt5.symbol_select(symbol, True):
        print(f"  ✓ {symbol}: activado en Market Watch")
        return True
    print(f"  ⚠️  {symbol}: no visible y no se pudo activar")
    return False


def _comm_from_history_rt(symbol: str):
    try:
        deals = mt5.history_deals_get(
            datetime.now() - timedelta(days=COMMISSION_LOOKBACK_DAYS),
            datetime.now()
        )
        if deals is None or len(deals) == 0:
            return None
        hits = []
        for d in deals:
            if getattr(d, 'symbol', None) != symbol:
                continue
            comm = getattr(d, 'commission', 0)
            vol = getattr(d, 'volume', 0)
            if comm in (None, 0) or vol in (None, 0):
                continue
            hits.append(abs(comm) / vol)   # por lote, por lado
        if not hits:
            return None
        return round(float(sum(hits) / len(hits)) * 2, 4)
    except Exception:
        return None


def _comm_from_order_check_rt(symbol: str, volume_min: float):
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        check = mt5.order_check({
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       volume_min,
            "type":         mt5.ORDER_TYPE_BUY,
            "price":        tick.ask,
            "deviation":    50,
            "magic":        0,
            "comment":      "spec_check",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })
        if check is None:
            return None
        commission = getattr(check, 'commission', None)
        if commission in (None, 0):
            return None
        side_per_lot = abs(float(commission)) / max(volume_min, 1e-8)
        return round(side_per_lot * 2, 4)
    except Exception:
        return None


def get_live_commission_rt(symbol: str, volume_min: float, fallback_rt: float):
    c = _comm_from_history_rt(symbol)
    if c is not None:
        return c, "history"
    c = _comm_from_order_check_rt(symbol, volume_min)
    if c is not None:
        return c, "order_check"
    return float(fallback_rt), "fallback"


def load_broker_specs_json(data_dir):
    json_files = sorted(glob.glob(os.path.join(data_dir, 'especificaciones_*.json')), reverse=True)
    if json_files:
        print(f"  ✓ Specs JSON: {os.path.basename(json_files[0])}")
        with open(json_files[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    print("  ℹ️  Sin JSON de specs — usando parámetros base")
    return None


def apply_json_specs(specs, base_params):
    if specs is None:
        return {k: v.copy() for k, v in base_params.items()}
    updated = {}
    for asset, p in base_params.items():
        sp = specs.get(asset)
        if sp is None:
            updated[asset] = p.copy()
            continue
        q = p.copy()
        q['comm']   = float(sp.get('commission_round_trip', p['comm']) or p['comm'])
        q['cs']     = int(sp.get('contract_size', p['cs']) or p['cs'])
        q['ml']     = float(sp.get('volume_min', p['ml']) or p['ml'])
        q['step']   = float(sp.get('volume_step', p.get('step', p['ml'])) or p.get('step', p['ml']))
        q['digits'] = int(sp.get('digits', p['digits']) or p['digits'])
        q['jpy']    = bool(sp.get('is_jpy', p['jpy']))
        updated[asset] = q
    return updated


def apply_global_risk(params: dict, risk_pct: float) -> dict:
    out = {}
    for asset, p in params.items():
        q = p.copy()
        q['risk_pct'] = float(risk_pct)
        out[asset] = q
    return out


def apply_live_specs(base_params: dict):
    out = {}
    connected = connect_mt5()
    if not connected:
        print("  ⚠️  MT5 no disponible. Se usarán parámetros base/fallback.")
        return apply_global_risk(base_params, GLOBAL_RISK_PCT)

    for asset, p in base_params.items():
        if not safe_symbol(asset):
            q = p.copy()
            q['risk_pct'] = GLOBAL_RISK_PCT
            q['comm_source'] = 'fallback_no_symbol'
            out[asset] = q
            continue

        info = mt5.symbol_info(asset)
        q = p.copy()

        if info is not None:
            q['cs']     = int(info.trade_contract_size) if info.trade_contract_size else q['cs']
            q['ml']     = float(info.volume_min) if info.volume_min else q['ml']
            q['step']   = float(info.volume_step) if info.volume_step else q['step']
            q['digits'] = int(info.digits) if info.digits is not None else q['digits']
            q['jpy']    = bool(getattr(info, 'currency_profit', '') == 'JPY')

        comm_rt, source = get_live_commission_rt(
            asset,
            volume_min=q['ml'],
            fallback_rt=COMMISSION_MANUAL_RT.get(asset, q['comm'])
        )
        q['comm'] = float(comm_rt)
        q['comm_source'] = source
        q['risk_pct'] = GLOBAL_RISK_PCT
        out[asset] = q

    try:
        mt5.shutdown()
    except Exception:
        pass

    return out


def resolve_asset_params(data_dir):
    merged = {**ASSET_PARAMS_BASE}
    for k, v in OPTIONAL_PARAMS.items():
        merged[k] = v.copy()

    if USE_LIVE_SPECS:
        print("1. Cargando specs live desde MT5...")
        params = apply_live_specs(merged)
    else:
        print("1. Buscando specs del broker en JSON...")
        specs = load_broker_specs_json(data_dir)
        params = apply_json_specs(specs, merged)

    params = apply_global_risk(params, GLOBAL_RISK_PCT)
    return params


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


# ═══════════════════════════════════════════════════════════════════════
# MOTOR DE BACKTEST
# ═══════════════════════════════════════════════════════════════════════

def dynamic_spread(sig_row, p):
    v = sig_row.get('spread_px', np.nan)
    if pd.notna(v) and v > 0:
        return float(v)
    return float(p['sp'])


def run_backtest(asset, df, lr, vm, p, cap_start,
                 date_start=None, date_end=None, label='FULL'):
    H = df['high'].values
    L = df['low'].values
    C = df['close'].values

    sigs = []
    for h in p['hours']:
        s = df[(df['time'].dt.hour == h) & (df['time'].dt.minute == 0)].copy()
        s = s.merge(lr, on='date', how='left')
        s['sh'] = h
        sigs.append(s)
    if not sigs:
        return pd.DataFrame(), np.array([cap_start])

    sg = pd.concat(sigs).sort_values(['date','time']).reset_index(drop=True)
    if date_start is not None:
        sg = sg[pd.to_datetime(sg['date']) >= date_start]
    if date_end is not None:
        sg = sg[pd.to_datetime(sg['date']) < date_end]
    sg = sg.reset_index(drop=True)
    if len(sg) == 0:
        return pd.DataFrame(), np.array([cap_start])

    cap = cap_start
    equity = [cap]
    trades = []
    ops = {}
    DOW    = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday'}

    for _, sig in sg.iterrows():
        d   = sig['date']; dow = sig['dow']; ep = sig['close']
        si  = int(sig['idx']); av = sig['atr14']
        lh  = sig['lh'];   ll = sig['ll'];   tv = sig['tick_volume']
        cr  = sig['high'] - sig['low']
        lrr = sig.get('lrr', np.nan)
        sp  = dynamic_spread(sig, p)

        if dow not in p['dow']: continue
        if tv < vm: continue
        if ops.get(d, 0) >= 2: continue
        if pd.isna(av) or av == 0: continue
        if pd.isna(lrr) or lrr <= p['lrr_min']: continue

        direction = None; stype = ''
        if not (pd.isna(lh) or pd.isna(ll)):
            if ep > lh:
                direction = 1
                stype = f"Breakout ALCISTA (London High={round(lh, p['digits'])})"
            elif ep < ll:
                direction = -1
                stype = f"Breakout BAJISTA (London Low={round(ll, p['digits'])})"
        if direction is None and cr > p['atr_mult'] * av:
            direction = -1 if sig['close'] > sig['open'] else 1
            stype = f"Vela grande ({round(cr, p['digits'])} > {p['atr_mult']}xATR)"
        if direction is None: continue

        sl = ep*(1-p['sl_pct']) if direction==1 else ep*(1+p['sl_pct'])
        if not (pd.isna(lh) or pd.isna(ll)):
            sl = max(sl, ll) if direction==1 else min(sl, lh)

        sl_dist = max(abs(ep - sl) + sp, ep * 0.0015)
        if sl_dist < 1e-8: continue

        ppu = p['cs'] / max(ep, 1) if p['jpy'] else p['cs']
        step = p.get('step', p['ml'])

        lots = (cap * p['risk_pct']) / max(ppu * sl_dist + p['comm'], 1e-8)
        lots = round(lots / step) * step
        lots = max(p['ml'], lots)

        lots_max = (cap * 0.20) / max(ppu * sl_dist + p['comm'], 0.001)
        lots_max = round(lots_max / step) * step
        lots = max(p['ml'], min(lots, lots_max))
        lots = round(lots, 4)

        ei  = min(si + 1 + MAX_BARS, len(df))
        fH  = H[si+1:ei]; fL = L[si+1:ei]; fC = C[si+1:ei]
        ib  = direction == 1
        csl = sl; bp = ep; ber = False
        be_p = (ep + sp) if ib else (ep - sp)
        xp  = ep; bars = MAX_BARS

        for i in range(len(fH)):
            h2, l2 = fH[i], fL[i]
            if np.isnan(h2): break
            if ib  and l2 <= csl: xp = csl; bars = i+1; break
            if not ib and h2 >= csl: xp = csl; bars = i+1; break
            if ib:  bp = max(bp, h2); ber = ber or h2 >= be_p
            else:   bp = min(bp, l2); ber = ber or l2 <= be_p
            if not ber: continue
            ns  = (bp - av*p['trail_mult']) if ib else (bp + av*p['trail_mult'])
            ns  = max(ns, be_p) if ib else min(ns, be_p)
            csl = max(csl, ns) if ib else min(csl, ns)
        else:
            li = len(fC) - 1
            while li >= 0 and np.isnan(fC[li]): li -= 1
            xp = fC[li] if li >= 0 else ep; bars = len(fH)

        raw = (xp-ep)*direction*lots*(p['cs']/max((ep+xp)/2,1) if p['jpy'] else p['cs'])
        spread_cost = sp * lots * p['cs']
        net = raw - lots * p['comm'] - spread_cost
        result = 'WIN' if net > 0.01 else ('LOSS' if net < -0.01 else 'BE')
        cap = max(cap + net, 0.01)
        equity.append(cap)
        ops[d] = ops.get(d, 0) + 1

        trades.append({
            '#': len(trades)+1,             'Sample': label,
            'Activo': asset,                'Fecha Apertura': str(pd.Timestamp(sig['time']))[:16],
            'Fecha Cierre': str(pd.Timestamp(sig['time'])+pd.Timedelta(minutes=bars*2))[:16],
            'Dia Semana': DOW.get(dow,''),   'Hora Señal MT5': f"{sig['sh']}:00",
            'Dirección': 'BUY' if direction==1 else 'SELL',
            'Tipo Señal': stype,
            'London High': round(lh, p['digits']) if not pd.isna(lh) else '',
            'London Low':  round(ll, p['digits']) if not pd.isna(ll) else '',
            'Ratio L/ATR': round(lrr, 2) if pd.notna(lrr) else np.nan,
            'ATR': round(av, p['digits']),    'Spread Usado': round(sp, p['digits']),
            'Precio Entrada': round(ep, p['digits']),
            'SL Inicial': round(sl, p['digits']),
            'Precio Salida': round(xp, p['digits']),
            'Distancia SL pts': round(sl_dist, p['digits']),
            'Lotes': lots,                    'Riesgo %': round(p['risk_pct'] * 100, 2),
            'Riesgo USD': round(cap*p['risk_pct'],2),   'Comisión RT': p['comm'],
            'PnL Bruto USD': round(raw,2),    'Comisión USD': round(lots*p['comm'],2),
            'PnL Neto USD': round(net,2),      'Capital Tras Op': round(cap,2),
            'Resultado': result,              'Duración (min)': bars*2,
            'Mes': pd.to_datetime(d).strftime('%Y-%m'),
            'Año': pd.to_datetime(d).year,
            'Date': str(pd.to_datetime(d).date()),
        })

    return pd.DataFrame(trades), np.array(equity)


# ═══════════════════════════════════════════════════════════════════════
# ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_mean_ci(x, runs=BOOTSTRAP_RUNS, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0: return 0.0, 0.0
    means = np.array([np.random.choice(x, len(x), replace=True).mean() for _ in range(runs)])
    return float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))


def sign_test_pvalue(x):
    x = np.asarray(x, dtype=float); x = x[x != 0]
    n = len(x)
    if n == 0: return 1.0
    return float(scipy_stats.binomtest(int((x>0).sum()), n=n, p=0.5, alternative='two-sided').pvalue)


def monte_carlo_dd(pnl_arr, cap0, runs=MONTE_CARLO_RUNS):
    pnl_arr = np.asarray(pnl_arr, dtype=float)
    if len(pnl_arr) == 0:
        return 0.0, 0.0, 0.0, cap0

    dds = np.empty(runs)
    finals = np.empty(runs)

    for i in range(runs):
        shuffled = np.random.permutation(pnl_arr)
        cap = cap0
        eq = [cap]
        for p in shuffled:
            cap = max(cap + p, 0.01)
            eq.append(cap)
        eq = np.array(eq)
        pk = np.maximum.accumulate(eq)
        dds[i] = ((eq - pk) / pk * 100).min()
        finals[i] = eq[-1]

    return (
        float(np.quantile(dds, 0.05)),
        float(np.median(dds)),
        float(np.quantile(dds, 0.95)),
        float(np.median(finals)),
    )


def daily_equity(t, cap0):
    if len(t) == 0:
        return pd.DataFrame(columns=['Date','DailyPnL','Equity','ReturnPct','DrawdownPct'])
    d = t.groupby('Date', as_index=False)['PnL Neto USD'].sum().sort_values('Date')
    d['Equity']     = cap0 + d['PnL Neto USD'].cumsum()
    d['ReturnPct']  = d['PnL Neto USD'] / cap0 * 100.0
    pk = d['Equity'].cummax()
    d['DrawdownPct'] = (d['Equity'] - pk) / pk * 100.0
    return d.rename(columns={'PnL Neto USD':'DailyPnL'})


def rolling_sharpe(t, cap0, window=ROLLING_SHARPE_WIN):
    d = daily_equity(t, cap0)
    if len(d) == 0: return pd.DataFrame()
    r = d['ReturnPct'] / 100.0
    d['RollingSharpe'] = (r.rolling(window).mean() / r.rolling(window).std(ddof=1)) * np.sqrt(252)
    return d[['Date','ReturnPct','RollingSharpe']]


def monthly_heatmap(t):
    if len(t) == 0: return pd.DataFrame()
    m = t.copy()
    m['Year'] = pd.to_datetime(m['Date']).dt.year
    m['MonthNum'] = pd.to_datetime(m['Date']).dt.month
    heat = m.pivot_table(index='Year', columns='MonthNum',
                         values='PnL Neto USD', aggfunc='sum', fill_value=0.0)
    names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
             7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    return heat.rename(columns=names).reset_index()


def return_distribution(t, cap0):
    d = daily_equity(t, cap0)
    if len(d) == 0: return pd.DataFrame(columns=['Metric','Value'])
    x = d['ReturnPct'].values
    rows = [
        ('CountDays',   len(x)),
        ('MeanDailyPct',   np.mean(x)),
        ('MedianDailyPct', np.median(x)),
        ('StdDailyPct',    np.std(x,ddof=1) if len(x)>1 else 0.0),
        ('SkewDaily',      scipy_stats.skew(x,bias=False) if len(x)>2 else 0.0),
        ('KurtosisDaily',  scipy_stats.kurtosis(x,fisher=True,bias=False) if len(x)>3 else 0.0),
        ('Pct05',  np.quantile(x,.05)), ('Pct25', np.quantile(x,.25)),
        ('Pct50',  np.quantile(x,.50)), ('Pct75', np.quantile(x,.75)),
        ('Pct95',  np.quantile(x,.95)),
        ('WorstDayPct', np.min(x)),     ('BestDayPct',  np.max(x)),
    ]
    return pd.DataFrame(rows, columns=['Metric','Value'])


def calc_metrics(t, e, cap0):
    if len(t) == 0:
        return {}

    pk = np.maximum.accumulate(e)
    mdd = ((e - pk) / pk * 100).min()

    w  = t[t['Resultado'] == 'WIN']
    lo = t[t['Resultado'] == 'LOSS']
    be = t[t['Resultado'] == 'BE']

    ret = (e[-1] - e[0]) / e[0] * 100
    wr  = len(w) / len(t) * 100

    gp = w['PnL Neto USD'].sum() if len(w) > 0 else 0
    gl = abs(lo['PnL Neto USD'].sum()) if len(lo) > 0 else 0.001
    pf = gp / gl

    days = t['Date'].nunique()
    aw = w['PnL Neto USD'].mean() if len(w) > 0 else 0
    al = lo['PnL Neto USD'].mean() if len(lo) > 0 else 0
    exp = t['PnL Neto USD'].mean()

    monthly = t.groupby('Mes')['PnL Neto USD'].sum()
    mret = monthly / cap0 * 100

    mret_std = mret.std(ddof=1)
    sharpe = (
        mret.mean() / mret_std * np.sqrt(12)
        if len(mret) > 1 and pd.notna(mret_std) and mret_std > 0
        else np.nan
    )

    downside = np.minimum(mret, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2)) if len(mret) > 0 else np.nan
    sortino = (
        mret.mean() / downside_dev * np.sqrt(12)
        if pd.notna(downside_dev) and downside_dev > 0
        else np.nan
    )

    ann = (
        ((e[-1] / e[0]) ** (252 / days) - 1) * 100
        if days > 0 and e[0] > 0 and e[-1] > 0
        else 0
    )
    calmar = ann / abs(mdd) if mdd != 0 else 0

    daily_pnl = t.groupby('Date')['PnL Neto USD'].sum().sort_index()
    t_p = scipy_stats.ttest_1samp(daily_pnl, 0).pvalue if len(daily_pnl) > 1 else 1.0
    sign_p = sign_test_pvalue(daily_pnl.values)
    ci_lo, ci_hi = bootstrap_mean_ci(daily_pnl.values)

    mc_dd_p5, mc_dd_p50, mc_dd_p95, mc_final = monte_carlo_dd(
        t['PnL Neto USD'].values, cap0
    )

    ms = 0
    cur = 0
    for r in t['Resultado']:
        if r == 'LOSS':
            cur += 1
            ms = max(ms, cur)
        else:
            cur = 0

    return {
        'n': len(t), 'n_win': len(w), 'n_loss': len(lo), 'n_be': len(be),
        'wr': round(wr, 1), 'ret': round(ret, 2), 'final': round(e[-1], 2),
        'mdd': round(mdd, 2), 'pf': round(pf, 2),
        'sharpe': round(float(sharpe), 3) if pd.notna(sharpe) else np.nan,
        'sortino': round(float(sortino), 3) if pd.notna(sortino) else np.nan,
        'calmar': round(calmar, 3), 'aw': round(aw, 2), 'al': round(al, 2),
        'rr': round(abs(aw / al), 3) if al != 0 else 0,
        'exp': round(exp, 4), 'p_val': round(float(t_p), 6),
        'sign_p': round(float(sign_p), 6),
        'boot_lo': round(float(ci_lo), 4), 'boot_hi': round(float(ci_hi), 4),
        'mc_dd_p5': round(float(mc_dd_p5), 2),
        'mc_dd_p50': round(float(mc_dd_p50), 2),
        'mc_dd_p95': round(float(mc_dd_p95), 2),
        'mc_final_p50': round(float(mc_final), 2),
        'ms': ms, 'days': days, 'avg_day': round(ret / days, 3) if days > 0 else 0,
        'total_comm': round(t['Comisión USD'].sum(), 2),
        'total_raw': round(t['PnL Bruto USD'].sum(), 2),
        'monthly': monthly, 'monthly_ret': mret, 'equity': e,
    }


# ═══════════════════════════════════════════════════════════════════════
# EXCEL
# ═══════════════════════════════════════════════════════════════════════

HFILL = PatternFill('solid', fgColor='1F2937')
HFONT = Font(color='FFFFFF', bold=True)
thin  = Side(style='thin', color='D1D5DB')
BRD   = Border(left=thin, right=thin, top=thin, bottom=thin)


def write_df_sheet(wb, name, df):
    ws = wb.create_sheet(name)
    if df is None or len(df) == 0:
        ws['A1'] = 'Sin datos'
        return ws
    ws.append(df.columns.tolist())
    for row in df.itertuples(index=False):
        ws.append(list(row))
    for cell in ws[1]:
        cell.fill = HFILL
        cell.font = HFONT
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    for row in ws.iter_rows(min_row=2, max_row=min(ws.max_row, 500)):
        for cell in row:
            cell.border = BRD
    ws.freeze_panes = 'A2'
    for col in ws.iter_cols(max_row=min(ws.max_row, 200)):
        letter = col[0].column_letter
        width  = max((len(str(cell.value)) for cell in col if cell.value), default=10)
        ws.column_dimensions[letter].width = min(width + 2, 28)
    return ws


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   BACKTEST HÍBRIDO — v2 ULTRA ALIGNED                   ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    data_dir = os.path.dirname(os.path.abspath(__file__))
    ts       = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = os.path.join(data_dir, f"Backtest_Hibrido_Aligned_{ts}.xlsx")

    ASSET_PARAMS = resolve_asset_params(data_dir)

    # solo mantener activos con archivo de precios disponible
    final_assets = {}
    for asset, p in ASSET_PARAMS.items():
        pq = os.path.join(data_dir, f"{asset}_Data.parquet")
        xl = os.path.join(data_dir, f"{asset}_Data.xlsx")
        if os.path.exists(pq) or os.path.exists(xl):
            final_assets[asset] = p

    if not final_assets:
        print("\n❌ Sin archivos de datos.")
        return

    print("\n2. Cargando precios...")
    asset_data = {}
    qc_rows = []
    warnings_rows = []

    for asset, p in final_assets.items():
        df, lr, vm, qc = load_price_data(asset, data_dir, p['digits'])
        if df is not None:
            asset_data[asset] = (df, lr, vm)
            qc_rows.append(qc)
            if qc['Gap %'] > 0.5:
                warnings_rows.append({'Asset':asset, 'Warning':f"Gaps: {qc['Gap Count']} ({qc['Gap %']}%)"})
            if qc['Duplicates Removed'] > 0:
                warnings_rows.append({'Asset':asset, 'Warning':f"Duplicados: {qc['Duplicates Removed']}"})

    if not asset_data:
        print("\n❌ Sin trades / sin datos cargables.")
        return

    print("\n3. Corriendo backtest...")
    results_full = {}
    results_test = {}
    all_full = []
    all_test = []

    for asset, (df, lr, vm) in asset_data.items():
        p = final_assets[asset]
        cap = INITIAL_PER_ASSET

        print(f"   {asset}... ", end='', flush=True)

        t_f, e_f = run_backtest(asset, df, lr, vm, p, cap, None, None, 'FULL')
        m_f = calc_metrics(t_f, e_f, cap)
        results_full[asset] = m_f
        if len(t_f) > 0:
            t_f['asset'] = asset
            all_full.append(t_f)

        t_t, e_t = run_backtest(asset, df, lr, vm, p, cap, TEST_START, None, 'TEST')
        m_t = calc_metrics(t_t, e_t, cap)
        results_test[asset] = m_t
        if len(t_t) > 0:
            all_test.append(t_t)

        print(
            f"FULL {m_f.get('ret',0):+.1f}% | TEST {m_t.get('ret',0):+.1f}% | "
            f"PF={m_f.get('pf',0)} | comm={p['comm']} ({p.get('comm_source','base/json')}) | "
            f"risk={p['risk_pct']:.2%}"
        )

    if not all_full:
        print("❌ Sin trades.")
        return

    print("\n4. Calculando portafolio...")
    all_t_full = pd.concat(all_full).sort_values('Fecha Apertura').reset_index(drop=True)
    all_t_full['#'] = range(1, len(all_t_full)+1)
    all_t_test = pd.concat(all_test).sort_values('Fecha Apertura').reset_index(drop=True) if all_test else pd.DataFrame()

    cap0 = INITIAL_PER_ASSET * len(results_full)
    cap  = cap0
    eq = [cap]
    for pnl in all_t_full['PnL Neto USD'].values:
        cap = max(cap + pnl, 0.01)
        eq.append(cap)
    eq = np.array(eq)
    pm_full = calc_metrics(all_t_full, eq, cap0)

    if len(all_t_test) > 0:
        cap = cap0
        eq_t = [cap]
        for pnl in all_t_test['PnL Neto USD'].values:
            cap = max(cap + pnl, 0.01)
            eq_t.append(cap)
        eq_t = np.array(eq_t)
        pm_test = calc_metrics(all_t_test, eq_t, cap0)
    else:
        pm_test = {}

    print(f"   Portfolio FULL: Ret={pm_full['ret']:+.1f}% | Sharpe={pm_full['sharpe']} | DD={pm_full['mdd']}%")
    if pm_test:
        print(f"   Portfolio TEST: Ret={pm_test['ret']:+.1f}% | Sharpe={pm_test['sharpe']} | DD={pm_test['mdd']}%")

    print("\n5. Generando Excel...")
    wb = Workbook()
    ws = wb.active
    ws.title = 'Resumen'

    resumen_rows = []
    for asset in results_full:
        mf = results_full.get(asset, {})
        mt = results_test.get(asset, {})
        p  = final_assets[asset]
        resumen_rows.append({
            'Asset':asset,
            'RiskPct': p['risk_pct'],
            'CommRT': p['comm'],
            'CommSource': p.get('comm_source', 'json/base'),
            'SpecsMode': 'live_mt5' if USE_LIVE_SPECS else 'json/base',
            'Params':json.dumps({k:p[k] for k in
                ['sl_pct','trail_mult','risk_pct','lrr_min','hours','dow','atr_mult']},ensure_ascii=False),
            'Trades_FULL':mf.get('n',0), 'RetPct_FULL':mf.get('ret',0),
            'PF_FULL':mf.get('pf',0), 'Sharpe_FULL':mf.get('sharpe',0),
            'Sortino_FULL':mf.get('sortino',0), 'DDPct_FULL':mf.get('mdd',0),
            'Trades_TEST':mt.get('n',0), 'RetPct_TEST':mt.get('ret',0),
            'PF_TEST':mt.get('pf',0), 'Sharpe_TEST':mt.get('sharpe',0),
            'Sortino_TEST':mt.get('sortino',0), 'DDPct_TEST':mt.get('mdd',0),
        })

    res_df = pd.DataFrame(resumen_rows)
    ws.append(res_df.columns.tolist())
    for row in res_df.itertuples(index=False):
        ws.append(list(row))
    for cell in ws[1]:
        cell.fill = HFILL
        cell.font = HFONT
        cell.alignment = Alignment(horizontal='center', wrap_text=True)

    port_df = pd.DataFrame([{
        'Sample':'FULL',
        'UseLiveSpecs': USE_LIVE_SPECS,
        'GlobalRiskPct': GLOBAL_RISK_PCT,
        **{k:v for k,v in pm_full.items() if k not in ['monthly','monthly_ret','equity']}
    }])
    if pm_test:
        port_df = pd.concat([port_df, pd.DataFrame([{
            'Sample':'TEST',
            'UseLiveSpecs': USE_LIVE_SPECS,
            'GlobalRiskPct': GLOBAL_RISK_PCT,
            **{k:v for k,v in pm_test.items() if k not in ['monthly','monthly_ret','equity']}
        }])], ignore_index=True)

    write_df_sheet(wb, 'Portfolio',          port_df)
    write_df_sheet(wb, 'Trades_FULL',        all_t_full)
    write_df_sheet(wb, 'Trades_TEST',        all_t_test if len(all_t_test)>0 else pd.DataFrame())
    write_df_sheet(wb, 'DataQuality',        pd.DataFrame(qc_rows))
    write_df_sheet(wb, 'Warnings',           pd.DataFrame(warnings_rows) if warnings_rows else pd.DataFrame(columns=['Asset','Warning']))

    eq_r=[]; dd_r=[]; rs_r=[]
    for asset in sorted(all_t_full['Activo'].unique()):
        t = all_t_full[all_t_full['Activo']==asset].copy()
        d = daily_equity(t, INITIAL_PER_ASSET)
        if len(d)>0:
            x=d.copy(); x['Asset']=asset; eq_r.append(x[['Date','Asset','Equity','DailyPnL','ReturnPct']])
            y=d.copy(); y['Asset']=asset; dd_r.append(y[['Date','Asset','DrawdownPct']])
            rs=rolling_sharpe(t, INITIAL_PER_ASSET)
            if len(rs)>0:
                rs['Asset']=asset
                rs_r.append(rs[['Date','Asset','ReturnPct','RollingSharpe']])

    d_p=daily_equity(all_t_full, cap0)
    if len(d_p)>0:
        z=d_p.copy(); z['Asset']='PORTFOLIO'; eq_r.append(z[['Date','Asset','Equity','DailyPnL','ReturnPct']])
        dd_r.append(z[['Date','Asset','DrawdownPct']])
        rs_p=rolling_sharpe(all_t_full,cap0)
        if len(rs_p)>0:
            rs_p['Asset']='PORTFOLIO'
            rs_r.append(rs_p[['Date','Asset','ReturnPct','RollingSharpe']])

    write_df_sheet(wb, 'EquityCurve_FULL',   pd.concat(eq_r,ignore_index=True) if eq_r else pd.DataFrame())
    write_df_sheet(wb, 'Drawdown_Curve',     pd.concat(dd_r,ignore_index=True) if dd_r else pd.DataFrame())
    write_df_sheet(wb, 'Rolling_Sharpe',     pd.concat(rs_r,ignore_index=True) if rs_r else pd.DataFrame())
    write_df_sheet(wb, 'Monthly_Heatmap',    monthly_heatmap(all_t_full))
    write_df_sheet(wb, 'Return_Distribution', return_distribution(all_t_full, cap0))

    assumptions_df = pd.DataFrame([
        {'Key':'GLOBAL_RISK_PCT', 'Value':GLOBAL_RISK_PCT},
        {'Key':'USE_LIVE_SPECS', 'Value':USE_LIVE_SPECS},
        {'Key':'PriceSource', 'Value':'parquet_first_then_xlsx'},
        {'Key':'SignalCandle', 'Value':'15:00 / 18:00 MT5 candle label'},
        {'Key':'VolumeFilter', 'Value':'global mean tick_volume from loaded dataset'},
        {'Key':'LRR', 'Value':'(London High - London Low) / mean ATR14 within London'},
        {'Key':'Spread', 'Value':'dynamic spread from candle if available else fallback sp'},
        {'Key':'Commission', 'Value':'history -> order_check -> fallback (if live specs enabled)'},
        {'Key':'Slippage', 'Value':'not simulated randomly'},
    ])
    write_df_sheet(wb, 'Assumptions', assumptions_df)

    wb.save(out_path)
    print(f"\n✅ Excel generado: {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()