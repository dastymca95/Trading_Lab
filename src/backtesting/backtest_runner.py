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

import argparse
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

try:
    from .backtest_config import (
        SEED, INITIAL_PER_ASSET, GLOBAL_RISK_PCT, USE_LIVE_SPECS,
        AUTO_ENABLE_SYMBOL, COMMISSION_LOOKBACK_DAYS,
        TRAIN_END, TEST_START, MAX_BARS, ROLLING_SHARPE_WIN,
        MONTE_CARLO_RUNS, BOOTSTRAP_RUNS, TIMEFRAME_MINUTES,
        ASSET_PARAMS_BASE, OPTIONAL_ASSETS, OPTIONAL_PARAMS,
        COMMISSION_MANUAL_RT,
    )
except ImportError:
    from backtest_config import (
        SEED, INITIAL_PER_ASSET, GLOBAL_RISK_PCT, USE_LIVE_SPECS,
        AUTO_ENABLE_SYMBOL, COMMISSION_LOOKBACK_DAYS,
        TRAIN_END, TEST_START, MAX_BARS, ROLLING_SHARPE_WIN,
        MONTE_CARLO_RUNS, BOOTSTRAP_RUNS, TIMEFRAME_MINUTES,
        ASSET_PARAMS_BASE, OPTIONAL_ASSETS, OPTIONAL_PARAMS,
        COMMISSION_MANUAL_RT,
    )


# ═══════════════════════════════════════════════════════════════════════
# UTILIDADES MT5 / SPECS
# ═══════════════════════════════════════════════════════════════════════

try:
    from .backtest_specs import resolve_asset_params
except ImportError:
    from backtest_specs import resolve_asset_params


# ═══════════════════════════════════════════════════════════════════════
# CARGA DE DATOS + QC
# ═══════════════════════════════════════════════════════════════════════

try:
    from .backtest_data import validate_price_data, load_price_data
except ImportError:
    from backtest_data import validate_price_data, load_price_data


# ═══════════════════════════════════════════════════════════════════════
# MOTOR DE BACKTEST
# ═══════════════════════════════════════════════════════════════════════

def dynamic_spread(sig_row, p):
    mult = p.get('stress_spread_mult', 1.0)   # 1.0 = no stress (default)
    v = sig_row.get('spread_px', np.nan)
    if pd.notna(v) and v > 0:
        return float(v) * mult
    return float(p['sp']) * mult


def generate_signal(ep, lh, ll, cr, av, sig_open, p):
    direction = None
    stype = ''
    if not (pd.isna(lh) or pd.isna(ll)):
        if ep > lh:
            direction = 1
            stype = f"Breakout ALCISTA (London High={round(lh, p['digits'])})"
        elif ep < ll:
            direction = -1
            stype = f"Breakout BAJISTA (London Low={round(ll, p['digits'])})"
    if direction is None and cr > p['atr_mult'] * av:
        direction = -1 if ep > sig_open else 1
        stype = f"Vela grande ({round(cr, p['digits'])} > {p['atr_mult']}xATR)"
    return direction, stype


def calculate_position_size(cap, sl_dist, ep, p):
    ppu = p['cs'] / max(ep, 1) if p['jpy'] else p['cs']
    step = p.get('step', p['ml'])

    lots = (cap * p['risk_pct']) / max(ppu * sl_dist + p['comm'], 1e-8)
    lots = round(lots / step) * step
    lots = max(p['ml'], lots)

    lots_max = (cap * 0.20) / max(ppu * sl_dist + p['comm'], 0.001)
    lots_max = round(lots_max / step) * step
    lots = max(p['ml'], min(lots, lots_max))
    lots = round(lots, 4)

    return lots


def calculate_trade_pnl(xp, ep, direction, lots, sp, p):
    raw = (xp-ep)*direction*lots*(p['cs']/max((ep+xp)/2,1) if p['jpy'] else p['cs'])
    spread_cost = sp * lots * p['cs']
    net = raw - lots * p['comm'] - spread_cost
    result = 'WIN' if net > 0.01 else ('LOSS' if net < -0.01 else 'BE')
    return raw, net, result


def simulate_trade_lifecycle(si, H, L, C, sl, ep, sp, direction, av, p):
    ei  = min(si + 1 + MAX_BARS, len(H))
    fH  = H[si+1:ei]; fL = L[si+1:ei]; fC = C[si+1:ei]
    ib  = direction == 1
    csl = sl; bp = ep; ber = False
    _be = p.get('be_atr_mult', 0.0) * av or sp   # ATR-based BE threshold; fallback to spread
    be_p = (ep + _be) if ib else (ep - _be)
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

    return xp, bars


def _build_daily_filter(asset, df, p):
    """
    Returns frozenset of allowed trading dates based on regime filters,
    or None if no filter is configured for this asset.

    Filters applied (all optional, keyed in p):
      low_vol_pct / low_vol_win  : LOW_VOL regime — daily ATR <= rolling quantile
      atr_cap                    : hard ATR ceiling (US500 guardrail, absolute price units)
      roll_mfe_n / roll_mfe_min  : rolling session mfe/mae median filter (USTEC quality gate)
    """
    lv_pct = p.get('low_vol_pct')
    lv_win = p.get('low_vol_win')
    if lv_pct is None or lv_win is None:
        return None

    # --- LOW_VOL base filter ---
    daily_atr = df.groupby('date')['atr14'].mean()
    rolling_thr = (
        daily_atr
        .rolling(lv_win, min_periods=lv_win // 2)
        .quantile(lv_pct / 100.0)
    )
    allowed = frozenset(
        d for d in daily_atr.index
        if pd.notna(rolling_thr.loc[d]) and daily_atr.loc[d] <= rolling_thr.loc[d]
    )

    # --- Absolute ATR cap (US500 lenient guardrail: p75 of LOW_VOL non-2022 ≈ 1.45) ---
    atr_cap = p.get('atr_cap')
    if atr_cap is not None:
        allowed = frozenset(d for d in allowed if daily_atr.loc[d] <= atr_cap)

    # --- roll_mfe quality gate (USTEC: rolling median of session mfe/mae > threshold) ---
    roll_mfe_n   = p.get('roll_mfe_n')
    roll_mfe_min = p.get('roll_mfe_min')
    if roll_mfe_n is not None and roll_mfe_min is not None:
        # MT5 hours 18 + 19 = UTC 16:00–17:59 = US_MID early session (matches research_runner)
        sess = df[df['time'].dt.hour.isin([18, 19])].copy()
        mfe_rows = []
        for d_s, grp in sess.groupby('date'):
            # Only LOW_VOL days — mirrors research_runner sub_q[date.isin(lv_primary)]
            # Rolling N=20 then counts 20 LOW_VOL sessions, not 20 calendar days
            if d_s not in allowed:
                continue
            o_s = grp['open'].iloc[0]
            mfe = grp['high'].max() - o_s
            mae = o_s - grp['low'].min()
            if mae > 1.0:   # minimum 1 price-unit range to avoid noise
                mfe_rows.append({'date': d_s, 'mfe_mae': mfe / mae})
        if mfe_rows:
            mfe_s = pd.DataFrame(mfe_rows).sort_values('date').reset_index(drop=True)
            mfe_s['mfe_roll'] = (
                mfe_s['mfe_mae']
                .shift(1)
                .rolling(roll_mfe_n, min_periods=roll_mfe_n // 2)
                .median()
            )
            mfe_pass = frozenset(
                mfe_s.loc[mfe_s['mfe_roll'] > roll_mfe_min, 'date']
            )
            allowed = frozenset(d for d in allowed if d in mfe_pass)

    return allowed


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

    daily_filter = _build_daily_filter(asset, df, p)

    sample_start = pd.Timestamp(date_start).normalize() if date_start is not None else pd.Timestamp(df['date'].min())
    sample_end = (
        min(pd.Timestamp(date_end).normalize() - pd.Timedelta(days=1), pd.Timestamp(df['date'].max()))
        if date_end is not None else pd.Timestamp(df['date'].max())
    )

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
        if daily_filter is not None and d not in daily_filter: continue
        vm_ref = sig.get('volume_mean_prior', np.nan)
        if pd.isna(vm_ref) or tv < vm_ref: continue
        if ops.get(d, 0) >= 2: continue
        if pd.isna(av) or av == 0: continue
        if pd.isna(lrr) or lrr <= p['lrr_min']: continue

        direction, stype = generate_signal(ep, lh, ll, cr, av, sig['open'], p)
        if direction is None: continue

        # Direction override: when force_direction is set in params, override signal direction.
        # Signal must still exist (breakout/large-candle + LRR + volume filters all apply).
        # Used by USTEC operational variant: edge validated as LONG-only in US_MID early.
        force_dir = p.get('force_direction')
        if force_dir is not None and direction != force_dir:
            stype = stype + f' [→{"LONG" if force_dir==1 else "SHORT"}]'
            direction = force_dir

        # Execution degradation stress: worse fill (higher entry for LONG, lower for SHORT).
        # entry_slippage_pts=0 by default — no effect on normal runs.
        ep = ep + p.get('entry_slippage_pts', 0.0) * direction

        sl = ep*(1-p['sl_pct']) if direction==1 else ep*(1+p['sl_pct'])
        if not (pd.isna(lh) or pd.isna(ll)):
            sl = max(sl, ll) if direction==1 else min(sl, lh)
        if (direction == 1 and sl >= ep) or (direction == -1 and sl <= ep):
            continue

        sl_dist = max(abs(ep - sl) + sp, ep * 0.0015)
        if sl_dist < 1e-8: continue

        lots = calculate_position_size(cap, sl_dist, ep, p)

        risk_usd = round(cap * p['risk_pct'], 2)
        xp, bars = simulate_trade_lifecycle(si, H, L, C, sl, ep, sp, direction, av, p)

        raw, net, result = calculate_trade_pnl(xp, ep, direction, lots, sp, p)
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
            'Riesgo USD': risk_usd,                     'Comisión RT': p['comm'],
            'PnL Bruto USD': round(raw,2),    'Comisión USD': round(lots*p['comm'],2),
            'PnL Neto USD': round(net,2),      'Capital Tras Op': round(cap,2),
            'Resultado': result,              'Duración (min)': bars*2,
            'Mes': pd.to_datetime(d).strftime('%Y-%m'),
            'Año': pd.to_datetime(d).year,
            'Date': str(pd.to_datetime(d).date()),
        })

    trades_df = pd.DataFrame(trades)
    trades_df.attrs['sample_start'] = sample_start
    trades_df.attrs['sample_end'] = sample_end
    return trades_df, np.array(equity)


# ═══════════════════════════════════════════════════════════════════════
# ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════════════════

try:
    from .backtest_stats import (
        bootstrap_mean_ci, sign_test_pvalue, monte_carlo_dd,
        daily_equity, rolling_sharpe, monthly_heatmap,
        return_distribution, calc_metrics,
    )
except ImportError:
    from backtest_stats import (
        bootstrap_mean_ci, sign_test_pvalue, monte_carlo_dd,
        daily_equity, rolling_sharpe, monthly_heatmap,
        return_distribution, calc_metrics,
    )


# ═══════════════════════════════════════════════════════════════════════
# EXCEL
# ═══════════════════════════════════════════════════════════════════════

try:
    from .backtest_reporter import HFILL, HFONT, thin, BRD, write_df_sheet
except ImportError:
    from backtest_reporter import HFILL, HFONT, thin, BRD, write_df_sheet


def aggregate_portfolio(all_full, all_test, cap0):
    all_t_full = pd.concat(all_full).sort_values('Fecha Apertura').reset_index(drop=True)
    all_t_full['#'] = range(1, len(all_t_full)+1)
    all_t_test = pd.concat(all_test).sort_values('Fecha Apertura').reset_index(drop=True) if all_test else pd.DataFrame()
    all_t_full.attrs['sample_start'] = min(pd.Timestamp(t.attrs.get('sample_start')) for t in all_full)
    all_t_full.attrs['sample_end'] = max(pd.Timestamp(t.attrs.get('sample_end')) for t in all_full)
    if len(all_t_test) > 0:
        all_t_test.attrs['sample_start'] = min(pd.Timestamp(t.attrs.get('sample_start')) for t in all_test)
        all_t_test.attrs['sample_end'] = max(pd.Timestamp(t.attrs.get('sample_end')) for t in all_test)

    cap  = cap0
    eq = [cap]
    for pnl in all_t_full['PnL Neto USD'].values:
        cap = max(cap + pnl, 0.01)
        eq.append(cap)
    eq = np.array(eq)
    pm_full = calc_metrics(
        all_t_full,
        eq,
        cap0,
        sample_start=all_t_full.attrs.get('sample_start'),
        sample_end=all_t_full.attrs.get('sample_end'),
    )

    if len(all_t_test) > 0:
        cap = cap0
        eq_t = [cap]
        for pnl in all_t_test['PnL Neto USD'].values:
            cap = max(cap + pnl, 0.01)
            eq_t.append(cap)
        eq_t = np.array(eq_t)
        pm_test = calc_metrics(
            all_t_test,
            eq_t,
            cap0,
            sample_start=all_t_test.attrs.get('sample_start'),
            sample_end=all_t_test.attrs.get('sample_end'),
        )
    else:
        pm_test = {}

    return all_t_full, all_t_test, pm_full, pm_test


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def _fmt_dir(p):
    fd = p.get('force_direction')
    if fd == 1:  return 'forced=LONG'
    if fd == -1: return 'forced=SHORT'
    return 'bidirectional'


def _fmt_m(m, period, W=72):
    """One result row: period label + key metrics."""
    if not m:
        return f"  {period:<6}  — no trades"
    sh = f"{m['sharpe']:>7.3f}" if pd.notna(m.get('sharpe', float('nan'))) else "      —"
    return (
        f"  {period:<6}  {m['n']:>5}  {m['wr']:>5.1f}%  {m['pf']:>5.2f}  "
        f"{m['ret']:>+7.2f}%  {m['mdd']:>7.2f}%  "
        f"{m['aw']:>+7.2f}  {m['al']:>+7.2f}  {m['exp']:>+8.4f}  {sh}"
    )


def main():
    parser = argparse.ArgumentParser(description='Backtest operativo hybrid breakout system')
    parser.add_argument('--be_atr_mult', type=float, default=None,
                        help='Override solo para backtesting del umbral BE ATR.')
    args = parser.parse_args()

    W = 76   # console width

    print("╔" + "═"*(W-2) + "╗")
    print("║   BACKTEST OPERATIVO — HYBRID BREAKOUT SYSTEM" + " "*(W-48) + "║")
    print("╚" + "═"*(W-2) + "╝")

    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR   = os.path.join(_REPO_ROOT, "data", "backtesting")
    ts         = datetime.now().strftime('%Y%m%d_%H%M')
    REPORTS_DIR = os.path.join(_REPO_ROOT, "reports", "backtests")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path    = os.path.join(REPORTS_DIR, f"Backtest_Hibrido_Aligned_{ts}.xlsx")

    ASSET_PARAMS = resolve_asset_params(DATA_DIR)
    if args.be_atr_mult is not None:
        for _asset, _p in ASSET_PARAMS.items():
            _p['be_atr_mult'] = float(args.be_atr_mult)
        print(f"  [CLI override] be_atr_mult = {args.be_atr_mult}")

    # solo mantener activos con archivo de precios disponible
    final_assets = {}
    for asset, p in ASSET_PARAMS.items():
        pq = os.path.join(DATA_DIR, f"{asset}_Data.parquet")
        xl = os.path.join(DATA_DIR, f"{asset}_Data.xlsx")
        if os.path.exists(pq) or os.path.exists(xl):
            final_assets[asset] = p

    if not final_assets:
        print("\n  ❌ Sin archivos de datos.")
        return

    specs_mode = 'live_mt5' if USE_LIVE_SPECS else 'json/base'
    print(f"\n  Specs      : {specs_mode}")
    print(f"  Risk/asset : {GLOBAL_RISK_PCT:.2%}  |  Capital : ${INITIAL_PER_ASSET:.0f}/asset")
    print(f"  Period     : full history → {TRAIN_END.date()}  /  test from {TEST_START.date()}")
    print(f"  Assets     : {', '.join(final_assets.keys())}")
    print(f"  Data dir   : {DATA_DIR}")

    # ── Data loading ──────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  DATA")
    print(f"{'─'*W}")

    asset_data = {}
    qc_rows = []
    warnings_rows = []

    for asset, p in final_assets.items():
        df, lr, vm, qc = load_price_data(asset, DATA_DIR, p['digits'])
        if df is not None:
            asset_data[asset] = (df, lr, vm)
            qc_rows.append(qc)
            if qc['Gap %'] > 0.5:
                warnings_rows.append({'Asset':asset, 'Warning':f"Gaps: {qc['Gap Count']} ({qc['Gap %']}%)"})
            if qc['Duplicates Removed'] > 0:
                warnings_rows.append({'Asset':asset, 'Warning':f"Duplicados: {qc['Duplicates Removed']}"})

    if not asset_data:
        print("\n  ❌ Sin datos cargables.")
        return

    # ── Backtest results ──────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  BACKTEST RESULTS")
    print(f"{'─'*W}")

    _RHDR = (
        f"  {'':6}  {'n':>5}  {'WR%':>6}  {'PF':>5}  "
        f"{'ret%':>8}  {'MDD%*':>7}  "
        f"{'avg_win':>7}  {'avg_loss':>8}  {'exp/trd':>8}  {'Sharpe**':>8}"
    )
    _RSEP = f"  {'─'*74}"

    results_full = {}
    results_test = {}
    all_full = []
    all_test = []

    for asset, (df, lr, vm) in asset_data.items():
        p = final_assets[asset]
        cap = INITIAL_PER_ASSET

        t_f, e_f = run_backtest(asset, df, lr, vm, p, cap, None, None, 'FULL')
        m_f = calc_metrics(
            t_f, e_f, cap,
            sample_start=t_f.attrs.get('sample_start'),
            sample_end=t_f.attrs.get('sample_end'),
        )
        results_full[asset] = m_f
        if len(t_f) > 0:
            t_f['asset'] = asset
            all_full.append(t_f)

        t_t, e_t = run_backtest(asset, df, lr, vm, p, cap, TEST_START, None, 'TEST')
        m_t = calc_metrics(
            t_t, e_t, cap,
            sample_start=t_t.attrs.get('sample_start'),
            sample_end=t_t.attrs.get('sample_end'),
        )
        results_test[asset] = m_t
        if len(t_t) > 0:
            all_test.append(t_t)

        comm_src = p.get('comm_source', 'base/json')
        print(
            f"\n  {asset}"
            f"  │  comm={p['comm']:.2f} ({comm_src})"
            f"  │  risk={p['risk_pct']:.2%}"
            f"  │  hours={p['hours']}"
            f"  │  {_fmt_dir(p)}"
        )
        if p.get('low_vol_pct') is not None:
            lv_tag = f"LOW_VOL(p{p['low_vol_pct']},w{p['low_vol_win']})"
            rm_tag = (f" + roll_mfe>{p['roll_mfe_min']}" if p.get('roll_mfe_n') else "")
            ac_tag = (f" + ATR_cap={p['atr_cap']}" if p.get('atr_cap') else "")
            print(f"  {'':6}  filter: {lv_tag}{rm_tag}{ac_tag}")
        print(_RSEP)
        print(_RHDR)
        print(_RSEP)
        print(_fmt_m(m_f, 'FULL'))
        print(_fmt_m(m_t, 'TEST'))
        print(_RSEP)

    if not all_full:
        print("\n  ❌ Sin trades.")
        return

    # ── Portfolio summary ─────────────────────────────────────────────────────
    cap0 = INITIAL_PER_ASSET * len(results_full)
    all_t_full, all_t_test, pm_full, pm_test = aggregate_portfolio(all_full, all_test, cap0)

    print(f"\n{'─'*W}")
    print(f"  PORTFOLIO  (${cap0:.0f} total  |  {len(results_full)} assets)")
    print(f"{'─'*W}")
    print(_RSEP)
    print(_RHDR)
    print(_RSEP)
    print(_fmt_m(pm_full, 'FULL'))
    print(_fmt_m(pm_test if pm_test else {}, 'TEST'))
    print(_RSEP)
    print(f"   * MDD%: trade-by-trade equity (intra-day resolution). Differs from daily_equity MDD.")
    print(f"  ** Sharpe/Sortino: monthly PnL / initial_cap, sqrt(12). Not equity-weighted. See Assumptions.")

    # ── Excel generation ──────────────────────────────────────────────────────
    print(f"\n  Generando Excel...", end='', flush=True)
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
                [k for k in [
                    'sl_pct','trail_mult','risk_pct','lrr_min','hours','dow','atr_mult',
                    'be_atr_mult','force_direction','low_vol_pct','low_vol_win',
                    'roll_mfe_n','roll_mfe_min','atr_cap'
                ] if k in p]}, ensure_ascii=False),
            'Trades_FULL':mf.get('n',0), 'RetPct_FULL':mf.get('ret',0),
            'PF_FULL':mf.get('pf',0), 'Sharpe_cap0_FULL':mf.get('sharpe',0),
            'Sortino_cap0_FULL':mf.get('sortino',0), 'MDD_trade_FULL':mf.get('mdd',0),
            'Trades_TEST':mt.get('n',0), 'RetPct_TEST':mt.get('ret',0),
            'PF_TEST':mt.get('pf',0), 'Sharpe_cap0_TEST':mt.get('sharpe',0),
            'Sortino_cap0_TEST':mt.get('sortino',0), 'MDD_trade_TEST':mt.get('mdd',0),
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
    port_df = port_df.rename(columns={
        'sharpe':  'Sharpe_cap0',
        'sortino': 'Sortino_cap0',
        'mdd':     'MDD_trade_pct',
    })

    write_df_sheet(wb, 'Portfolio',          port_df)
    write_df_sheet(wb, 'Trades_FULL',        all_t_full)
    write_df_sheet(wb, 'Trades_TEST',        all_t_test if len(all_t_test)>0 else pd.DataFrame())
    write_df_sheet(wb, 'DataQuality',        pd.DataFrame(qc_rows))
    write_df_sheet(wb, 'Warnings',           pd.DataFrame(warnings_rows) if warnings_rows else pd.DataFrame(columns=['Asset','Warning']))

    eq_r=[]; dd_r=[]; rs_r=[]
    for asset in sorted(all_t_full['Activo'].unique()):
        t = all_t_full[all_t_full['Activo']==asset].copy()
        d = daily_equity(
            t, INITIAL_PER_ASSET,
            sample_start=t.attrs.get('sample_start'),
            sample_end=t.attrs.get('sample_end'),
        )
        if len(d)>0:
            x=d.copy(); x['Asset']=asset
            x=x.rename(columns={'ReturnPct':'ReturnPct_cap0'})
            eq_r.append(x[['Date','Asset','Equity','DailyPnL','ReturnPct_cap0']])
            y=d.copy(); y['Asset']=asset; dd_r.append(y[['Date','Asset','DrawdownPct']])
            rs=rolling_sharpe(
                t, INITIAL_PER_ASSET,
                sample_start=t.attrs.get('sample_start'),
                sample_end=t.attrs.get('sample_end'),
            )
            if len(rs)>0:
                rs['Asset']=asset
                rs=rs.rename(columns={'ReturnPct':'ReturnPct_cap0'})
                rs_r.append(rs[['Date','Asset','ReturnPct_cap0','RollingSharpe']])

    d_p=daily_equity(
        all_t_full, cap0,
        sample_start=all_t_full.attrs.get('sample_start'),
        sample_end=all_t_full.attrs.get('sample_end'),
    )
    if len(d_p)>0:
        z=d_p.copy(); z['Asset']='PORTFOLIO'
        z=z.rename(columns={'ReturnPct':'ReturnPct_cap0'})
        eq_r.append(z[['Date','Asset','Equity','DailyPnL','ReturnPct_cap0']])
        dd_r.append(z[['Date','Asset','DrawdownPct']])
        rs_p=rolling_sharpe(
            all_t_full, cap0,
            sample_start=all_t_full.attrs.get('sample_start'),
            sample_end=all_t_full.attrs.get('sample_end'),
        )
        if len(rs_p)>0:
            rs_p['Asset']='PORTFOLIO'
            rs_p=rs_p.rename(columns={'ReturnPct':'ReturnPct_cap0'})
            rs_r.append(rs_p[['Date','Asset','ReturnPct_cap0','RollingSharpe']])

    write_df_sheet(wb, 'EquityCurve_FULL',   pd.concat(eq_r,ignore_index=True) if eq_r else pd.DataFrame())
    write_df_sheet(wb, 'Drawdown_Curve',     pd.concat(dd_r,ignore_index=True) if dd_r else pd.DataFrame())
    write_df_sheet(wb, 'Rolling_Sharpe',     pd.concat(rs_r,ignore_index=True) if rs_r else pd.DataFrame())
    write_df_sheet(wb, 'Monthly_Heatmap',    monthly_heatmap(all_t_full))
    write_df_sheet(
        wb,
        'Return_Distribution',
        return_distribution(
            all_t_full,
            cap0,
            sample_start=all_t_full.attrs.get('sample_start'),
            sample_end=all_t_full.attrs.get('sample_end'),
        ),
    )

    assumptions_df = pd.DataFrame([
        {'Key':'GLOBAL_RISK_PCT', 'Value':GLOBAL_RISK_PCT},
        {'Key':'USE_LIVE_SPECS', 'Value':USE_LIVE_SPECS},
        {'Key':'PriceSource', 'Value':'parquet_first_then_xlsx'},
        {'Key':'SignalCandle', 'Value':'15:00 / 18:00 MT5 candle label'},
        {'Key':'VolumeFilter', 'Value':'causal expanding mean tick_volume shifted(1)'},
        {'Key':'LRR', 'Value':'(London High - London Low) / mean ATR14 within London'},
        {'Key':'Spread', 'Value':'dynamic spread from candle if available else fallback sp'},
        {'Key':'Commission', 'Value':'history -> order_check -> fallback (if live specs enabled)'},
        {'Key':'Slippage', 'Value':'not simulated randomly'},
        # ── Methodology notes ──────────────────────────────────────────────────
        {'Key':'Sharpe_cap0_method', 'Value':
            'Monthly PnL / initial_cap_per_asset * 100 over all months in sample window, '
            'including zero-trade months; annualized sqrt(12). '
            'Denominator is fixed initial capital, not equity at month start. '
            'Consistent internally; not equity-weighted. Label: Sharpe_cap0.'},
        {'Key':'Sortino_cap0_method', 'Value':
            'Same monthly base as Sharpe_cap0, including zero-trade months in sample window. '
            'Label: Sortino_cap0.'},
        {'Key':'MDD_trade_pct', 'Value':
            'Max drawdown from trade-by-trade equity array (one point per closed trade). '
            'Differs from daily MDD (Drawdown_Curve sheet) when >1 trade occurs on same day.'},
        {'Key':'DrawdownPct_daily', 'Value':
            'Drawdown_Curve sheet: day-grouped equity = cap0 + cumsum(DailyPnL). '
            'No floor. Lower intra-day resolution than MDD_trade_pct.'},
        {'Key':'ReturnPct_cap0', 'Value':
            'EquityCurve_FULL and Rolling_Sharpe sheets: DailyPnL / initial_cap * 100 '
            'across all business days in sample window. '
            'Denominator is fixed initial capital, not current equity. '
            'Not a true daily return on equity.'},
    ])
    write_df_sheet(wb, 'Assumptions', assumptions_df)

    wb.save(out_path)
    print(f" ✓")
    print(f"\n  {out_path}")
    print(f"\n{'═'*W}")
    print("  Done.")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
