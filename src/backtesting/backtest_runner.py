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

from backtest_specs import resolve_asset_params


# ═══════════════════════════════════════════════════════════════════════
# CARGA DE DATOS + QC
# ═══════════════════════════════════════════════════════════════════════

from backtest_data import validate_price_data, load_price_data


# ═══════════════════════════════════════════════════════════════════════
# MOTOR DE BACKTEST
# ═══════════════════════════════════════════════════════════════════════

def dynamic_spread(sig_row, p):
    v = sig_row.get('spread_px', np.nan)
    if pd.notna(v) and v > 0:
        return float(v)
    return float(p['sp'])


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


def simulate_trade_lifecycle(si, H, L, C, sl, ep, sp, direction, av, p):
    ei  = min(si + 1 + MAX_BARS, len(H))
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

    return xp, bars


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

        direction, stype = generate_signal(ep, lh, ll, cr, av, sig['open'], p)
        if direction is None: continue

        sl = ep*(1-p['sl_pct']) if direction==1 else ep*(1+p['sl_pct'])
        if not (pd.isna(lh) or pd.isna(ll)):
            sl = max(sl, ll) if direction==1 else min(sl, lh)

        sl_dist = max(abs(ep - sl) + sp, ep * 0.0015)
        if sl_dist < 1e-8: continue

        lots = calculate_position_size(cap, sl_dist, ep, p)

        xp, bars = simulate_trade_lifecycle(si, H, L, C, sl, ep, sp, direction, av, p)

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

from backtest_stats import (
    bootstrap_mean_ci, sign_test_pvalue, monte_carlo_dd,
    daily_equity, rolling_sharpe, monthly_heatmap,
    return_distribution, calc_metrics,
)


# ═══════════════════════════════════════════════════════════════════════
# EXCEL
# ═══════════════════════════════════════════════════════════════════════

from backtest_reporter import HFILL, HFONT, thin, BRD, write_df_sheet


def aggregate_portfolio(all_full, all_test, cap0):
    all_t_full = pd.concat(all_full).sort_values('Fecha Apertura').reset_index(drop=True)
    all_t_full['#'] = range(1, len(all_t_full)+1)
    all_t_test = pd.concat(all_test).sort_values('Fecha Apertura').reset_index(drop=True) if all_test else pd.DataFrame()

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

    return all_t_full, all_t_test, pm_full, pm_test


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
    cap0 = INITIAL_PER_ASSET * len(results_full)
    all_t_full, all_t_test, pm_full, pm_test = aggregate_portfolio(all_full, all_test, cap0)

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