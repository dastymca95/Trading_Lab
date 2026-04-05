#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RISK SENSITIVITY RUNNER — v1 PROFESSIONAL
Barrido de escenarios de riesgo para el sistema London Range Breakout.

OBJETIVO
- Probar múltiples niveles de risk_pct
- Comparar FULL vs TEST
- Encontrar zonas robustas de riesgo
- Generar Excel + CSV + gráficos

REQUISITOS
    pip install pandas numpy scipy openpyxl pyarrow matplotlib MetaTrader5

USO
    python risk_sensitivity_runner_v1.py

SALIDA
- Risk_Sensitivity_<timestamp>.xlsx
- Risk_Sensitivity_<timestamp>.csv
- carpeta plots_risk_<timestamp> con gráficos PNG
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Alignment

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

from backtest_config import (
    INITIAL_PER_ASSET, USE_LIVE_SPECS,
    TEST_START, MAX_BARS,
    ASSET_PARAMS_BASE, OPTIONAL_PARAMS,
    COMMISSION_MANUAL_RT,
)

# Grilla de riesgo: específico del barrido
RISK_GRID = [
    0.0025, 0.0050, 0.0075, 0.0100, 0.0125,
    0.0150, 0.0175, 0.0200, 0.0250, 0.0300,
    0.0350, 0.0400, 0.0500, 0.0550, 0.0600
]


# ═══════════════════════════════════════════════════════════════════════
# MT5 / SPECS
# ═══════════════════════════════════════════════════════════════════════

from backtest_specs import (
    connect_mt5, safe_symbol,
    _comm_from_history_rt, _comm_from_order_check_rt, get_live_commission_rt,
    load_broker_specs_json, apply_json_specs,
)


def apply_risk(params: dict, risk_pct: float) -> dict:
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
        print("  ⚠️ MT5 no disponible. Se usarán parámetros base/fallback.")
        for asset, p in base_params.items():
            q = p.copy()
            q['comm_source'] = 'fallback_no_mt5'
            out[asset] = q
        return out

    for asset, p in base_params.items():
        q = p.copy()
        if not safe_symbol(asset):
            q['comm_source'] = 'fallback_no_symbol'
            out[asset] = q
            continue

        info = mt5.symbol_info(asset)
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
        out[asset] = q

    try:
        mt5.shutdown()
    except Exception:
        pass

    return out


def resolve_asset_base_params(data_dir):
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

    return params


# ═══════════════════════════════════════════════════════════════════════
# DATOS
# ═══════════════════════════════════════════════════════════════════════

from backtest_data import validate_price_data, load_price_data


# ═══════════════════════════════════════════════════════════════════════
# MOTOR
# ═══════════════════════════════════════════════════════════════════════

def dynamic_spread(sig_row, p):
    v = sig_row.get('spread_px', np.nan)
    if pd.notna(v) and v > 0:
        return float(v)
    return float(p['sp'])


def run_backtest(asset, df, lr, vm, p, cap_start, date_start=None, date_end=None, label='FULL'):
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
    DOW = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday'}

    for _, sig in sg.iterrows():
        d   = sig['date']; dow = sig['dow']; ep = sig['close']
        si  = int(sig['idx']); av = sig['atr14']
        lh  = sig['lh']; ll = sig['ll']; tv = sig['tick_volume']
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
        xp = ep; bars = MAX_BARS

        for i in range(len(fH)):
            h2, l2 = fH[i], fL[i]
            if np.isnan(h2): break
            if ib and l2 <= csl: xp = csl; bars = i+1; break
            if not ib and h2 >= csl: xp = csl; bars = i+1; break
            if ib: bp = max(bp, h2); ber = ber or h2 >= be_p
            else:  bp = min(bp, l2); ber = ber or l2 <= be_p
            if not ber: continue
            ns = (bp - av*p['trail_mult']) if ib else (bp + av*p['trail_mult'])
            ns = max(ns, be_p) if ib else min(ns, be_p)
            csl = max(csl, ns) if ib else min(csl, ns)
        else:
            li = len(fC) - 1
            while li >= 0 and np.isnan(fC[li]): li -= 1
            xp = fC[li] if li >= 0 else ep
            bars = len(fH)

        raw = (xp-ep)*direction*lots*(p['cs']/max((ep+xp)/2,1) if p['jpy'] else p['cs'])
        spread_cost = sp * lots * p['cs']
        net = raw - lots * p['comm'] - spread_cost
        result = 'WIN' if net > 0.01 else ('LOSS' if net < -0.01 else 'BE')
        cap = max(cap + net, 0.01)
        equity.append(cap)
        ops[d] = ops.get(d, 0) + 1

        trades.append({
            '#': len(trades)+1,
            'Sample': label,
            'Activo': asset,
            'Fecha Apertura': str(pd.Timestamp(sig['time']))[:16],
            'Fecha Cierre': str(pd.Timestamp(sig['time'])+pd.Timedelta(minutes=bars*2))[:16],
            'Hora Señal MT5': f"{sig['sh']}:00",
            'Dirección': 'BUY' if direction==1 else 'SELL',
            'Tipo Señal': stype,
            'ATR': round(av, p['digits']),
            'Ratio L/ATR': round(lrr, 2) if pd.notna(lrr) else np.nan,
            'Spread Usado': round(sp, p['digits']),
            'Precio Entrada': round(ep, p['digits']),
            'SL Inicial': round(sl, p['digits']),
            'Precio Salida': round(xp, p['digits']),
            'Distancia SL pts': round(sl_dist, p['digits']),
            'Lotes': lots,
            'Riesgo %': round(p['risk_pct'] * 100, 3),
            'Comisión RT': p['comm'],
            'PnL Bruto USD': round(raw,2),
            'Comisión USD': round(lots*p['comm'],2),
            'PnL Neto USD': round(net,2),
            'Capital Tras Op': round(cap,2),
            'Resultado': result,
            'Duración (min)': bars*2,
            'Mes': pd.to_datetime(d).strftime('%Y-%m'),
            'Date': str(pd.to_datetime(d).date()),
        })

    return pd.DataFrame(trades), np.array(equity)


# ═══════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════

from backtest_stats import (
    bootstrap_mean_ci, sign_test_pvalue, monte_carlo_dd,
    daily_equity, calc_metrics,
)


# ═══════════════════════════════════════════════════════════════════════
# EXCEL
# ═══════════════════════════════════════════════════════════════════════

from backtest_reporter import HFILL, HFONT, write_df_sheet


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def save_line_plot(df, x, y, title, path, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(df[x], df[y], marker='o')
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_dual_sample_plot(df, metric_full, metric_test, title, path, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(df['risk_pct'], df[metric_full], marker='o', label='FULL')
    plt.plot(df['risk_pct'], df[metric_test], marker='o', label='TEST')
    plt.xlabel('risk_pct')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_scatter_plot(df, x, y, title, path, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x], df[y])
    for _, row in df.iterrows():
        plt.annotate(f"{row['risk_pct']:.3f}", (row[x], row[y]), fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   RISK SENSITIVITY RUNNER — v1 PROFESSIONAL             ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    data_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_xlsx = os.path.join(data_dir, f"Risk_Sensitivity_{ts}.xlsx")
    out_csv = os.path.join(data_dir, f"Risk_Sensitivity_{ts}.csv")
    plot_dir = os.path.join(data_dir, f"plots_risk_{ts}")
    os.makedirs(plot_dir, exist_ok=True)

    base_params = resolve_asset_base_params(data_dir)

    # solo activos con archivo de precios
    asset_data = {}
    qc_rows = []
    final_assets = {}

    print("\n2. Cargando precios...")
    for asset, p in base_params.items():
        pq = os.path.join(data_dir, f"{asset}_Data.parquet")
        xl = os.path.join(data_dir, f"{asset}_Data.xlsx")
        if not (os.path.exists(pq) or os.path.exists(xl)):
            continue
        df, lr, vm, qc = load_price_data(asset, data_dir, p['digits'])
        if df is not None:
            asset_data[asset] = (df, lr, vm)
            final_assets[asset] = p.copy()
            qc_rows.append(qc)

    if not asset_data:
        print("\n❌ No hay archivos de precios disponibles.")
        return

    print("\n3. Ejecutando barrido de riesgos...")
    summary_rows = []
    per_asset_rows = []

    for risk_pct in RISK_GRID:
        print(f"\n   ▶ risk_pct={risk_pct:.4f}")
        params_for_risk = apply_risk(final_assets, risk_pct)

        all_full = []
        all_test = []
        results_full = {}
        results_test = {}

        for asset, (df, lr, vm) in asset_data.items():
            p = params_for_risk[asset]
            cap = INITIAL_PER_ASSET

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

            per_asset_rows.append({
                'risk_pct': risk_pct,
                'asset': asset,
                'comm_rt': p['comm'],
                'comm_source': p.get('comm_source', 'json/base'),
                'ret_full': m_f.get('ret', np.nan),
                'mdd_full': m_f.get('mdd', np.nan),
                'mc_dd_p5_full': m_f.get('mc_dd_p5', np.nan),
                'sharpe_full': m_f.get('sharpe', np.nan),
                'calmar_full': m_f.get('calmar', np.nan),
                'ret_test': m_t.get('ret', np.nan),
                'mdd_test': m_t.get('mdd', np.nan),
                'mc_dd_p5_test': m_t.get('mc_dd_p5', np.nan),
                'sharpe_test': m_t.get('sharpe', np.nan),
                'calmar_test': m_t.get('calmar', np.nan),
            })

        if not all_full:
            continue

        all_t_full = pd.concat(all_full).sort_values('Fecha Apertura').reset_index(drop=True)
        cap0 = INITIAL_PER_ASSET * len(results_full)
        cap  = cap0
        eq = [cap]
        for pnl in all_t_full['PnL Neto USD'].values:
            cap = max(cap + pnl, 0.01)
            eq.append(cap)
        eq = np.array(eq)
        pm_full = calc_metrics(all_t_full, eq, cap0)

        if all_test:
            all_t_test = pd.concat(all_test).sort_values('Fecha Apertura').reset_index(drop=True)
            cap = cap0
            eq_t = [cap]
            for pnl in all_t_test['PnL Neto USD'].values:
                cap = max(cap + pnl, 0.01)
                eq_t.append(cap)
            eq_t = np.array(eq_t)
            pm_test = calc_metrics(all_t_test, eq_t, cap0)
        else:
            pm_test = {}

        summary_rows.append({
            'risk_pct': risk_pct,
            'use_live_specs': USE_LIVE_SPECS,
            'n_assets': len(results_full),
            'ret_full': pm_full.get('ret', np.nan),
            'mdd_full': pm_full.get('mdd', np.nan),
            'mc_dd_p5_full': pm_full.get('mc_dd_p5', np.nan),
            'mc_dd_p50_full': pm_full.get('mc_dd_p50', np.nan),
            'mc_dd_p95_full': pm_full.get('mc_dd_p95', np.nan),
            'sharpe_full': pm_full.get('sharpe', np.nan),
            'sortino_full': pm_full.get('sortino', np.nan),
            'calmar_full': pm_full.get('calmar', np.nan),
            'pf_full': pm_full.get('pf', np.nan),
            'wr_full': pm_full.get('wr', np.nan),
            'avg_day_full': pm_full.get('avg_day', np.nan),
            'worst_day_usd_full': pm_full.get('worst_day_usd', np.nan),
            'best_day_usd_full': pm_full.get('best_day_usd', np.nan),
            'ret_test': pm_test.get('ret', np.nan),
            'mdd_test': pm_test.get('mdd', np.nan),
            'mc_dd_p5_test': pm_test.get('mc_dd_p5', np.nan),
            'mc_dd_p50_test': pm_test.get('mc_dd_p50', np.nan),
            'mc_dd_p95_test': pm_test.get('mc_dd_p95', np.nan),
            'sharpe_test': pm_test.get('sharpe', np.nan),
            'sortino_test': pm_test.get('sortino', np.nan),
            'calmar_test': pm_test.get('calmar', np.nan),
            'pf_test': pm_test.get('pf', np.nan),
            'wr_test': pm_test.get('wr', np.nan),
            'avg_day_test': pm_test.get('avg_day', np.nan),
            'worst_day_usd_test': pm_test.get('worst_day_usd', np.nan),
            'best_day_usd_test': pm_test.get('best_day_usd', np.nan),
        })

        print(
            f"      FULL ret={pm_full.get('ret',0):+.1f}% | mdd={pm_full.get('mdd',0):.2f}% | "
            f"mc_p5={pm_full.get('mc_dd_p5',0):.2f}% | calmar={pm_full.get('calmar',0)}"
        )
        if pm_test:
            print(
                f"      TEST ret={pm_test.get('ret',0):+.1f}% | mdd={pm_test.get('mdd',0):.2f}% | "
                f"mc_p5={pm_test.get('mc_dd_p5',0):.2f}% | calmar={pm_test.get('calmar',0)}"
            )

    if not summary_rows:
        print("\n❌ No se generaron resultados.")
        return

    summary_df = pd.DataFrame(summary_rows).sort_values('risk_pct').reset_index(drop=True)
    per_asset_df = pd.DataFrame(per_asset_rows).sort_values(['asset', 'risk_pct']).reset_index(drop=True)
    summary_df.to_csv(out_csv, index=False)

    print("\n4. Generando gráficos...")
    save_dual_sample_plot(summary_df, 'ret_full', 'ret_test', 'Return % vs Risk', os.path.join(plot_dir, '01_return_vs_risk.png'), 'Return %')
    save_dual_sample_plot(summary_df, 'mdd_full', 'mdd_test', 'Max Drawdown vs Risk', os.path.join(plot_dir, '02_mdd_vs_risk.png'), 'Max Drawdown %')
    save_dual_sample_plot(summary_df, 'mc_dd_p5_full', 'mc_dd_p5_test', 'Monte Carlo DD p5 vs Risk', os.path.join(plot_dir, '03_mcddp5_vs_risk.png'), 'MC DD p5 %')
    save_dual_sample_plot(summary_df, 'calmar_full', 'calmar_test', 'Calmar vs Risk', os.path.join(plot_dir, '04_calmar_vs_risk.png'), 'Calmar')
    save_dual_sample_plot(summary_df, 'sharpe_full', 'sharpe_test', 'Sharpe vs Risk', os.path.join(plot_dir, '05_sharpe_vs_risk.png'), 'Sharpe')
    save_dual_sample_plot(summary_df, 'worst_day_usd_full', 'worst_day_usd_test', 'Worst Day USD vs Risk', os.path.join(plot_dir, '06_worstday_vs_risk.png'), 'Worst Day USD')

    save_scatter_plot(summary_df, 'mdd_test', 'ret_test', 'Risk-Return Frontier (TEST)', os.path.join(plot_dir, '07_frontier_test.png'), 'MDD TEST %', 'Return TEST %')
    save_scatter_plot(summary_df, 'mc_dd_p5_test', 'ret_test', 'MC p5 vs Return (TEST)', os.path.join(plot_dir, '08_mcddp5_return_test.png'), 'MC DD p5 TEST %', 'Return TEST %')

    print("\n5. Generando Excel...")
    wb = Workbook()
    ws = wb.active
    ws.title = 'Resumen'

    ws.append(summary_df.columns.tolist())
    for row in summary_df.itertuples(index=False):
        ws.append(list(row))
    for cell in ws[1]:
        cell.fill = HFILL
        cell.font = HFONT
        cell.alignment = Alignment(horizontal='center', wrap_text=True)

    write_df_sheet(wb, 'Per_Asset', per_asset_df)
    write_df_sheet(wb, 'DataQuality', pd.DataFrame(qc_rows))

    assumptions_df = pd.DataFrame([
        {'Key':'USE_LIVE_SPECS', 'Value':USE_LIVE_SPECS},
        {'Key':'INITIAL_PER_ASSET', 'Value':INITIAL_PER_ASSET},
        {'Key':'RISK_GRID', 'Value':', '.join([str(x) for x in RISK_GRID])},
        {'Key':'PriceSource', 'Value':'parquet_first_then_xlsx'},
        {'Key':'SignalCandle', 'Value':'15:00 / 18:00 MT5 candle label'},
        {'Key':'VolumeFilter', 'Value':'global mean tick_volume from loaded dataset'},
        {'Key':'LRR', 'Value':'(London High - London Low) / mean ATR14 within London'},
        {'Key':'Spread', 'Value':'dynamic spread from candle if available else fallback sp'},
        {'Key':'Commission', 'Value':'history -> order_check -> fallback (if USE_LIVE_SPECS=True)'},
        {'Key':'Slippage', 'Value':'not simulated randomly'},
        {'Key':'TestStart', 'Value':str(TEST_START.date())},
    ])
    write_df_sheet(wb, 'Assumptions', assumptions_df)

    top_candidates = summary_df.copy()
    top_candidates = top_candidates[
        (top_candidates['mc_dd_p5_test'] > -10) &
        (top_candidates['mdd_test'] > -10)
    ].copy()
    if len(top_candidates) > 0:
        top_candidates = top_candidates.sort_values(['calmar_test', 'ret_test'], ascending=[False, False]).head(10)
    write_df_sheet(wb, 'Top_Candidates', top_candidates)

    wb.save(os.path.join(data_dir, f"Risk_Sensitivity_{ts}.xlsx"))

    print(f"\n✅ CSV:  {os.path.basename(out_csv)}")
    print(f"✅ Excel: Risk_Sensitivity_{ts}.xlsx")
    print(f"✅ Plots: {os.path.basename(plot_dir)}")


if __name__ == "__main__":
    main()