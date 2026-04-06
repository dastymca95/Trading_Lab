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
    TEST_START,
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
    get_live_commission_rt,
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

from backtest_data import load_price_data


# ═══════════════════════════════════════════════════════════════════════
# MOTOR
# ═══════════════════════════════════════════════════════════════════════

from backtest_runner import run_backtest


# ═══════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════

from backtest_stats import calc_metrics


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