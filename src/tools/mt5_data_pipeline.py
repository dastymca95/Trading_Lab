#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MT5 EXTRACT AND PACKAGE — v3
Correcciones vs v2:
  - Los datos se guardan en .parquet (sin límite de filas, 5x más rápido)
  - El Excel solo contiene Specs + QC (sin la hoja Data)
  - El backtest lee .parquet automáticamente si existe, o .xlsx si no

Por qué parquet:
  - Excel tiene límite de 1,048,576 filas
  - Con M2 desde 2018 todos los activos superan ese límite (~1.3M filas)
  - Parquet no tiene límite, ocupa 5x menos espacio y carga 5x más rápido

Output por activo:
    XAUUSD_Data.parquet         ← datos (sin límite de filas) — lo lee el backtest
    XAUUSD_Specs_QC.xlsx        ← specs + QC — para revisión humana
    especificaciones_*.json     ← specs del broker — lo lee el backtest
    package_summary_*.csv       ← resumen de auditoría

Uso:
    pip install MetaTrader5 pandas openpyxl pyarrow
    python mt5_extract_and_package_v3.py
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

# ══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════

ASSETS = [
    "XAUUSD",   # Oro          — en uso en el backtest actual
    "EURUSD",   # Euro/USD     — en uso
    "USDJPY",   # USD/Yen      — en uso
    "US30",     # Dow Jones    — en uso
    "USTEC",    # Nasdaq       — para futura optimización del portafolio
    "US500",    # S&P 500      — para futura optimización
    "DE40",    # DAX          — para futura optimización
    "XAGUSD",   # Plata        — para futura optimización
    "GBPUSD",   # GBP/USD      — para futura optimización
]

DATE_FROM = datetime(2018, 1, 1)
DATE_TO   = datetime.now()          # dinámico — siempre hasta hoy

MT5_TIMEFRAME    = mt5.TIMEFRAME_M2
TIMEFRAME_LABEL  = "M2"
EXPECTED_MINUTES = 2

_REPO_ROOT         = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR         = os.path.join(_REPO_ROOT, "data", "backtesting")
os.makedirs(OUTPUT_DIR, exist_ok=True)
AUTO_ENABLE_SYMBOL = True

# Columnas que el backtest necesita — en este orden exacto
DATA_COLUMNS = ["time", "open", "high", "low", "close",
                "tick_volume", "spread", "real_volume"]

# Comisión manual (fallback si no hay historial de deals)
# Raw Trading Ltd: $3.50/lado para pares forex y metales
COMMISSION_MANUAL = {
    "XAUUSD": 3.50,
    "EURUSD": 3.50,
    "USDJPY": 3.50,
    "US30":   0.00,
    "USTEC":  0.00,
    "US500":  0.00,
    "DE40":  0.00,
    "GBPUSD": 3.50,
    "XAGUSD": 3.50,
}

# ── Estilos Excel ─────────────────────────────────────────────────────
BD='1F2937'; W='FFFFFF'; BA='F3F4F6'; BK='111827'
GR='15803D'; GL='DCFCE7'; YL='92400E'; YLL='FEF3C7'
BLU='1D4ED8'; BLUL='DBEAFE'; GY='6B7280'
thin = Side(style='thin', color='D1D5DB')
BRD  = Border(left=thin, right=thin, top=thin, bottom=thin)


# ══════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════

def log(msg: str) -> None:
    print(msg, flush=True)


def cell_style(ws, r, c, v=None, bg=None, fg=BK, bold=False,
               sz=9, ha='center', wrap=False):
    cell = ws.cell(r, c)
    if v is not None:
        cell.value = v
    if bg:
        cell.fill = PatternFill('solid', fgColor=bg)
    cell.font      = Font(name='Arial', size=sz, bold=bold, color=fg)
    cell.alignment = Alignment(horizontal=ha, vertical='center',
                               wrap_text=wrap)
    cell.border = BRD
    return cell


# ══════════════════════════════════════════════════════════════════════
# CONEXIÓN MT5
# ══════════════════════════════════════════════════════════════════════

def connect_mt5() -> bool:
    if not mt5.initialize():
        log(f"❌ Error initialize(): {mt5.last_error()}")
        return False
    info = mt5.account_info()
    if info is None:
        log("❌ No se pudo leer account_info()")
        return False
    log(f"✓ Conectado | Cuenta: {info.login} | Broker: {info.company}")
    return True


def safe_symbol(symbol: str) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        log(f"  ⚠️  {symbol}: no existe en este broker")
        return False
    if info.visible:
        return True
    if AUTO_ENABLE_SYMBOL and mt5.symbol_select(symbol, True):
        log(f"  ✓ {symbol}: activado en Market Watch")
        return True
    log(f"  ⚠️  {symbol}: no visible y no se pudo activar")
    return False


# ══════════════════════════════════════════════════════════════════════
# COMISIONES
# ══════════════════════════════════════════════════════════════════════

def _comm_from_history(symbol: str) -> Optional[float]:
    """Método 1: comisión promedio de deals reales de los últimos 90 días."""
    deals = mt5.history_deals_get(datetime.now() - timedelta(days=90),
                                  datetime.now())
    if deals is None or len(deals) == 0:
        return None
    hits = [d for d in deals
            if getattr(d, 'symbol', None) == symbol
            and getattr(d, 'commission', 0) != 0
            and getattr(d, 'volume', 0) > 0]
    if not hits:
        return None
    avg = sum(abs(d.commission) / d.volume for d in hits) / len(hits)
    log(f"    → Historial: {len(hits)} deals | comm avg: ${avg:.4f}/lot/lado")
    return round(avg, 4)


def _comm_from_order_check(symbol: str, info) -> Optional[float]:
    """Método 2: simulación de orden ficticia."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    check = mt5.order_check({
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       info.volume_min,
        "type":         mt5.ORDER_TYPE_BUY,
        "price":        tick.ask,
        "deviation":    50,
        "magic":        0,
        "comment":      "spec_check",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    })
    if (check is not None
            and hasattr(check, 'commission')
            and check.commission not in (None, 0)):
        comm = round(abs(check.commission) / info.volume_min, 4)
        log(f"    → order_check: ${comm:.4f}/lot/lado")
        return comm
    return None


def get_commission(symbol: str, info) -> Tuple[float, float, str]:
    """Retorna (comm_por_lado, comm_round_trip, fuente)."""
    log(f"  {symbol} — buscando comisión...")
    comm = _comm_from_history(symbol)
    if comm is not None:
        source = "historial"
    else:
        comm = _comm_from_order_check(symbol, info)
        if comm is not None:
            source = "order_check"
        else:
            comm   = float(COMMISSION_MANUAL.get(symbol, 0.0))
            source = "⚠️ manual"
            log(f"    → manual: ${comm:.2f}/lado")
    rt = round(comm * 2, 4)
    log(f"    ✓ ${comm:.2f}/lado → ${rt:.2f} RT  [{source}]")
    return round(comm, 4), rt, source


# ══════════════════════════════════════════════════════════════════════
# ESPECIFICACIONES
# ══════════════════════════════════════════════════════════════════════

def get_specs(symbol: str) -> Optional[Dict]:
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    tick       = mt5.symbol_info_tick(symbol)
    spread_pts = round(tick.ask - tick.bid, info.digits) if tick else 0.0
    comm_s, comm_rt, comm_src = get_commission(symbol, info)
    return {
        "symbol":                symbol,
        "description":           info.description,
        "digits":                info.digits,
        "contract_size":         info.trade_contract_size,
        "volume_min":            info.volume_min,
        "volume_max":            info.volume_max,
        "volume_step":           info.volume_step,
        "point":                 info.point,
        "tick_size":             info.trade_tick_size,
        "tick_value":            info.trade_tick_value,
        "spread_current_pts":    spread_pts,
        "spread_avg_pts":        round(info.spread * info.point, info.digits),
        "swap_long":             info.swap_long,
        "swap_short":            info.swap_short,
        "swap_mode":             info.swap_mode,
        "commission_per_side":   comm_s,
        "commission_round_trip": comm_rt,
        "commission_source":     comm_src,
        "currency_base":         info.currency_base,
        "currency_profit":       info.currency_profit,
        "currency_margin":       info.currency_margin,
        "is_jpy":                info.currency_profit == "JPY",
        "stops_level":           info.trade_stops_level,
        "margin_initial":        info.margin_initial,
        "calc_mode":             info.trade_calc_mode,
        "execution_mode":        info.trade_exemode,
        "filling_mode":          info.filling_mode,
        "extracted_at":          datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════
# DESCARGA DE VELAS
# ══════════════════════════════════════════════════════════════════════

def download_rates(symbol: str) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, MT5_TIMEFRAME, DATE_FROM, DATE_TO)
    if rates is None or len(rates) == 0:
        log(f"  ⚠️  {symbol}: sin datos en el rango solicitado")
        return pd.DataFrame()
    df         = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    # Garantizar que existan todas las columnas necesarias
    for col in DATA_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[DATA_COLUMNS]   # solo columnas necesarias, en orden exacto


# ══════════════════════════════════════════════════════════════════════
# VALIDACIÓN Y QC
# ══════════════════════════════════════════════════════════════════════

def validate_and_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    qc = {
        "rows_initial":         int(len(df)),
        "duplicates_removed":   0,
        "nan_rows_removed":     0,
        "corrupt_rows_removed": 0,
        "gap_count":            0,
        "gap_pct":              0.0,
        "rows_final":           0,
        "start_time":           "",
        "end_time":             "",
    }
    if df.empty:
        return df, qc

    # 1. Duplicados
    dup = int(df.duplicated(subset=["time"]).sum())
    if dup:
        df = df.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    qc["duplicates_removed"] = dup

    # 2. NaN — incluye 'spread' (fix crítico vs v1)
    needed = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
    nan_rows = int(df[needed].isna().any(axis=1).sum())
    if nan_rows:
        df = df.dropna(subset=needed).reset_index(drop=True)
    qc["nan_rows_removed"] = nan_rows

    # 3. OHLC corrupto
    corrupt = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"]  > df[["open", "close", "high"]].min(axis=1)) |
        (df["low"]  > df["high"])
    )
    n_c = int(corrupt.sum())
    if n_c:
        df = df.loc[~corrupt].reset_index(drop=True)
    qc["corrupt_rows_removed"] = n_c

    # 4. Gaps
    df    = df.sort_values("time").reset_index(drop=True)
    delta = df["time"].diff().dt.total_seconds().div(60)
    gaps  = int((delta.dropna() != EXPECTED_MINUTES).sum())
    qc["gap_count"] = gaps
    qc["gap_pct"]   = round(gaps / max(len(df), 1) * 100, 4)
    qc["rows_final"]  = int(len(df))
    qc["start_time"]  = str(df["time"].iloc[0])  if len(df) else ""
    qc["end_time"]    = str(df["time"].iloc[-1]) if len(df) else ""
    return df, qc


# ══════════════════════════════════════════════════════════════════════
# GUARDADO — PARQUET para datos, Excel solo para Specs + QC
# ══════════════════════════════════════════════════════════════════════

def save_parquet(asset: str, df: pd.DataFrame) -> str:
    """
    Guarda los datos de precio en formato Parquet.
    Sin límite de filas, 5x más rápido que Excel, 3x menos espacio.
    El backtest lee este archivo directamente.
    """
    path = os.path.join(OUTPUT_DIR, f"{asset}_Data.parquet")
    df.to_parquet(path, index=False, engine="pyarrow")
    size_mb = os.path.getsize(path) / 1024 / 1024
    log(f"  ✅ {asset}_Data.parquet ({len(df):,} filas | {size_mb:.1f} MB)")
    return path


def write_specs_sheet(wb: Workbook, asset: str, specs: Dict) -> None:
    ws = wb.active
    ws.title = "Specs"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 28

    # Título
    ws.merge_cells("A1:B1")
    t = ws["A1"]
    t.value     = f"ESPECIFICACIONES MT5 — {asset} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    t.font      = Font(name='Arial', size=11, bold=True, color=W)
    t.fill      = PatternFill('solid', fgColor=BD)
    t.alignment = Alignment(horizontal='center', vertical='center')
    ws.row_dimensions[1].height = 22

    # Banner de fuente de comisión
    is_manual = str(specs.get("commission_source","")).startswith("⚠️")
    ws.merge_cells("A2:B2")
    s = ws["A2"]
    if is_manual:
        s.value = "⚠️ Comisión: configuración manual — verificar con el broker"
        s.font  = Font(name='Arial', size=9, bold=True, color=YL)
        s.fill  = PatternFill('solid', fgColor=YLL)
    else:
        s.value = "✅ Comisión obtenida de historial real de deals del broker"
        s.font  = Font(name='Arial', size=9, bold=True, color=GR)
        s.fill  = PatternFill('solid', fgColor=GL)
    s.alignment = Alignment(horizontal='center', vertical='center')
    s.border    = BRD
    ws.row_dimensions[2].height = 16

    # Headers
    cell_style(ws, 3, 1, "CAMPO", bg=BD, fg=W, bold=True, ha='left')
    cell_style(ws, 3, 2, "VALOR", bg=BD, fg=W, bold=True)
    ws.row_dimensions[3].height = 18

    sections = [
        ("── IDENTIFICACIÓN ─────────────────", None),
        ("symbol",                    specs.get("symbol")),
        ("description",               specs.get("description")),
        ("currency_base",             specs.get("currency_base")),
        ("currency_profit",           specs.get("currency_profit")),
        ("is_jpy (profit en JPY)",    specs.get("is_jpy")),
        ("── VOLUMEN ────────────────────────", None),
        ("contract_size",             specs.get("contract_size")),
        ("volume_min",                specs.get("volume_min")),
        ("volume_max",                specs.get("volume_max")),
        ("volume_step",               specs.get("volume_step")),
        ("── PRECIO ─────────────────────────", None),
        ("digits",                    specs.get("digits")),
        ("point",                     specs.get("point")),
        ("tick_size",                 specs.get("tick_size")),
        ("tick_value ($)",            specs.get("tick_value")),
        ("── SPREAD ─────────────────────────", None),
        ("spread_current_pts",        specs.get("spread_current_pts")),
        ("spread_avg_pts",            specs.get("spread_avg_pts")),
        ("── COMISIONES ─────────────────────", None),
        ("commission_per_side ($/lot)",   specs.get("commission_per_side")),
        ("★ commission_round_trip ($/lot)", specs.get("commission_round_trip")),
        ("commission_source",         specs.get("commission_source")),
        ("── SWAPS ──────────────────────────", None),
        ("swap_long",                 specs.get("swap_long")),
        ("swap_short",                specs.get("swap_short")),
        ("swap_mode",                 specs.get("swap_mode")),
        ("── EJECUCIÓN ──────────────────────", None),
        ("stops_level",               specs.get("stops_level")),
        ("margin_initial",            specs.get("margin_initial")),
        ("execution_mode",            specs.get("execution_mode")),
        ("filling_mode",              specs.get("filling_mode")),
        ("extracted_at",              specs.get("extracted_at")),
    ]

    r = 4
    for label, val in sections:
        is_sep  = val is None
        is_star = isinstance(label, str) and "★" in label
        row_bg  = BA if (not is_sep and r % 2 == 0) else None
        if is_star:
            row_bg = GL

        if is_sep:
            ws.merge_cells(f"A{r}:B{r}")
            c = ws.cell(r, 1, label)
            c.font      = Font(name='Arial', size=8, bold=True, color=GY)
            c.fill      = PatternFill('solid', fgColor=BA)
            c.alignment = Alignment(horizontal='left', vertical='center')
            ws.row_dimensions[r].height = 13
        else:
            disp = str(val) if isinstance(val, bool) else val
            fg2  = GR if is_star else BK
            cell_style(ws, r, 1, label, bg=row_bg, fg=fg2, bold=is_star, ha='left')
            cell_style(ws, r, 2, disp,  bg=row_bg, fg=fg2, bold=is_star)
            ws.row_dimensions[r].height = 15
        r += 1

    # Nota al pie
    ws.merge_cells(f"A{r}:B{r}")
    note = ws.cell(r, 1,
        "★ commission_round_trip = costo total abrir + cerrar 1 lote "
        "(el backtest descuenta este valor de cada trade)")
    note.font      = Font(name='Arial', size=8, italic=True, color=GR)
    note.fill      = PatternFill('solid', fgColor=GL)
    note.alignment = Alignment(horizontal='left', vertical='center')
    note.border    = BRD
    ws.row_dimensions[r].height = 14


def write_qc_sheet(wb: Workbook, asset: str, qc: Dict,
                   parquet_file: str) -> None:
    ws = wb.create_sheet("QC")
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 22

    ws.merge_cells("A1:B1")
    t = ws["A1"]
    t.value     = f"CONTROL DE CALIDAD — {asset} — {TIMEFRAME_LABEL}"
    t.font      = Font(name='Arial', size=11, bold=True, color=W)
    t.fill      = PatternFill('solid', fgColor=BD)
    t.alignment = Alignment(horizontal='center', vertical='center')
    ws.row_dimensions[1].height = 22

    cell_style(ws, 2, 1, "MÉTRICA", bg=BD, fg=W, bold=True, ha='left')
    cell_style(ws, 2, 2, "VALOR",   bg=BD, fg=W, bold=True)
    ws.row_dimensions[2].height = 18

    size_mb = os.path.getsize(parquet_file) / 1024 / 1024 \
              if os.path.exists(parquet_file) else 0.0

    rows = [
        ("Archivo de datos",          os.path.basename(parquet_file)),
        ("Tamaño del archivo",        f"{size_mb:.1f} MB"),
        ("Período",                   f"{qc.get('start_time','')[:10]} → {qc.get('end_time','')[:10]}"),
        ("Timeframe",                 TIMEFRAME_LABEL),
        ("Fecha de extracción",       datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("── CALIDAD ─────────────────────────────────────", None),
        ("Filas iniciales",           f"{qc.get('rows_initial',0):,}"),
        ("Filas finales",             f"{qc.get('rows_final',0):,}"),
        ("Duplicados eliminados",     qc.get("duplicates_removed", 0)),
        ("Filas NaN eliminadas",      qc.get("nan_rows_removed", 0)),
        ("Filas OHLC corruptas",      qc.get("corrupt_rows_removed", 0)),
        ("Gaps detectados",           qc.get("gap_count", 0)),
        ("Gaps %",                    f"{qc.get('gap_pct', 0.0):.4f}%"),
    ]

    r = 3
    for label, val in rows:
        is_sep = val is None
        if is_sep:
            ws.merge_cells(f"A{r}:B{r}")
            c = ws.cell(r, 1, label)
            c.font      = Font(name='Arial', size=8, bold=True, color=GY)
            c.fill      = PatternFill('solid', fgColor=BA)
            c.alignment = Alignment(horizontal='left', vertical='center')
            ws.row_dimensions[r].height = 13
        else:
            row_bg = BA if r % 2 == 0 else None
            cell_style(ws, r, 1, label, bg=row_bg, fg=BK, ha='left')
            cell_style(ws, r, 2, val,   bg=row_bg, fg=BK)
            ws.row_dimensions[r].height = 15
        r += 1

    # Resultado del QC
    r += 1
    all_clean = all(qc.get(k, 0) == 0 for k in
                    ["duplicates_removed","nan_rows_removed","corrupt_rows_removed"])
    high_gaps = qc.get("gap_pct", 0) > 0.5

    if all_clean and not high_gaps:
        ws.merge_cells(f"A{r}:B{r}")
        ok = ws.cell(r, 1, "✅ Datos limpios — sin duplicados, NaN ni OHLC corrupto")
        ok.font      = Font(name='Arial', size=9, bold=True, color=GR)
        ok.fill      = PatternFill('solid', fgColor=GL)
        ok.alignment = Alignment(horizontal='left', vertical='center')
        ok.border    = BRD
        ws.row_dimensions[r].height = 16
    if high_gaps:
        r2 = r + (1 if all_clean else 0)
        ws.merge_cells(f"A{r2}:B{r2}")
        warn = ws.cell(r2, 1,
            f"⚠️ Gaps altos: {qc.get('gap_count')} ({qc.get('gap_pct')}%) "
            f"— verifica la conexión o el rango de fechas")
        warn.font      = Font(name='Arial', size=9, bold=True, color=YL)
        warn.fill      = PatternFill('solid', fgColor=YLL)
        warn.alignment = Alignment(horizontal='left', vertical='center')
        warn.border    = BRD
        ws.row_dimensions[r2].height = 16


def save_specs_qc_excel(asset: str, specs: Dict,
                        qc: Dict, parquet_path: str) -> str:
    """Excel pequeño con Specs + QC. Sin datos de precio (van en parquet)."""
    path = os.path.join(OUTPUT_DIR, f"{asset}_Specs_QC.xlsx")
    wb   = Workbook()
    write_specs_sheet(wb, asset, specs)
    write_qc_sheet(wb, asset, qc, parquet_path)
    wb.save(path)
    return path


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    log("╔══════════════════════════════════════════════════════════╗")
    log("║  MT5 EXTRACT AND PACKAGE — v3  (datos en parquet)       ║")
    log("╚══════════════════════════════════════════════════════════╝\n")
    log(f"Período:  {DATE_FROM.strftime('%Y-%m-%d')} → {DATE_TO.strftime('%Y-%m-%d %H:%M')}")
    log(f"Activos:  {', '.join(ASSETS)}")
    log(f"Formato:  datos → .parquet | specs/QC → .xlsx\n")

    if not connect_mt5():
        return

    specs_snapshot: Dict[str, Dict] = {}
    summary_rows:   List[Dict]      = []

    for asset in ASSETS:
        log(f"\n{'─'*50}")
        log(f"→ {asset}")
        row = {"asset": asset, "status": "ok", "rows": 0,
               "parquet_mb": 0.0, "commission_rt": "",
               "comm_source": ""}

        if not safe_symbol(asset):
            row["status"] = "symbol_unavailable"
            summary_rows.append(row)
            continue

        # 1. Especificaciones
        specs = get_specs(asset)
        if specs is None:
            log(f"  ❌ No se pudieron extraer specs")
            row["status"] = "specs_failed"
            summary_rows.append(row)
            continue

        # 2. Datos históricos
        log(f"\n  Descargando velas {TIMEFRAME_LABEL} "
            f"{DATE_FROM.strftime('%Y-%m-%d')} → {DATE_TO.strftime('%Y-%m-%d')}...")
        df = download_rates(asset)
        if df.empty:
            row["status"] = "no_data"
            summary_rows.append(row)
            specs_snapshot[asset] = specs
            continue

        # 3. QC
        df, qc = validate_and_engineer(df)
        log(f"  ✓ QC: dup={qc['duplicates_removed']} | "
            f"nan={qc['nan_rows_removed']} | "
            f"corrupt={qc['corrupt_rows_removed']} | "
            f"gaps={qc['gap_count']} ({qc['gap_pct']}%)")

        # 4. Guardar parquet (datos)
        parquet_path = save_parquet(asset, df)
        size_mb      = os.path.getsize(parquet_path) / 1024 / 1024

        # 5. Guardar Excel (specs + QC)
        excel_path = save_specs_qc_excel(asset, specs, qc, parquet_path)
        log(f"  ✅ {os.path.basename(excel_path)}")

        specs_snapshot[asset] = specs
        row.update({
            "rows":          len(df),
            "parquet_mb":    round(size_mb, 1),
            "commission_rt": specs["commission_round_trip"],
            "comm_source":   specs["commission_source"],
        })
        summary_rows.append(row)

    # ── JSON de specs → lo detecta automáticamente el backtest ──────
    ts        = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = os.path.join(OUTPUT_DIR, f"especificaciones_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(specs_snapshot, f, indent=2,
                  ensure_ascii=False, default=str)

    # ── CSV de resumen ───────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, f"package_summary_{ts}.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    # ── Resumen final ────────────────────────────────────────────────
    log(f"\n{'═'*60}")
    log("✅  COMPLETADO")
    log(f"{'═'*60}")
    log(f"  JSON specs → {os.path.basename(json_path)}")
    log(f"  Resumen    → {os.path.basename(csv_path)}")
    log(f"\n  {'Activo':<10} {'Filas':>10}  {'MB':>6}  {'Comm RT':>9}  Fuente")
    log(f"  {'─'*55}")
    for r in summary_rows:
        filas = f"{r['rows']:,}" if r['rows'] else "—"
        mb    = f"{r['parquet_mb']:.1f}" if r['parquet_mb'] else "—"
        comm  = f"${r['commission_rt']:.2f}" if r['commission_rt'] != "" else "—"
        log(f"  {r['asset']:<10} {filas:>10}  {mb:>6}  {comm:>9}  "
            f"{r['comm_source']}")

    log(f"\n  PRÓXIMOS PASOS:")
    log(f"  1. Los .parquet contienen todos los datos de precio")
    log(f"  2. El backtest los lee automáticamente en vez de los .xlsx")
    log(f"  3. El JSON de specs también es detectado automáticamente")
    log(f"  → python backtest_final_completo.py")
    mt5.shutdown()


if __name__ == "__main__":
    main()
