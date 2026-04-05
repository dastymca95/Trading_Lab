from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_audit_dataframe(audit_file: str, logger) -> pd.DataFrame:
    """
    Carga el CSV de auditoría. Si no existe o está vacío, devuelve DataFrame vacío.
    """
    audit_path = Path(audit_file)

    if not audit_path.exists():
        logger.info(f"No existe audit_file todavía: {audit_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(audit_path)
        if df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"Error leyendo audit_file {audit_file}: {e}")
        return pd.DataFrame()


def build_pnl_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Construye un resumen global de performance a partir del audit CSV.
    """
    if df.empty:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl_real": 0.0,
            "total_pnl_backtest_approx": 0.0,
            "avg_pnl_real": 0.0,
            "avg_execution_ms": 0.0,
            "avg_slippage_pts": 0.0,
        }

    total_trades = len(df)
    wins = int((df["pnl_real"] > 0).sum())
    losses = int((df["pnl_real"] <= 0).sum())
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    total_pnl_real = float(df["pnl_real"].sum())
    total_pnl_backtest_approx = float(df["pnl_backtest_approx"].sum())
    avg_pnl_real = float(df["pnl_real"].mean())
    avg_execution_ms = float(df["execution_ms"].mean())
    avg_slippage_pts = float(df["slippage_pts"].mean())

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "total_pnl_real": round(total_pnl_real, 2),
        "total_pnl_backtest_approx": round(total_pnl_backtest_approx, 2),
        "avg_pnl_real": round(avg_pnl_real, 2),
        "avg_execution_ms": round(avg_execution_ms, 2),
        "avg_slippage_pts": round(avg_slippage_pts, 6),
    }


def build_symbol_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye resumen por activo.
    """
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby("asset", dropna=False)
        .agg(
            total_trades=("asset", "count"),
            wins=("pnl_real", lambda s: int((s > 0).sum())),
            losses=("pnl_real", lambda s: int((s <= 0).sum())),
            total_pnl_real=("pnl_real", "sum"),
            total_pnl_backtest_approx=("pnl_backtest_approx", "sum"),
            avg_pnl_real=("pnl_real", "mean"),
            avg_execution_ms=("execution_ms", "mean"),
            avg_slippage_pts=("slippage_pts", "mean"),
        )
        .reset_index()
    )

    grouped["win_rate"] = grouped["wins"] / grouped["total_trades"]

    numeric_cols = [
        "total_pnl_real",
        "total_pnl_backtest_approx",
        "avg_pnl_real",
        "avg_execution_ms",
        "avg_slippage_pts",
        "win_rate",
    ]
    grouped[numeric_cols] = grouped[numeric_cols].round(4)

    return grouped


def build_exit_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume cuántos trades cerraron por cada exit_reason.
    """
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby("exit_reason", dropna=False)
        .agg(
            total_trades=("exit_reason", "count"),
            total_pnl_real=("pnl_real", "sum"),
            avg_pnl_real=("pnl_real", "mean"),
        )
        .reset_index()
    )

    grouped[["total_pnl_real", "avg_pnl_real"]] = grouped[
        ["total_pnl_real", "avg_pnl_real"]
    ].round(4)

    return grouped


def log_pnl_summary(audit_file: str, logger) -> None:
    """
    Carga el audit CSV y escribe en logs un resumen general de performance.
    """
    df = load_audit_dataframe(audit_file=audit_file, logger=logger)
    summary = build_pnl_summary(df)

    logger.info(
        "PNL SUMMARY | "
        f"trades={summary['total_trades']} | "
        f"wins={summary['wins']} | "
        f"losses={summary['losses']} | "
        f"win_rate={summary['win_rate']:.2%} | "
        f"pnl_real=${summary['total_pnl_real']:+.2f} | "
        f"pnl_bt≈${summary['total_pnl_backtest_approx']:+.2f} | "
        f"avg_exec_ms={summary['avg_execution_ms']:.2f} | "
        f"avg_slip={summary['avg_slippage_pts']:.6f}"
    )