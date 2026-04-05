from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from src.shared.mt5_connector import copy_rates_from_pos


def get_candles(
    symbol: str,
    timeframe: int = mt5.TIMEFRAME_M2,
    n: int = 1800,
) -> Optional[pd.DataFrame]:
    """
    Descarga velas desde MT5 y devuelve un DataFrame ordenado por tiempo.
    """
    rates = copy_rates_from_pos(symbol, timeframe, 0, n)

    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def engineer_rates_df(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    """
    Enriquece el DataFrame de velas con:
    - date
    - dow
    - previous close
    - true range
    - atr14
    - spread_px
    """
    out = df.copy().sort_values("time").reset_index(drop=True)

    out["date"] = out["time"].dt.date
    out["dow"] = out["time"].dt.dayofweek
    out["pc"] = out["close"].shift(1)

    out["tr"] = np.maximum(
        out["high"] - out["low"],
        np.maximum(
            abs(out["high"] - out["pc"]),
            abs(out["low"] - out["pc"]),
        ),
    )

    out["atr14"] = out["tr"].rolling(14).mean()

    point = 10 ** (-digits) if digits > 0 else 1.0

    if "spread" in out.columns and out["spread"].notna().any():
        out["spread_px"] = (
            pd.to_numeric(out["spread"], errors="coerce")
            .ffill()
            .bfill()
            .fillna(0)
            * point
        )
    else:
        out["spread_px"] = np.nan

    return out


def get_signal_context(
    symbol: str,
    signal_date: date,
    signal_hour: int,
    asset_params: Dict[str, Any],
    candles_n: int = 1800,
) -> Optional[Dict[str, Any]]:
    """
    Construye el contexto de señal para el bot London:
    - vela de señal exacta a signal_hour:00
    - London high / low
    - ATR medio de la sesión London
    - London range ratio (lrr)
    - spread de señal
    """
    df_raw = get_candles(symbol=symbol, timeframe=mt5.TIMEFRAME_M2, n=candles_n)
    if df_raw is None or len(df_raw) < 100:
        return None

    df = engineer_rates_df(df_raw, digits=int(asset_params["digits"]))

    today = df[df["date"] == signal_date].copy()
    if len(today) < 20:
        return None

    signal_rows = today[
        (today["time"].dt.hour == signal_hour) &
        (today["time"].dt.minute == 0)
    ].copy()

    if len(signal_rows) == 0:
        return None

    signal_row = signal_rows.iloc[-1]

    london = today[
        (today["time"].dt.hour >= 9) &
        (today["time"].dt.hour < 15)
    ].copy()

    if len(london) == 0:
        return None

    london_high = float(london["high"].max())
    london_low = float(london["low"].min())

    if london["atr14"].notna().any():
        london_atr_mean = float(london["atr14"].mean())
    else:
        london_atr_mean = 0.0

    london_range_ratio = (
        (london_high - london_low) / london_atr_mean
        if london_atr_mean > 0
        else 0.0
    )

    spread_signal = signal_row.get("spread_px", np.nan)
    if pd.isna(spread_signal) or float(spread_signal) <= 0:
        spread_signal = float(asset_params["sp"])

    return {
        "signal_row": signal_row,
        "lh": london_high,
        "ll": london_low,
        "lam": london_atr_mean,
        "lrr": london_range_ratio,
        "spread_signal": float(spread_signal),
        "today_df": today,
        "london_df": london,
    }