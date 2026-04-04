from __future__ import annotations

from datetime import datetime, date
from typing import Iterable


def is_new_mt5_day(last_date: date | None, mt5_now: datetime) -> bool:
    """
    Devuelve True si cambió el día MT5 respecto al último día registrado.
    """
    return last_date != mt5_now.date()


def reset_daily_trades(asset_params: dict) -> dict:
    """
    Reinicia el contador diario de trades por símbolo.
    """
    return {symbol: 0 for symbol in asset_params}


def get_current_time_parts(mt5_now: datetime) -> tuple[int, int]:
    """
    Devuelve (hour, minute) de la hora MT5.
    """
    return mt5_now.hour, mt5_now.minute


def is_signal_scan_window(
    mt5_now: datetime,
    signal_hours: Iterable[int],
    scan_minutes: Iterable[int],
) -> bool:
    """
    Devuelve True si el momento actual cae dentro de la ventana
    de escaneo de señales.
    """
    return mt5_now.hour in signal_hours and mt5_now.minute in scan_minutes


def build_signal_key(symbol: str, mt5_now: datetime, signal_hour: int) -> str:
    """
    Construye una clave única para evitar revisar dos veces
    la misma vela de señal por símbolo/día/hora.
    """
    return f"{symbol}_{mt5_now.date()}_{signal_hour}"


def should_run_for_asset(current_hour: int, asset_params: dict) -> bool:
    """
    Devuelve True si el activo está habilitado para operar
    en la hora actual según su configuración.
    """
    return current_hour in asset_params["hours"]


def should_emit_heartbeat(now: datetime, current_minute: int) -> bool:
    """
    Devuelve True si corresponde emitir el heartbeat horario.
    """
    return current_minute == 0 and now.second < 30