from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import MetaTrader5 as mt5


def initialize_mt5(terminal_path: Optional[str] = None) -> bool:
    """
    Inicializa la conexión con MT5.
    Si terminal_path es None, usa la instalación por defecto.
    """
    if terminal_path:
        ok = mt5.initialize(path=terminal_path)
    else:
        ok = mt5.initialize()

    return bool(ok)


def shutdown_mt5() -> None:
    """Cierra la conexión con MT5."""
    mt5.shutdown()


def get_last_error() -> Any:
    """Devuelve el último error reportado por MT5."""
    return mt5.last_error()


def get_account_info() -> Optional[Any]:
    """Devuelve account_info() de MT5."""
    return mt5.account_info()


def get_terminal_info() -> Optional[Any]:
    """Devuelve terminal_info() de MT5."""
    return mt5.terminal_info()


def connect_mt5(logger, terminal_path: Optional[str] = None) -> bool:
    """
    Inicializa MT5 y valida que account_info() esté disponible.
    """
    if not initialize_mt5(terminal_path=terminal_path):
        logger.error(f"MT5 initialize() falló: {get_last_error()}")
        return False

    info = get_account_info()
    if info is None:
        logger.error("No se pudo leer account_info()")
        return False

    logger.info(
        f"Conectado | Cuenta: {info.login} | Balance: ${info.balance:.2f} | Demo: {info.trade_mode == 0}"
    )
    return True


def get_mt5_now(offset_hours: int = 0) -> datetime:
    """
    Devuelve hora local ajustada al offset configurado para alinear con MT5.
    """
    return datetime.now() + timedelta(hours=offset_hours)


def get_symbol_info(symbol: str) -> Optional[Any]:
    """Devuelve symbol_info(symbol)."""
    return mt5.symbol_info(symbol)


def ensure_symbol_selected(symbol: str) -> bool:
    """
    Asegura que el símbolo esté visible/seleccionado en MT5.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return False
    if not info.visible:
        return bool(mt5.symbol_select(symbol, True))
    return True


def get_tick(symbol: str) -> Optional[Any]:
    """Devuelve symbol_info_tick(symbol)."""
    return mt5.symbol_info_tick(symbol)


def get_spread_points(symbol: str) -> float:
    """
    Devuelve el spread actual en unidades de precio (ask - bid),
    redondeado según los dígitos del símbolo.
    """
    tick = get_tick(symbol)
    info = get_symbol_info(symbol)

    if tick is None or info is None:
        return 0.0

    return round(tick.ask - tick.bid, int(info.digits))


def get_live_spread_fallback(symbol: str, digits: int, fallback_spread: float) -> float:
    """
    Si hay tick disponible, usa spread en vivo.
    Si no, usa spread fallback.
    """
    tick = get_tick(symbol)
    if tick is None:
        return float(fallback_spread)
    return round(tick.ask - tick.bid, int(digits))


def get_point_value(symbol: str) -> Optional[float]:
    """
    Devuelve el point del símbolo.
    """
    info = get_symbol_info(symbol)
    if info is None:
        return None
    return float(info.point) if info.point else None


def get_positions(symbol: Optional[str] = None, ticket: Optional[int] = None) -> Optional[tuple]:
    """
    Wrapper para positions_get().
    """
    if ticket is not None:
        return mt5.positions_get(ticket=ticket)
    if symbol is not None:
        return mt5.positions_get(symbol=symbol)
    return mt5.positions_get()


def get_orders(symbol: Optional[str] = None) -> Optional[tuple]:
    """
    Wrapper para orders_get().
    """
    if symbol is not None:
        return mt5.orders_get(symbol=symbol)
    return mt5.orders_get()


def copy_rates_from_pos(symbol: str, timeframe: int, start_pos: int, count: int):
    """
    Wrapper para copy_rates_from_pos().
    """
    return mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)


def copy_rates_range(symbol: str, timeframe: int, date_from: datetime, date_to: datetime):
    """
    Wrapper para copy_rates_range().
    """
    return mt5.copy_rates_range(symbol, timeframe, date_from, date_to)


def order_check(request: Dict[str, Any]):
    """
    Wrapper para order_check().
    """
    return mt5.order_check(request)


def order_send(request: Dict[str, Any]):
    """
    Wrapper para order_send().
    """
    return mt5.order_send(request)


def history_deals_get(date_from: datetime, date_to: datetime):
    """
    Wrapper para history_deals_get().
    """
    return mt5.history_deals_get(date_from, date_to)


def is_mt5_connected() -> bool:
    """
    Verifica si el terminal MT5 sigue respondiendo.
    Devuelve False si el terminal fue cerrado o la API está rota.
    """
    return mt5.terminal_info() is not None