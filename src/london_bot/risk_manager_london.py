from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from src.core.execution import ExecutionDecision


def is_trading_day_allowed(mt5_now: datetime, asset_params: Dict[str, Any]) -> bool:
    """
    Verifica si el weekday actual está permitido para el activo.
    """
    return mt5_now.weekday() in asset_params["dow"]


def is_signal_hour_allowed(signal_hour: int, asset_params: Dict[str, Any]) -> bool:
    """
    Verifica si la hora de señal actual está habilitada para el activo.
    """
    return signal_hour in asset_params["hours"]


def has_reached_max_trades(
    trades_today: int,
    max_trades_per_day_per_asset: int,
) -> bool:
    """
    Devuelve True si ya se alcanzó el límite diario de trades por activo.
    """
    return trades_today >= max_trades_per_day_per_asset


def has_open_position(symbol: str, open_positions: Dict[str, Dict[str, Any]]) -> bool:
    """
    Devuelve True si el activo ya tiene una posición abierta en memoria.
    """
    return symbol in open_positions


def is_execution_enabled(config: Dict[str, Any]) -> bool:
    """
    Devuelve True si el entorno permite ejecución real.
    """
    return bool(config["mode"]["execution_enabled"])


def can_evaluate_signal(
    symbol: str,
    signal_hour: int,
    mt5_now: datetime,
    asset_params: Dict[str, Any],
    trades_today: int,
    max_trades_per_day_per_asset: int,
    open_positions: Dict[str, Dict[str, Any]],
) -> ExecutionDecision:
    """
    Verifica si tiene sentido evaluar señal para este símbolo en este momento.
    """
    if not is_trading_day_allowed(mt5_now, asset_params):
        return ExecutionDecision.deny("weekday no permitido")

    if not is_signal_hour_allowed(signal_hour, asset_params):
        return ExecutionDecision.deny("hora no permitida para el activo")

    if has_reached_max_trades(trades_today, max_trades_per_day_per_asset):
        return ExecutionDecision.deny("máximo de trades diarios alcanzado")

    if has_open_position(symbol, open_positions):
        return ExecutionDecision.deny("ya existe posición abierta en este símbolo")

    return ExecutionDecision.permit()


def can_execute_trade(
    config: Dict[str, Any],
    symbol: str,
    logger,
) -> ExecutionDecision:
    """
    Verifica si el entorno permite ejecutar una orden real.
    """
    if not is_execution_enabled(config):
        logger.info(f"[SHADOW] execution_enabled=False | {symbol}")
        return ExecutionDecision.deny("execution disabled")

    return ExecutionDecision.permit()