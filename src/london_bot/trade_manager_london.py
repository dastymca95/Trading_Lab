from __future__ import annotations

from src.core.position import PositionState
from src.shared.mt5_connector import (
    get_positions,
    get_symbol_info,
    get_tick,
    order_check,
    order_send,
)
import MetaTrader5 as mt5


def sl_is_valid_for_broker(
    symbol: str,
    direction: int,
    new_sl: float,
    digits: int,
) -> tuple[bool, str]:
    """
    Verifica si el SL propuesto cumple con las distancias mínimas del broker:
    - stops level
    - freeze level
    - point mínimo
    """
    info = get_symbol_info(symbol)
    tick = get_tick(symbol)

    if info is None or tick is None:
        return False, "symbol_info/tick unavailable"

    point = float(info.point) if info.point else (10 ** (-digits) if digits > 0 else 1.0)

    stops_level_px = float(getattr(info, "trade_stops_level", 0) or 0) * point
    freeze_level_px = float(getattr(info, "trade_freeze_level", 0) or 0) * point
    min_distance = max(stops_level_px, freeze_level_px, point)

    current_ref = tick.bid if direction == 1 else tick.ask
    dist = abs(current_ref - new_sl)

    if dist < min_distance:
        return (
            False,
            f"SL demasiado cerca del precio actual | dist={dist:.8f} < min_distance={min_distance:.8f}",
        )

    return True, "ok"


def modify_sl(
    ticket: int,
    symbol: str,
    direction: int,
    new_sl: float,
    digits: int,
    logger,
) -> bool:
    """
    Envía modificación de stop loss al broker.
    """
    valid, reason = sl_is_valid_for_broker(
        symbol=symbol,
        direction=direction,
        new_sl=new_sl,
        digits=digits,
    )

    if not valid:
        logger.info(f"{symbol} ticket={ticket}: modify_sl omitido | {reason}")
        return False

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": symbol,
        "sl": round(new_sl, digits),
    }

    check = order_check(request)
    if check is not None:
        logger.info(
            f"{symbol} ticket={ticket}: order_check modify_sl | "
            f"retcode={getattr(check, 'retcode', None)} | "
            f"comment={getattr(check, 'comment', '')}"
        )

    result = order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        logger.warning(
            f"{symbol} ticket={ticket}: modify_sl falló retcode={code} | "
            f"last_error={mt5.last_error()}"
        )
        return False

    return True


def update_trailing(position: PositionState, logger) -> str:
    """
    Gestiona trailing stop y break-even de una posición abierta.

    Returns:
        - "ok"       -> posición sigue activa
        - "stopped"  -> posición ya no existe en MT5 (cerrada)
        - "error"    -> no se pudo procesar tick
    """
    symbol = position.symbol
    direction = position.direction
    ticket = position.ticket
    atr = position.atr
    digits = position.digits

    tick = get_tick(symbol)
    if tick is None:
        return "error"

    current_price = tick.bid if direction == 1 else tick.ask

    positions = get_positions(ticket=ticket)
    if not positions:
        return "stopped"

    current_sl = float(positions[0].sl)

    if direction == 1:
        position.best_price = max(position.best_price, current_price)
    else:
        position.best_price = min(position.best_price, current_price)

    be_level = position.be_level

    position.breakeven_hit = position.breakeven_hit or (
        position.best_price >= be_level if direction == 1
        else position.best_price <= be_level
    )

    if not position.breakeven_hit:
        return "ok"

    if direction == 1:
        new_sl = max(
            position.best_price - atr * position.trail_mult,
            be_level,
            current_sl,
        )
    else:
        new_sl = min(
            position.best_price + atr * position.trail_mult,
            be_level,
            current_sl,
        )

    new_sl = round(new_sl, digits)

    if abs(new_sl - current_sl) > 10 ** (-digits):
        modified = modify_sl(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            new_sl=new_sl,
            digits=digits,
            logger=logger,
        )
        if modified:
            position.sl = new_sl

    return "ok"