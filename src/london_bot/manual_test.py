
from __future__ import annotations

from typing import Any, Dict, Optional

from src.shared.mt5_connector import get_tick, get_spread_points
from src.london_bot.order_manager_london import open_position


def force_demo_trade(
    symbol: str,
    direction: int,
    asset_params: Dict[str, Dict[str, Any]],
    bot_magic: int,
    deviation: int,
    logger,
) -> Optional[Dict[str, Any]]:
    """
    Fuerza una operación market de prueba usando el minimum lot del activo.
    direction:
        1  -> BUY
       -1  -> SELL
    """
    if symbol not in asset_params:
        logger.error(f"Símbolo de prueba no soportado: {symbol}")
        return None

    p = asset_params[symbol]
    tick = get_tick(symbol)

    if tick is None:
        logger.error(f"{symbol}: no se pudo obtener tick para prueba")
        return None

    entry_price = tick.ask if direction == 1 else tick.bid
    spread_signal = float(p["sp"])
    spread_entry = get_spread_points(symbol)

    min_buffer = (10 ** (-p["digits"])) * 50
    sl_buffer = max(entry_price * p["sl_pct"], spread_entry * 3, min_buffer)

    if direction == 1:
        stop_loss = entry_price - sl_buffer
    else:
        stop_loss = entry_price + sl_buffer

    lots = p["ml"]

    pos = open_position(
        symbol=symbol,
        direction=direction,
        lots=lots,
        sl=stop_loss,
        asset_params=p,
        signal_price=entry_price,
        spread_signal=spread_signal,
        signal_type="FORCE_TEST_TRADE",
        bot_magic=bot_magic,
        deviation=deviation,
        logger=logger,
    )

    if pos:
        pos["atr"] = 1.0
        logger.info(f"🧪 PRUEBA EXITOSA {symbol} | pos_ticket={pos['ticket']}")
    else:
        logger.error(f"🧪 PRUEBA FALLÓ {symbol}")

    return pos