from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import MetaTrader5 as mt5

from src.core.position import PositionState
from src.shared.mt5_connector import (
    get_positions,
    get_spread_points,
    get_symbol_info,
    get_tick,
    order_check,
    order_send,
)


def get_filling_candidates(symbol: str) -> List[int]:
    """
    Devuelve una lista ordenada de filling modes compatibles o probables
    para probar al enviar la orden.
    """
    info = get_symbol_info(symbol)

    if info is None:
        return [
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
            mt5.ORDER_FILLING_RETURN,
        ]

    reported = info.filling_mode
    preferred: List[int] = []
    modes = [
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_RETURN,
    ]

    for mode in modes:
        try:
            if reported == mode or (reported & mode) == mode:
                preferred.append(mode)
        except TypeError:
            if reported == mode:
                preferred.append(mode)

    for mode in modes:
        if mode not in preferred:
            preferred.append(mode)

    return preferred


def open_position(
    symbol: str,
    direction: int,
    lots: float,
    sl: float,
    asset_params: Dict[str, Any],
    signal_price: float,
    spread_signal: float,
    signal_type: str,
    bot_magic: int,
    deviation: int,
    logger,
) -> Optional[PositionState]:
    """
    Envía una orden market BUY/SELL y devuelve el objeto posición si se ejecuta.
    """
    tick = get_tick(symbol)
    if tick is None:
        logger.error(f"{symbol}: no se pudo obtener tick")
        return None

    info = get_symbol_info(symbol)
    if info is None:
        logger.error(f"{symbol}: symbol_info() devolvió None")
        return None

    price = tick.ask if direction == 1 else tick.bid
    spread_entry = get_spread_points(symbol)
    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL

    filling_candidates = get_filling_candidates(symbol)
    last_result = None

    for filling_mode in filling_candidates:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "sl": round(sl, int(asset_params["digits"])),
            "deviation": deviation,
            "magic": bot_magic,
            "comment": "london_bot_main",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        check = order_check(request)
        if check is None:
            logger.warning(
                f"{symbol}: order_check devolvió None | last_error={mt5.last_error()} | request={request}"
            )
        else:
            logger.info(
                f"{symbol}: order_check filling_mode={filling_mode} | "
                f"retcode={getattr(check, 'retcode', None)} | "
                f"comment={getattr(check, 'comment', '')}"
            )

        t_before = time.time()
        logger.info(f"{symbol}: intentando orden con filling_mode={filling_mode}")

        result = order_send(request)
        exec_ms = int((time.time() - t_before) * 1000)
        last_result = result

        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            real_price = float(result.price)
            slippage = round(
                (real_price - signal_price) * direction,
                int(asset_params["digits"]),
            )

            time.sleep(0.3)

            real_ticket = int(result.order)
            open_pos_mt5 = get_positions(symbol=symbol)

            if open_pos_mt5:
                for p_mt5 in open_pos_mt5:
                    same_volume = abs(p_mt5.volume - lots) < 0.001
                    same_type = p_mt5.type == (0 if direction == 1 else 1)
                    same_magic = p_mt5.magic == bot_magic
                    if same_magic and same_volume and same_type:
                        real_ticket = int(p_mt5.ticket)
                        break

            be_level = (
                real_price + spread_entry
                if direction == 1
                else real_price - spread_entry
            )

            logger.info(
                f"✅ {symbol} {'BUY' if direction == 1 else 'SELL'} {lots}L @ {real_price} | "
                f"signal={signal_price} | "
                f"slip={slippage:+.{asset_params['digits']}f} | "
                f"spread_entry={spread_entry} | "
                f"spread_signal={spread_signal} | "
                f"{exec_ms}ms | "
                f"order_ticket={result.order} | "
                f"pos_ticket={real_ticket} | "
                f"{signal_type}"
            )

            return PositionState(
                ticket=real_ticket,
                symbol=symbol,
                direction=int(direction),
                lots=float(lots),
                real_entry_price=float(real_price),
                signal_price=float(signal_price),
                sl=float(sl),
                be_level=float(be_level),
                best_price=float(real_price),
                open_time=datetime.now(),
                digits=int(asset_params["digits"]),
                cs=int(asset_params["cs"]),
                jpy=bool(asset_params["jpy"]),
                comm=float(asset_params["comm"]),
                trail_mult=float(asset_params["trail_mult"]),
                stype=signal_type,
                backtest_price=float(signal_price),
                slippage=float(slippage),
                spread=float(spread_entry),
                spread_signal=float(spread_signal),
                exec_ms=int(exec_ms),
            )

        if result is None:
            logger.warning(
                f"{symbol}: intento con filling_mode={filling_mode} devolvió None | "
                f"last_error={mt5.last_error()} | request={request}"
            )
        else:
            logger.warning(
                f"{symbol}: intento con filling_mode={filling_mode} rechazado | "
                f"retcode={result.retcode} | "
                f"comment={getattr(result, 'comment', '')}"
            )

    if last_result is None:
        logger.error(
            f"{symbol}: orden rechazada definitivamente | result=None | last_error={mt5.last_error()}"
        )
    else:
        logger.error(
            f"{symbol}: orden rechazada definitivamente | "
            f"retcode={last_result.retcode} | "
            f"comment={getattr(last_result, 'comment', '')}"
        )

    return None