from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from src.london_bot.london_levels import get_signal_context
from src.shared.risk_utils import calc_lots


def check_signal(
    symbol: str,
    asset_params: Dict[str, Any],
    trades_today: int,
    signal_hour: int,
    mt5_now: datetime,
    volume_means: Dict[str, float],
    initial_capital_per_asset: float,
    max_trades_per_day_per_asset: int,
) -> Optional[Dict[str, Any]]:
    """
    Evalúa si existe señal London Range Breakout para un símbolo dado.

    Devuelve un dict con la señal lista para ejecución o None si no hay setup válido.
    """
    # Filtro por día permitido
    if mt5_now.weekday() not in asset_params["dow"]:
        return None

    # Filtro por máximo de trades por día
    if trades_today >= max_trades_per_day_per_asset:
        return None

    # Contexto de señal
    ctx = get_signal_context(
        symbol=symbol,
        signal_date=mt5_now.date(),
        signal_hour=signal_hour,
        asset_params=asset_params,
    )
    if not ctx:
        return None

    signal_row = ctx["signal_row"]

    entry_price = float(signal_row["close"])
    atr_value = float(signal_row["atr14"]) if np.isfinite(signal_row["atr14"]) else 0.0
    candle_range = float(signal_row["high"] - signal_row["low"])
    tick_volume = float(signal_row["tick_volume"])

    london_high = float(ctx["lh"])
    london_low = float(ctx["ll"])
    lrr = float(ctx["lrr"])
    spread_signal = float(ctx["spread_signal"])

    # Si ATR es inválido, descartar
    if atr_value <= 0:
        return None

    # Filtro por volumen relativo vs media histórica
    volume_mean = volume_means.get(symbol, 0.0)
    if volume_mean > 0 and tick_volume < volume_mean:
        return None

    # Filtro por London range ratio mínimo
    if not np.isfinite(lrr) or lrr <= asset_params["lrr_min"]:
        return None

    direction = None
    signal_type = ""

    # Breakout London High / Low
    if entry_price > london_high:
        direction = 1
        signal_type = f"Breakout ALCISTA (London High={london_high:.{asset_params['digits']}f})"

    elif entry_price < london_low:
        direction = -1
        signal_type = f"Breakout BAJISTA (London Low={london_low:.{asset_params['digits']}f})"

    # Vela grande tipo expansión
    elif candle_range > asset_params["atr_mult"] * atr_value:
        direction = -1 if signal_row["close"] > signal_row["open"] else 1
        signal_type = (
            f"Vela grande ({candle_range:.{asset_params['digits']}f} > "
            f"{asset_params['atr_mult']}xATR)"
        )

    if direction is None:
        return None

    # Stop inicial
    if direction == 1:
        stop_loss = entry_price * (1 - asset_params["sl_pct"])
        stop_loss = max(stop_loss, london_low)
    else:
        stop_loss = entry_price * (1 + asset_params["sl_pct"])
        stop_loss = min(stop_loss, london_high)

    # Distancia efectiva del stop incluyendo spread
    stop_distance = max(
        abs(entry_price - stop_loss) + spread_signal,
        entry_price * 0.0015,
    )

    if stop_distance < 1e-8:
        return None

    lots = calc_lots(
        capital=initial_capital_per_asset,
        risk_pct=asset_params["risk_pct"],
        sl_dist=stop_distance,
        entry=entry_price,
        p=asset_params,
    )

    if lots <= 0:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "stype": signal_type,
        "ep": entry_price,
        "sl": stop_loss,
        "sl_dist": stop_distance,
        "lots": lots,
        "atr": atr_value,
        "lh": london_high,
        "ll": london_low,
        "lrr": lrr,
        "spread_signal": spread_signal,
        "signal_time": signal_row["time"],
        "signal_tick_volume": tick_volume,
    }