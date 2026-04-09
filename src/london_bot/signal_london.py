from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, FrozenSet, Optional, Tuple

import numpy as np

from src.london_bot.london_levels import get_signal_context
from src.shared.risk_utils import calc_lots
from src.core.signal import Signal


def check_signal(
    symbol: str,
    asset_params: Dict[str, Any],
    signal_hour: int,
    mt5_now: datetime,
    volume_means: Dict[str, float],
    initial_capital_per_asset: float,
    daily_filter: Optional[FrozenSet] = None,
) -> Tuple[Optional[Signal], str]:
    """
    Evalúa si existe señal London Range Breakout para un símbolo dado.

    Returns (signal, reason_code) where reason_code documents why evaluation
    stopped or succeeded. Used by the audit trail in main_london.py.

    Reason codes:
      daily_filter_fail   – regime filter blocked today
      context_unavailable – no candle context from MT5
      atr_zero            – ATR is zero or invalid
      volume_filter_fail  – tick_volume below historical mean
      lrr_fail            – LRR below threshold
      no_direction        – no breakout / large-candle detected
      stop_distance_zero  – SL distance collapsed to zero
      lots_zero           – position sizing returned 0
      ok                  – signal valid, direction natural
      ok_force_direction  – signal valid, direction overridden by force_direction

    daily_filter: frozenset of allowed date objects (from init_daily_filters).
                  None = no filter, all days pass. Checked before any signal logic.

    Asume que los filtros operativos externos ya fueron validados
    en risk_manager_london.py.
    """
    # Daily regime filter gate — must precede all signal logic (causal, no look-ahead)
    if daily_filter is not None and mt5_now.date() not in daily_filter:
        return None, "daily_filter_fail"

    ctx = get_signal_context(
        symbol=symbol,
        signal_date=mt5_now.date(),
        signal_hour=signal_hour,
        asset_params=asset_params,
    )
    if not ctx:
        return None, "context_unavailable"

    signal_row = ctx["signal_row"]

    entry_price = float(signal_row["close"])
    atr_value = float(signal_row["atr14"]) if np.isfinite(signal_row["atr14"]) else 0.0
    candle_range = float(signal_row["high"] - signal_row["low"])
    tick_volume = float(signal_row["tick_volume"])

    london_high = float(ctx["lh"])
    london_low = float(ctx["ll"])
    lrr = float(ctx["lrr"])
    spread_signal = float(ctx["spread_signal"])

    if atr_value <= 0:
        return None, "atr_zero"

    volume_mean = volume_means.get(symbol, 0.0)
    if volume_mean > 0 and tick_volume < volume_mean:
        return None, "volume_filter_fail"

    if not np.isfinite(lrr) or lrr <= asset_params["lrr_min"]:
        return None, "lrr_fail"

    direction = None
    signal_type = ""

    if entry_price > london_high:
        direction = 1
        signal_type = f"Breakout ALCISTA (London High={london_high:.{asset_params['digits']}f})"

    elif entry_price < london_low:
        direction = -1
        signal_type = f"Breakout BAJISTA (London Low={london_low:.{asset_params['digits']}f})"

    elif candle_range > asset_params["atr_mult"] * atr_value:
        direction = -1 if signal_row["close"] > signal_row["open"] else 1
        signal_type = (
            f"Vela grande ({candle_range:.{asset_params['digits']}f} > "
            f"{asset_params['atr_mult']}xATR)"
        )

    if direction is None:
        return None, "no_direction"

    # Direction override — mirrors backtest_runner force_direction block exactly.
    # Signal must still exist (breakout/large-candle + LRR + volume all apply).
    # Generic: only active when asset_params defines force_direction.
    force_dir = asset_params.get("force_direction")
    _force_applied = False
    if force_dir is not None and direction != force_dir:
        signal_type += f' [→{"LONG" if force_dir == 1 else "SHORT"}]'
        direction = force_dir
        _force_applied = True

    if direction == 1:
        stop_loss = entry_price * (1 - asset_params["sl_pct"])
        stop_loss = max(stop_loss, london_low)
    else:
        stop_loss = entry_price * (1 + asset_params["sl_pct"])
        stop_loss = min(stop_loss, london_high)

    if (direction == 1 and stop_loss >= entry_price) or (
        direction == -1 and stop_loss <= entry_price
    ):
        return None, "invalid_stop_geometry"

    stop_distance = max(
        abs(entry_price - stop_loss) + spread_signal,
        entry_price * 0.0015,
    )

    if stop_distance < 1e-8:
        return None, "stop_distance_zero"

    lots = calc_lots(
        capital=initial_capital_per_asset,
        risk_pct=asset_params["risk_pct"],
        sl_dist=stop_distance,
        entry=entry_price,
        p=asset_params,
    )

    if lots <= 0:
        return None, "lots_zero"

    reason = "ok_force_direction" if _force_applied else "ok"
    return Signal(
        symbol=symbol,
        direction=direction,
        stype=signal_type,
        ep=entry_price,
        sl=stop_loss,
        sl_dist=stop_distance,
        lots=lots,
        atr=atr_value,
        lh=london_high,
        ll=london_low,
        lrr=lrr,
        spread_signal=spread_signal,
        signal_time=signal_row["time"],
        signal_tick_volume=tick_volume,
    ), reason
