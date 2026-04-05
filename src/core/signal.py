from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Signal:
    # ── Obligatorios ──────────────────────────────────────────────────────
    symbol:             str
    direction:          int        # 1 = BUY  |  -1 = SELL
    ep:                 float      # entry price
    sl:                 float      # stop loss
    lots:               float
    stype:              str        # descripción del tipo de señal

    # ── Opcionales con default ────────────────────────────────────────────
    sl_dist:            float               = 0.0
    atr:                float               = 0.0
    lh:                 float               = 0.0   # London High
    ll:                 float               = 0.0   # London Low
    lrr:                float               = 0.0   # London Range Ratio
    spread_signal:      float               = 0.0
    signal_tick_volume: float               = 0.0
    signal_time:        Optional[datetime]  = None

