from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class PositionState:
    # ── Obligatorios (disponibles al abrir la posición) ───────────────────
    ticket:             int
    symbol:             str
    direction:          int        # 1 = BUY  |  -1 = SELL
    lots:               float
    real_entry_price:   float
    signal_price:       float
    sl:                 float
    be_level:           float      # precio de breakeven
    best_price:         float      # mejor precio alcanzado desde apertura
    open_time:          datetime
    digits:             int        # decimales del símbolo
    cs:                 int        # contract size
    jpy:                bool       # True si currency_profit == JPY
    comm:               float      # comisión round-trip
    trail_mult:         float      # multiplicador del trailing stop
    stype:              str        # tipo de señal que originó la posición

    # ── Opcionales / calculados post-apertura ─────────────────────────────
    backtest_price:     float      = 0.0
    slippage:           float      = 0.0
    spread:             float      = 0.0   # spread real en entry
    spread_signal:      float      = 0.0   # spread al momento de la señal
    exec_ms:            int        = 0     # latencia de ejecución en ms
    atr:                float      = 0.0   # ATR14; se asigna desde Signal en main
    breakeven_hit:      bool       = False

    # ── Compatibilidad temporal con código dict-based ─────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PositionState:
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

