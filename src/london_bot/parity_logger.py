from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
import csv

from src.core.signal import Signal


PARITY_COLUMNS = [
    "timestamp",
    "bot_version",
    "symbol",
    "signal_hour",
    "mt5_time",
    "can_eval",
    "eval_reason",
    "signal_found",
    "signal_type",
    "direction",
    "entry_price",
    "stop_loss",
    "stop_distance",
    "lots",
    "lrr",
    "spread_signal",
    "exec_allowed",
    "exec_reason",
]


def ensure_parity_csv(parity_file: str) -> None:
    path = Path(parity_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(PARITY_COLUMNS)


def write_parity_row(
    parity_file: str,
    bot_version: str,
    symbol: str,
    signal_hour: int,
    mt5_time: datetime,
    can_eval: bool,
    eval_reason: str,
    signal_found: bool,
    signal: Optional[Signal],
    exec_allowed: bool,
    exec_reason: str,
) -> None:
    path = Path(parity_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(),
        "bot_version": bot_version,
        "symbol": symbol,
        "signal_hour": signal_hour,
        "mt5_time": mt5_time.isoformat(),
        "can_eval": can_eval,
        "eval_reason": eval_reason,
        "signal_found": signal_found,
        "signal_type": signal.stype if signal else "",
        "direction": signal.direction if signal else "",
        "entry_price": signal.ep if signal else "",
        "stop_loss": signal.sl if signal else "",
        "stop_distance": signal.sl_dist if signal else "",
        "lots": signal.lots if signal else "",
        "lrr": signal.lrr if signal else "",
        "spread_signal": signal.spread_signal if signal else "",
        "exec_allowed": exec_allowed,
        "exec_reason": exec_reason,
    }

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PARITY_COLUMNS)
        writer.writerow(row)