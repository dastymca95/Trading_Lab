from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from src.core.position import PositionState


def _serialize_position(position: PositionState) -> Dict[str, Any]:
    """
    Convierte una posición a un formato serializable en JSON.
    """
    serialized: Dict[str, Any] = {}

    for key, value in position.to_dict().items():
        if isinstance(value, datetime):
            serialized[key] = {"__datetime__": value.isoformat()}
        elif isinstance(value, (int, float, bool, str)) or value is None:
            serialized[key] = value
        else:
            serialized[key] = str(value)

    return serialized


def _deserialize_position(position: Dict[str, Any]) -> PositionState:
    """
    Reconstruye una posición desde el JSON guardado.
    """
    deserialized: Dict[str, Any] = {}

    int_fields = {
        "ticket",
        "direction",
        "digits",
        "cs",
        "exec_ms",
    }

    float_fields = {
        "lots",
        "real_entry_price",
        "signal_price",
        "backtest_price",
        "slippage",
        "spread",
        "spread_signal",
        "sl",
        "atr",
        "trail_mult",
        "be_level",
        "best_price",
        "comm",
    }

    bool_fields = {
        "breakeven_hit",
        "jpy",
    }

    for key, value in position.items():
        if isinstance(value, dict) and "__datetime__" in value:
            deserialized[key] = datetime.fromisoformat(value["__datetime__"])
        elif key in int_fields:
            deserialized[key] = int(float(value))
        elif key in float_fields:
            deserialized[key] = float(value)
        elif key in bool_fields:
            if isinstance(value, bool):
                deserialized[key] = value
            else:
                deserialized[key] = str(value).lower() == "true"
        else:
            deserialized[key] = value

    return PositionState.from_dict(deserialized)


def save_state(
    open_positions: Dict[str, PositionState],
    trades_today: Dict[str, int],
    state_file: str,
    logger,
) -> None:
    """
    Guarda el estado actual del bot en un JSON.
    """
    try:
        state_path = Path(state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "open_positions": {
                symbol: _serialize_position(position)
                for symbol, position in open_positions.items()
            },
            "trades_today": trades_today,
            "saved_at": datetime.now().isoformat(),
        }

        tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        tmp_path.replace(state_path)

    except Exception as e:
        logger.error(f"Error guardando estado: {e}")


def load_state(state_file: str, logger) -> Tuple[Dict[str, PositionState], Dict[str, int]]:
    """
    Carga el estado del bot desde disco.
    """
    state_path = Path(state_file)

    if not state_path.exists():
        return {}, {}

    try:
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        open_positions = {
            symbol: _deserialize_position(position)
            for symbol, position in state.get("open_positions", {}).items()
        }

        trades_today = {
            key: int(value)
            for key, value in state.get("trades_today", {}).items()
        }

        logger.info(
            f"Estado cargado (guardado: {state.get('saved_at', '?')}) | "
            f"Posiciones: {list(open_positions.keys())} | "
            f"Trades hoy: {trades_today}"
        )

        return open_positions, trades_today

    except Exception as e:
        logger.error(f"Error cargando estado: {e}")
        return {}, {}