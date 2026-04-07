from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import MetaTrader5 as mt5

from src.core.position import PositionState
from src.shared.mt5_connector import get_positions, history_deals_get


def ensure_audit_csv(
    audit_file: str,
    audit_columns: List[str],
    logger,
) -> None:
    """
    Asegura que exista el CSV de auditoría con el esquema esperado.
    Si existe pero tiene columnas incompatibles, lo respalda y recrea.
    """
    audit_path = Path(audit_file)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    if not audit_path.exists():
        pd.DataFrame(columns=audit_columns).to_csv(audit_path, index=False)
        return

    try:
        existing_columns = list(pd.read_csv(audit_path, nrows=0).columns)

        if existing_columns != audit_columns:
            backup_path = audit_path.parent / (
                f"{audit_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{audit_path.suffix}"
            )
            audit_path.replace(backup_path)
            pd.DataFrame(columns=audit_columns).to_csv(audit_path, index=False)

            logger.warning(
                f"AUDIT CSV incompatible. Respaldado en {backup_path.name} y recreado."
            )

    except Exception:
        pd.DataFrame(columns=audit_columns).to_csv(audit_path, index=False)


def find_exit_deals(
    ticket: int,
    bot_magic: int,
    lookback_hours: int = 168,
) -> List[Any]:
    """
    Busca deals de salida asociados a una posición.
    """
    try:
        deals = history_deals_get(
            datetime.now() - timedelta(hours=lookback_hours),
            datetime.now(),
        )

        if not deals:
            return []

        exit_deals = [
            d
            for d in deals
            if d.magic == bot_magic and d.position_id == ticket and d.entry == 1
        ]

        return sorted(exit_deals, key=lambda d: d.time)

    except Exception:
        return []


def _classify_close_reason(position: PositionState, close_price: float) -> str:
    """Clasifica el tipo de cierre según el estado del trailing/BE."""
    tol = 10 ** (-position.digits)
    if position.breakeven_hit:
        if abs(close_price - position.be_level) <= tol:
            return "SL_BE"
        return "SL_TRAILING"
    return "SL_INITIAL"


def audit_closed_position(
    position: PositionState,
    exit_reason: str,
    audit_file: str,
    audit_columns: List[str],
    bot_magic: int,
    logger,
) -> None:
    """
    Audita una posición cerrada:
    - PnL real según history_deals_get
    - PnL aproximado tipo backtest
    - slippage registrado en la entrada
    """
    symbol = position.symbol
    ticket = position.ticket
    direction = position.direction
    lots = position.lots

    try:
        exit_deals = find_exit_deals(ticket=ticket, bot_magic=bot_magic)

        if not exit_deals:
            logger.warning(
                f"{symbol} ticket={ticket}: no se encontraron exit deals para auditoría."
            )
            return

        pnl_real = float(sum(getattr(d, "profit", 0.0) for d in exit_deals))
        close_price = float(exit_deals[-1].price)

        backtest_entry_price = float(position.signal_price)
        spread_signal = position.spread_signal

        if position.jpy:
            avg_price = (backtest_entry_price + close_price) / 2
            raw_bt = (
                (close_price - backtest_entry_price)
                * direction
                * lots
                * position.cs
                / max(avg_price, 1)
            )
        else:
            raw_bt = (
                (close_price - backtest_entry_price)
                * direction
                * lots
                * position.cs
            )

        pnl_backtest_approx = (
            raw_bt
            - lots * position.comm
            - spread_signal * lots * position.cs
        )

        row = {
            "datetime": datetime.now().isoformat(),
            "asset": symbol,
            "direction": "BUY" if direction == 1 else "SELL",
            "lots": lots,
            "signal_price": position.signal_price,
            "real_entry_price": position.real_entry_price,
            "slippage_pts": position.slippage,
            "spread_signal": spread_signal,
            "spread_at_entry": position.spread,
            "execution_ms": position.exec_ms,
            "pnl_real": round(pnl_real, 2),
            "pnl_backtest_approx": round(pnl_backtest_approx, 2),
            "exit_reason": exit_reason,
            "ticket": position.ticket,
            "be_level": position.be_level,
            "be_hit": position.breakeven_hit,
            "best_price": position.best_price,
            "close_price": close_price,
            "close_reason_detail": _classify_close_reason(position, close_price),
        }

        audit_path = Path(audit_file)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([row], columns=audit_columns).to_csv(
            audit_path,
            mode="a",
            header=False,
            index=False,
        )

        logger.info(
            f"📋 Auditado: {symbol} | pnl_real=${pnl_real:+.2f} | "
            f"pnl_bt≈${pnl_backtest_approx:+.2f} | razón={exit_reason}"
        )

    except Exception as e:
        logger.error(f"Error auditando cierre de {symbol}: {e}")


def reconcile_with_mt5(
    open_positions: Dict[str, PositionState],
    trades_today: Dict[str, int],
    audit_file: str,
    audit_columns: List[str],
    bot_magic: int,
    logger,
) -> Tuple[Dict[str, PositionState], Dict[str, int]]:
    """
    Reconciliación al reiniciar el bot:
    - si la posición sigue viva en MT5, sincroniza el SL
    - si ya no existe, la audita y la elimina del estado
    """
    for symbol in list(open_positions.keys()):
        position = open_positions[symbol]
        ticket = position.ticket

        mt5_pos = get_positions(ticket=ticket)

        if mt5_pos:
            current_sl = float(mt5_pos[0].sl)

            if current_sl != position.sl:
                logger.info(
                    f"{symbol}: SL actualizado durante reinicio "
                    f"{position.sl} → {current_sl}"
                )
                position.sl = current_sl

            logger.info(f"✅ {symbol} ticket={ticket}: activa en MT5")

        else:
            logger.info(
                f"⚠️ {symbol} ticket={ticket}: cerrada mientras bot estaba inactivo"
            )

            audit_closed_position(
                position=position,
                exit_reason="SL_MIENTRAS_BOT_APAGADO",
                audit_file=audit_file,
                audit_columns=audit_columns,
                bot_magic=bot_magic,
                logger=logger,
            )

            open_positions.pop(symbol)

    return open_positions, trades_today