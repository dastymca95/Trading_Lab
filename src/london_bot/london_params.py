from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.shared.mt5_connector import (
    copy_rates_range,
    ensure_symbol_selected,
    get_live_spread_fallback,
    get_symbol_info,
)


def build_live_asset_params(
    asset_strategy: Dict[str, Dict[str, Any]],
    logger,
) -> Dict[str, Dict[str, Any]]:
    """
    Combina la config base por activo con datos reales obtenidos desde MT5.
    Si symbol_info() falla, usa los fallbacks definidos en YAML.
    """
    final: Dict[str, Dict[str, Any]] = {}

    for symbol, s in asset_strategy.items():
        info = get_symbol_info(symbol)

        if info is None:
            final[symbol] = {
                "sl_pct": s["sl_pct"],
                "trail_mult": s["trail_mult"],
                "risk_pct": s["risk_pct"],
                "lrr_min": s["lrr_min"],
                "hours": s["hours"],
                "dow": s["dow"],
                "atr_mult": s["atr_mult"],
                "comm": float(s["fallback_comm"]),
                "cs": int(s["fallback_cs"]),
                "ml": float(s["fallback_ml"]),
                "step": float(s["fallback_step"]),
                "sp": float(s["fallback_sp"]),
                "digits": int(s["fallback_digits"]),
                "jpy": bool(s["fallback_jpy"]),
            }
            continue

        digits = int(info.digits)
        point = float(info.point) if info.point else (10 ** (-digits) if digits > 0 else 1.0)

        spread_now = get_live_spread_fallback(
            symbol=symbol,
            digits=digits,
            fallback_spread=s["fallback_sp"],
        )
        spread_fallback = max(spread_now, point)

        final[symbol] = {
            "sl_pct": s["sl_pct"],
            "trail_mult": s["trail_mult"],
            "risk_pct": s["risk_pct"],
            "lrr_min": s["lrr_min"],
            "hours": s["hours"],
            "dow": s["dow"],
            "atr_mult": s["atr_mult"],
            "comm": float(s["fallback_comm"]),
            "cs": int(info.trade_contract_size) if info.trade_contract_size else int(s["fallback_cs"]),
            "ml": float(info.volume_min) if info.volume_min else float(s["fallback_ml"]),
            "step": float(info.volume_step) if info.volume_step else float(s["fallback_step"]),
            "sp": float(spread_fallback),
            "digits": digits,
            "jpy": bool(getattr(info, "currency_profit", "") == "JPY"),
        }

    return final


def validate_symbol(
    symbol: str,
    asset_params: Dict[str, Any],
    logger,
) -> bool:
    """
    Verifica que el símbolo exista y esté visible en MT5.
    Loggea datos útiles del broker para depuración.
    """
    info = get_symbol_info(symbol)

    if info is None:
        logger.error(f"{symbol}: símbolo no encontrado en MT5")
        return False

    ensure_symbol_selected(symbol)

    logger.info(
        f"{symbol}: "
        f"trade_mode={getattr(info, 'trade_mode', None)} | "
        f"trade_exemode={getattr(info, 'trade_exemode', None)} | "
        f"filling_mode={getattr(info, 'filling_mode', None)} | "
        f"volume_min={getattr(info, 'volume_min', None)} | "
        f"volume_step={getattr(info, 'volume_step', None)} | "
        f"stops_level={getattr(info, 'trade_stops_level', None)} | "
        f"freeze_level={getattr(info, 'trade_freeze_level', None)}"
    )
    return True


def load_parquet_volume_mean(symbol: str, logger) -> float | None:
    """
    Carga la media histórica de tick_volume desde un parquet local.
    Busca el archivo en data/parquet/<SYMBOL>_Data.parquet
    """
    project_root = Path(__file__).resolve().parents[2]
    parquet_path = project_root / "data" / "parquet" / f"{symbol}_Data.parquet"

    if not parquet_path.exists():
        return None

    try:
        df = pd.read_parquet(parquet_path, engine="pyarrow", columns=["tick_volume"])
        if len(df) == 0:
            return None

        volume_mean = float(pd.to_numeric(df["tick_volume"], errors="coerce").dropna().mean())
        return volume_mean if np.isfinite(volume_mean) else None

    except Exception as e:
        logger.warning(f"{symbol}: no se pudo leer parquet para vm_global ({e})")
        return None


def load_mt5_volume_mean(
    symbol: str,
    volume_history_from: str,
    logger,
) -> float | None:
    """
    Calcula la media histórica de tick_volume usando velas M2 desde MT5.
    """
    try:
        date_from = datetime.fromisoformat(volume_history_from)
        date_to = datetime.now()

        rates = copy_rates_range(symbol,  mt5_timeframe_m2(), date_from, date_to)
        if rates is None or len(rates) == 0:
            return None

        volume_mean = float(pd.DataFrame(rates)["tick_volume"].mean())
        return volume_mean if np.isfinite(volume_mean) else None

    except Exception as e:
        logger.warning(f"{symbol}: no se pudo calcular vm_global desde MT5 ({e})")
        return None


def init_volume_means(
    asset_params: Dict[str, Dict[str, Any]],
    volume_history_from: str,
    prefer_parquet: bool,
    logger,
) -> Dict[str, float]:
    """
    Inicializa medias históricas de volumen para todos los símbolos.
    Prioriza parquet o MT5 según configuración.
    """
    volume_means: Dict[str, float] = {}

    logger.info("Calculando medias de volumen histórico...")

    for symbol in asset_params:
        volume_mean = None
        source = "disabled"

        if prefer_parquet:
            volume_mean = load_parquet_volume_mean(symbol, logger=logger)
            source = "parquet" if volume_mean is not None else "mt5_history"

            if volume_mean is None:
                volume_mean = load_mt5_volume_mean(
                    symbol=symbol,
                    volume_history_from=volume_history_from,
                    logger=logger,
                )
        else:
            volume_mean = load_mt5_volume_mean(
                symbol=symbol,
                volume_history_from=volume_history_from,
                logger=logger,
            )
            source = "mt5_history" if volume_mean is not None else "parquet"

            if volume_mean is None:
                volume_mean = load_parquet_volume_mean(symbol, logger=logger)

        if volume_mean is None:
            volume_mean = 0.0
            source = "disabled"

        volume_means[symbol] = float(volume_mean)
        logger.info(f"  {symbol}: vm_global={volume_mean:.1f} | source={source}")

    return volume_means


def mt5_timeframe_m2():
    """
    Helper aislado para evitar importar MetaTrader5 en el main solo por el timeframe.
    """
    import MetaTrader5 as mt5
    return mt5.TIMEFRAME_M2