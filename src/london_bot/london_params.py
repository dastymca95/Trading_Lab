from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional

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
            p = {
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
            for opt_key in ("force_direction", "be_atr_mult",
                            "low_vol_pct", "low_vol_win",
                            "roll_mfe_n", "roll_mfe_min"):
                if opt_key in s:
                    p[opt_key] = s[opt_key]
            final[symbol] = p
            continue

        digits = int(info.digits)
        point = float(info.point) if info.point else (10 ** (-digits) if digits > 0 else 1.0)

        spread_now = get_live_spread_fallback(
            symbol=symbol,
            digits=digits,
            fallback_spread=s["fallback_sp"],
        )
        spread_fallback = max(spread_now, point)

        p = {
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
        # Forward optional strategy keys — no-op for assets that don't define them
        for opt_key in ("force_direction", "be_atr_mult",
                        "low_vol_pct", "low_vol_win",
                        "roll_mfe_n", "roll_mfe_min"):
            if opt_key in s:
                p[opt_key] = s[opt_key]
        final[symbol] = p

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


def build_daily_filter(
    symbol: str,
    asset_params: Dict[str, Any],
    logger,
) -> Optional[FrozenSet[date]]:
    """
    Computes the set of allowed trading dates for an asset based on its
    daily regime filter keys (low_vol_pct, low_vol_win, roll_mfe_n, roll_mfe_min).

    Returns:
        frozenset of allowed date objects, or None if no filter is configured
        for this asset (all days allowed).

    Mirrors backtest_runner._build_daily_filter exactly (causal, shift=1, no look-ahead).
    Must be called at startup and refreshed at each new trading day.

    Data source: tries data/parquet/ first, then data/backtesting/. Requires M2 parquet
    with columns: time, high, low, open, atr14.
    """
    lv_pct = asset_params.get("low_vol_pct")
    lv_win = asset_params.get("low_vol_win")
    if lv_pct is None or lv_win is None:
        return None  # no filter configured for this asset

    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "data" / "parquet"     / f"{symbol}_Data.parquet",
        project_root / "data" / "backtesting" / f"{symbol}_Data.parquet",
    ]
    df = None
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_parquet(path, engine="pyarrow")
                break
            except Exception as e:
                logger.warning(f"{symbol}: build_daily_filter could not read {path}: {e}")

    if df is None:
        logger.warning(
            f"{symbol}: build_daily_filter — no parquet found in {[str(c) for c in candidates]}. "
            f"Daily filter DISABLED (all days allowed). Update parquet to enable."
        )
        return None

    # ── Replicate backtest_runner._build_daily_filter exactly ─────────────────
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["_date"] = df["time"].dt.date

    # Compute ATR14 if not already present
    if "atr14" not in df.columns:
        df["pc"] = df["close"].shift(1)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(abs(df["high"] - df["pc"]), abs(df["low"] - df["pc"])),
        )
        df["atr14"] = df["tr"].rolling(14).mean()

    # LOW_VOL base filter
    daily_atr = df.groupby("_date")["atr14"].mean()
    rolling_thr = (
        daily_atr
        .rolling(lv_win, min_periods=lv_win // 2)
        .quantile(lv_pct / 100.0)
    )
    allowed: FrozenSet[date] = frozenset(
        d for d in daily_atr.index
        if pd.notna(rolling_thr.loc[d]) and daily_atr.loc[d] <= rolling_thr.loc[d]
    )

    # roll_mfe quality gate
    roll_mfe_n   = asset_params.get("roll_mfe_n")
    roll_mfe_min = asset_params.get("roll_mfe_min")
    if roll_mfe_n is not None and roll_mfe_min is not None:
        # MT5 hours 18+19 = UTC 16-17 (matches backtest_runner parity)
        sess = df[df["time"].dt.hour.isin([18, 19])].copy()
        mfe_rows = []
        for d_s, grp in sess.groupby("_date"):
            if d_s not in allowed:          # only LOW_VOL days
                continue
            o_s = grp["open"].iloc[0]
            mfe = grp["high"].max() - o_s
            mae = o_s - grp["low"].min()
            if mae > 1.0:
                mfe_rows.append({"date": d_s, "mfe_mae": mfe / mae})
        if mfe_rows:
            mfe_s = (
                pd.DataFrame(mfe_rows)
                .sort_values("date")
                .reset_index(drop=True)
            )
            mfe_s["mfe_roll"] = (
                mfe_s["mfe_mae"]
                .shift(1)
                .rolling(roll_mfe_n, min_periods=roll_mfe_n // 2)
                .median()
            )
            mfe_pass: FrozenSet[date] = frozenset(
                mfe_s.loc[mfe_s["mfe_roll"] > roll_mfe_min, "date"]
            )
            allowed = frozenset(d for d in allowed if d in mfe_pass)

    logger.info(
        f"{symbol}: daily_filter built — {len(allowed)} allowed dates "
        f"(LOW_VOL p{lv_pct} w{lv_win}"
        + (f" + roll_mfe>{roll_mfe_min}" if roll_mfe_n else "")
        + ")"
    )
    return allowed


def init_daily_filters(
    asset_params: Dict[str, Dict[str, Any]],
    logger,
) -> Dict[str, Optional[FrozenSet[date]]]:
    """
    Builds daily regime filters for all assets at startup.
    Returns dict: symbol -> frozenset(allowed dates) or None (no filter).
    Call again at each new trading day to include today's date if eligible.
    """
    filters: Dict[str, Optional[FrozenSet[date]]] = {}
    for symbol, params in asset_params.items():
        filters[symbol] = build_daily_filter(symbol, params, logger)
    return filters