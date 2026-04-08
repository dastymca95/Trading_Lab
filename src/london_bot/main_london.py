from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from src.shared.config_loader import load_app_config
from src.shared.logger_setup import setup_logger
from src.shared.mt5_connector import connect_mt5, get_account_info, get_mt5_now, get_terminal_info, shutdown_mt5, is_mt5_connected, get_spread_points
from src.london_bot.state_manager import load_state, save_state
from src.london_bot.validation_london import (
    ensure_audit_csv,
    reconcile_with_mt5,
    audit_closed_position,
    check_runtime_divergence,
)
from src.london_bot.paper_signal_audit import (
    ensure_signal_audit_csv,
    write_signal_audit_row,
    compute_and_write_session_summary,
)
from src.london_bot.signal_london import check_signal
from src.london_bot.order_manager_london import open_position
from src.london_bot.trade_manager_london import update_trailing
from src.london_bot.london_params import (
    build_live_asset_params,
    validate_symbol,
    init_volume_means,
    init_daily_filters,
)
from src.london_bot.scheduler_london import (
    is_new_mt5_day,
    reset_daily_trades,
    get_current_time_parts,
    is_signal_scan_window,
    build_signal_key,
    should_run_for_asset,
    should_emit_heartbeat,
)
from src.london_bot.risk_manager_london import (
    can_evaluate_signal,
    can_execute_trade,
)
from src.london_bot.pnl_tracker_london import log_pnl_summary
from src.london_bot.manual_test import force_demo_trade
from src.london_bot.parity_logger import ensure_parity_csv, write_parity_row, write_parity_fill
from src.core.execution import ExecutionDecision
from src.core.position import PositionState
from src.core.signal import Signal


def main() -> None:
    parser = argparse.ArgumentParser(description="London Bot")
    parser.add_argument("--env", default=None,
                        help="Environment override: paper | demo | production")
    args, _ = parser.parse_known_args()

    config = load_app_config("london_bot", env_name=args.env)

    # Anchor configured paths to project root — CWD-independent
    _root = config["project_root"]
    config["paths"]["state_file"]         = os.path.join(_root, config["paths"]["state_file"])
    config["paths"]["audit_file"]         = os.path.join(_root, config["paths"]["audit_file"])
    config["paths"]["log_dir"]            = os.path.join(_root, config["paths"]["log_dir"])
    config["paths"]["parity_file"]        = os.path.join(_root, config["paths"]["parity_file"])
    config["paths"]["signal_audit_file"]  = os.path.join(_root, config["paths"]["signal_audit_file"])
    config["paths"]["session_summary_file"] = os.path.join(_root, config["paths"]["session_summary_file"])

    log = setup_logger(
        module_name="london_bot",
        log_dir=config["paths"]["log_dir"],
        level=config["logging"]["level"],
    )
    log.info(
        "Paper audit paths | signal_audit=%s | session_summary=%s",
        config["paths"]["signal_audit_file"],
        config["paths"]["session_summary_file"],
    )

    log.info("╔════════════════════════════════════════════════════════════╗")
    log.info("║ LONDON BOT — STRUCTURED VERSION                          ║")
    log.info("╚════════════════════════════════════════════════════════════╝")

    # ── PRIORITY 4: fail-safe — paper + execution_enabled must never be True ──
    _exec_enabled = bool(config["mode"].get("execution_enabled", False))
    _env_is_paper  = config["env_name"] == "paper"
    if _env_is_paper and _exec_enabled:
        log.error("╔══════════════════════════════════════════════════════════════╗")
        log.error("║  !!  SAFETY VIOLATION  !!                                   ║")
        log.error("║  env=paper requires execution_enabled=False                 ║")
        log.error("║  Current config has execution_enabled=True                  ║")
        log.error("║  Fix: set execution_enabled: false in environments/paper.yaml║")
        log.error("╚══════════════════════════════════════════════════════════════╝")
        return

    ensure_audit_csv(
        audit_file=config["paths"]["audit_file"],
        audit_columns=config["audit"]["columns"],
        logger=log,
    )
    if config["parity"]["enabled"]:
        ensure_parity_csv(
            parity_file=config["paths"]["parity_file"],
        )

    ensure_signal_audit_csv(config["paths"]["signal_audit_file"])
    log.info(
        "Signal audit CSV ready | file=%s",
        config["paths"]["signal_audit_file"],
    )

    if not connect_mt5(
        logger=log,
        terminal_path=config["mt5"].get("terminal_path"),
    ):
        log.error("No se pudo conectar a MT5.")
        return

    asset_params = build_live_asset_params(
        asset_strategy=config["assets"],
        logger=log,
    )

    if config["testing"]["force_test_trade"]:
        test_direction = int(config["testing"]["force_test_direction"])
        if test_direction not in (1, -1):
            log.error("force_test_direction debe ser 1 (BUY) o -1 (SELL)")
            return

        pos = force_demo_trade(
            symbol=config["testing"]["force_test_symbol"],
            direction=test_direction,
            asset_params=asset_params,
            bot_magic=config["bot"]["magic"],
            deviation=config["execution"]["deviation"],
            logger=log,
        )

        if pos:
            open_positions_test = {
                config["testing"]["force_test_symbol"]: pos
            }
            trades_today_test = {
                config["testing"]["force_test_symbol"]: 1
            }

            save_state(
                open_positions=open_positions_test,
                trades_today=trades_today_test,
                state_file=config["paths"]["state_file"],
                logger=log,
            )

        if config["testing"]["force_test_return_after_open"]:
            log.info("🧪 Prueba manual completada. Script finalizado por configuración.")
            shutdown_mt5()
            return

    log.info(
        f"env={config['env_name']} | "
        f"capital/activo=${config['bot']['initial_capital_per_asset']} | "
        f"mt5_offset={config['mt5']['offset_hours']}h"
    )

    for symbol, params in asset_params.items():
        validate_symbol(symbol=symbol, asset_params=params, logger=log)

    volume_means = init_volume_means(
        asset_params=asset_params,
        volume_history_from=config["filters"]["volume_history_from"],
        prefer_parquet=config["filters"]["prefer_parquet_for_volume_mean"],
        logger=log,
    )

    daily_filters = init_daily_filters(asset_params=asset_params, logger=log)

    # ── PRIORITY 4: fail-safe — daily_filter build must succeed for filtered assets ──
    for _sym, _df_set in daily_filters.items():
        _has_filter_cfg = asset_params[_sym].get("low_vol_pct") is not None
        if _has_filter_cfg and _df_set is None:
            log.warning(
                f"⚠️  FAIL-SAFE | {_sym}: daily_filter configured but NOT built "
                f"(parquet missing?). ALL dates allowed for {_sym} — "
                f"regime filter is inactive. Update parquet to re-enable."
            )

    open_positions, trades_today = load_state(
        state_file=config["paths"]["state_file"],
        logger=log,
    )

    open_positions, trades_today = reconcile_with_mt5(
        open_positions=open_positions,
        trades_today=trades_today,
        audit_file=config["paths"]["audit_file"],
        audit_columns=config["audit"]["columns"],
        bot_magic=config["bot"]["magic"],
        logger=log,
    )
    log_pnl_summary(
        audit_file=config["paths"]["audit_file"],
        logger=log,
    )

    save_state(
        open_positions=open_positions,
        trades_today=trades_today,
        state_file=config["paths"]["state_file"],
        logger=log,
    )

    def _write_runtime_event_rows(event_reason: str, notes: str = "") -> None:
        event_local = datetime.now()
        event_mt5 = get_mt5_now(offset_hours=config["mt5"]["offset_hours"])

        for symbol, params in asset_params.items():
            df_set = daily_filters.get(symbol)
            has_df_cfg = params.get("low_vol_pct") is not None

            try:
                spread_raw = get_spread_points(symbol)
            except Exception:
                spread_raw = 0.0

            spread_pts = spread_raw if spread_raw and spread_raw > 0 else None

            write_signal_audit_row(
                audit_file=config["paths"]["signal_audit_file"],
                ts_local=event_local,
                ts_mt5=event_mt5,
                env=config["env_name"],
                symbol=symbol,
                scan_hour=event_mt5.hour,
                scan_minute=event_mt5.minute,
                allowed_hour=event_mt5.hour in params["hours"],
                allowed_dow=event_mt5.weekday() in params["dow"],
                daily_filter_loaded=(not has_df_cfg) or (df_set is not None),
                daily_filter_pass=df_set is None or event_mt5.date() in df_set,
                volume_filter_pass=None,
                spread_points=spread_pts,
                force_direction_applied=False,
                raw_signal_detected=None,
                final_signal_direction=None,
                state_has_open_position=symbol in open_positions,
                execution_enabled=bool(config["mode"].get("execution_enabled", False)),
                action_taken="RUNTIME_EVENT",
                block_reason=event_reason,
                notes=notes,
                logger=log,
            )

        log.info(
            "Signal audit event written | event=%s | file=%s",
            event_reason,
            config["paths"]["signal_audit_file"],
        )

    last_signal_check: Dict[str, bool] = {}
    last_out_of_window_audit_hour: Dict[str, int] = {}
    last_date = None

    # Contador de fallos consecutivos de tick por símbolo (detección de desconexión MT5)
    _api_error_counts: Dict[str, int] = {}
    _API_ERR_WARN_FIRST  = 3   # fallos antes del primer warning  (3 × 10s = 30s)
    _API_ERR_WARN_REPEAT = 30  # fallos entre warnings periódicos (30 × 10s = 5 min)

    signal_hours = config["strategy"]["signal_hours_mt5"]
    scan_minutes = config["strategy"]["scan_minutes_after_signal"]

    log.info(
        f"Bot corriendo | scan global: horas MT5 {signal_hours} min {scan_minutes}"
    )
    for _sym, _p in asset_params.items():
        log.info(
            f"  {_sym} | active hours: {_p['hours']} | dow: {_p['dow']}"
        )
    _write_runtime_event_rows("startup", notes="bot_start")

    try:
        while True:
            now = datetime.now()
            mt5_now = get_mt5_now(offset_hours=config["mt5"]["offset_hours"])

            # Nuevo día MT5
            if is_new_mt5_day(last_date, mt5_now):
                # ── session summary for the day that just ended ───────────────
                if last_date is not None:
                    compute_and_write_session_summary(
                        audit_file=config["paths"]["signal_audit_file"],
                        summary_file=config["paths"]["session_summary_file"],
                        for_date=last_date,
                        logger=log,
                    )

                last_date = mt5_now.date()
                trades_today = reset_daily_trades(asset_params)
                last_signal_check.clear()
                last_out_of_window_audit_hour.clear()
                daily_filters = init_daily_filters(asset_params=asset_params, logger=log)

                save_state(
                    open_positions=open_positions,
                    trades_today=trades_today,
                    state_file=config["paths"]["state_file"],
                    logger=log,
                )

                log.info(f"── Nuevo día MT5: {mt5_now.date()} ──")

            # Gestionar trailing / detectar cierres
            state_changed = False

            for symbol, pos in list(open_positions.items()):
                digits = pos.digits
                sl_before = pos.sl
                be_before = pos.breakeven_hit
                bp_before = pos.best_price

                status = update_trailing(position=pos, logger=log)

                if status == "stopped":
                    audit_closed_position(
                        position=pos,
                        exit_reason="SL_MT5",
                        audit_file=config["paths"]["audit_file"],
                        audit_columns=config["audit"]["columns"],
                        bot_magic=config["bot"]["magic"],
                        logger=log,
                    )

                    log_pnl_summary(
                        audit_file=config["paths"]["audit_file"],
                        logger=log,
                    )

                    open_positions.pop(symbol)
                    _api_error_counts.pop(symbol, None)
                    state_changed = True

                elif status == "error":
                    _api_error_counts[symbol] = _api_error_counts.get(symbol, 0) + 1
                    n = _api_error_counts[symbol]

                    if n == _API_ERR_WARN_FIRST:
                        if not is_mt5_connected():
                            log.warning(
                                f"🔴 {symbol} | MT5 DESCONECTADO | trailing suspendido tras "
                                f"{n} fallos de tick | verifique el terminal y reinicie el bot"
                            )
                        else:
                            log.warning(
                                f"⚠️  {symbol} | tick no disponible ({n} intentos) | "
                                f"terminal accesible pero símbolo sin datos | "
                                f"posición sin protección activa"
                            )

                    elif n > _API_ERR_WARN_FIRST and (n - _API_ERR_WARN_FIRST) % _API_ERR_WARN_REPEAT == 0:
                        elapsed_s = n * config["bot"]["trailing_check_seconds"]
                        log.warning(
                            f"🔴 {symbol} | trailing aún inactivo | {n} fallos consecutivos "
                            f"({elapsed_s // 60}m {elapsed_s % 60}s) | SL original intacto"
                        )

                else:  # "ok" — trailing procesado normalmente
                    prev = _api_error_counts.pop(symbol, 0)
                    if prev > 0:
                        log.info(
                            f"✅ {symbol} | trailing restaurado tras {prev} fallos consecutivos"
                        )
                    if (
                        pos.sl != sl_before
                        or pos.breakeven_hit != be_before
                        or abs(pos.best_price - bp_before) > 10 ** (-digits)
                    ):
                        state_changed = True

            if state_changed:
                save_state(
                    open_positions=open_positions,
                    trades_today=trades_today,
                    state_file=config["paths"]["state_file"],
                    logger=log,
                )

            current_hour, current_minute = get_current_time_parts(mt5_now)

            # Escaneo de señales

            if config["parity"]["force_scan"] or is_signal_scan_window(mt5_now, signal_hours, scan_minutes):
                for symbol, params in asset_params.items():
                    # ── audit context shared for this symbol/cycle ─────────────
                    _df_set       = daily_filters.get(symbol)
                    _has_df_cfg   = params.get("low_vol_pct") is not None
                    _df_loaded    = (not _has_df_cfg) or (_df_set is not None)
                    _df_pass      = _df_set is None or mt5_now.date() in _df_set
                    _exec_en      = config["mode"].get("execution_enabled", False)
                    _spread_raw   = get_spread_points(symbol)
                    _spread_pts   = _spread_raw if _spread_raw and _spread_raw > 0 else None

                    # map eval_decision.reason → audit block_reason
                    _EVAL_REASON_MAP = {
                        "weekday no permitido":                     "out_of_dow",
                        "hora no permitida para el activo":         "out_of_hour",
                        "máximo de trades diarios alcanzado":       "max_trades_reached",
                        "ya existe posición abierta en este símbolo": "already_in_position",
                    }

                    # raw_signal/volume inference from check_signal reason code
                    _VOL_PASS_TRUE  = {"lrr_fail", "no_direction", "stop_distance_zero",
                                       "lots_zero", "ok", "ok_force_direction"}
                    _RAW_SIG_TRUE   = {"ok", "ok_force_direction", "stop_distance_zero", "lots_zero"}
                    _RAW_SIG_FALSE  = {"no_direction"}

                    def _write_audit(action: str, reason: str,
                                     sig_reason: str = "",
                                     sig=None, notes: str = "") -> None:
                        _raw = (True  if sig_reason in _RAW_SIG_TRUE
                                else False if sig_reason in _RAW_SIG_FALSE
                                else None)
                        _vol = (True  if sig_reason in _VOL_PASS_TRUE
                                else False if sig_reason == "volume_filter_fail"
                                else None)
                        write_signal_audit_row(
                            audit_file=config["paths"]["signal_audit_file"],
                            ts_local=now,
                            ts_mt5=mt5_now,
                            env=config["env_name"],
                            symbol=symbol,
                            scan_hour=current_hour,
                            scan_minute=current_minute,
                            allowed_hour=current_hour in params["hours"],
                            allowed_dow=mt5_now.weekday() in params["dow"],
                            daily_filter_loaded=_df_loaded,
                            daily_filter_pass=_df_pass,
                            volume_filter_pass=_vol,
                            spread_points=_spread_pts,
                            force_direction_applied=(sig_reason == "ok_force_direction"),
                            raw_signal_detected=_raw,
                            final_signal_direction=sig.direction if sig else None,
                            state_has_open_position=symbol in open_positions,
                            execution_enabled=bool(_exec_en),
                            action_taken=action,
                            block_reason=reason,
                            notes=notes,
                            logger=log,
                        )

                    if not config["parity"]["force_scan"] and not should_run_for_asset(current_hour, params):
                        _write_audit("EVAL_BLOCKED", "out_of_hour")
                        continue

                    key = build_signal_key(symbol, mt5_now, current_hour)
                    if key in last_signal_check:
                        continue

                    if symbol in open_positions:
                        log.info(f"{symbol}: evaluación omitida | reason=ya existe posición abierta en este símbolo")
                        _write_audit("POSITION_ALREADY_OPEN", "already_in_position")
                        last_signal_check[key] = True
                        continue

                    log.info(
                        f"🔍 Revisando vela {current_hour}:00 MT5 "
                        f"(scan {mt5_now.strftime('%H:%M')}) | "
                        f"capital=${config['bot']['initial_capital_per_asset']} | "
                        f"symbol={symbol}"
                    )

                    if config["parity"]["force_scan"]:
                        eval_decision = ExecutionDecision(allowed=True, reason="forced_parity_scan")
                    else:
                        eval_decision = can_evaluate_signal(
                            symbol=symbol,
                            signal_hour=current_hour,
                            mt5_now=mt5_now,
                            asset_params=params,
                            trades_today=trades_today.get(symbol, 0),
                            max_trades_per_day_per_asset=config["bot"]["max_trades_per_day_per_asset"],
                            open_positions=open_positions,
                        )

                    if not eval_decision.allowed:
                        log.info(f"{symbol}: evaluación omitida | reason={eval_decision.reason}")
                        _write_audit(
                            "EVAL_BLOCKED",
                            _EVAL_REASON_MAP.get(eval_decision.reason, eval_decision.reason),
                        )

                        if config["parity"]["enabled"]:
                            write_parity_row(
                                parity_file=config["paths"]["parity_file"],
                                bot_version="modular",
                                symbol=symbol,
                                signal_hour=current_hour,
                                mt5_time=mt5_now,
                                can_eval=eval_decision.allowed,
                                eval_reason=eval_decision.reason,
                                signal_found=False,
                                signal=None,
                                exec_allowed=False,
                                exec_reason="not_applicable",
                            )

                        last_signal_check[key] = True
                        continue

                    # ── check_signal — wrapped for PRIORITY 4 runtime_error ──
                    try:
                        sig, sig_reason = check_signal(
                            symbol=symbol,
                            asset_params=params,
                            signal_hour=current_hour,
                            mt5_now=mt5_now,
                            volume_means=volume_means,
                            initial_capital_per_asset=config["bot"]["initial_capital_per_asset"],
                            daily_filter=daily_filters.get(symbol),
                        )
                    except Exception as _exc:
                        log.error(f"{symbol}: exception en check_signal: {_exc}", exc_info=True)
                        _write_audit("RUNTIME_ERROR", "runtime_error", notes=str(_exc))
                        last_signal_check[key] = True
                        continue

                    if sig:
                        log.info(
                            f"⚡ {sig.stype} | "
                            f"lots={sig.lots} | "
                            f"sl={sig.sl:.{params['digits']}f} | "
                            f"lrr={sig.lrr:.2f} | "
                            f"spread_signal={sig.spread_signal:.{params['digits']}f} | "
                            f"tv={sig.signal_tick_volume:.1f} vs "
                            f"vm={volume_means.get(symbol, 0.0):.1f}"
                        )

                        exec_decision = can_execute_trade(
                            config=config,
                            symbol=symbol,
                            logger=log,
                        )
                        if config["parity"]["enabled"]:
                            write_parity_row(
                                parity_file=config["paths"]["parity_file"],
                                bot_version="modular",
                                symbol=symbol,
                                signal_hour=current_hour,
                                mt5_time=mt5_now,
                                can_eval=True,
                                eval_reason=eval_decision.reason,
                                signal_found=True,
                                signal=sig,
                                exec_allowed=exec_decision.allowed,
                                exec_reason=exec_decision.reason,
                            )

                        if not exec_decision.allowed:
                            log.info(f"{symbol}: ejecución omitida | reason={exec_decision.reason}")
                            # In paper mode this is the normal path: SIGNAL_ONLY
                            _action = (
                                "SIGNAL_ONLY"
                                if exec_decision.reason == "execution disabled"
                                else "ORDER_BLOCKED"
                            )
                            _reason = (
                                "execution_disabled"
                                if exec_decision.reason == "execution disabled"
                                else exec_decision.reason
                            )
                            _write_audit(_action, _reason, sig_reason=sig_reason, sig=sig)
                        else:
                            if config["env_name"] == "paper":
                                log.error(f"{symbol}: paper safety violation prevented order send")
                                _write_audit(
                                    "ORDER_BLOCKED",
                                    "paper_safety_block",
                                    sig_reason=sig_reason,
                                    sig=sig,
                                    notes="env=paper blocked order before open_position",
                                )
                                last_signal_check[key] = True
                                continue

                            pos = open_position(
                                symbol=sig.symbol,
                                direction=sig.direction,
                                lots=sig.lots,
                                sl=sig.sl,
                                asset_params=params,
                                signal_price=sig.ep,
                                spread_signal=sig.spread_signal,
                                signal_type=sig.stype,
                                bot_magic=config["bot"]["magic"],
                                deviation=config["execution"]["deviation"],
                                logger=log,
                            )

                            if pos:
                                pos.atr = sig.atr
                                open_positions[symbol] = pos
                                trades_today[symbol] = trades_today.get(symbol, 0) + 1

                                save_state(
                                    open_positions=open_positions,
                                    trades_today=trades_today,
                                    state_file=config["paths"]["state_file"],
                                    logger=log,
                                )

                                _write_audit("ORDER_SENT", "", sig_reason=sig_reason, sig=sig)

                                if config["parity"]["enabled"]:
                                    write_parity_fill(
                                        parity_file=config["paths"]["parity_file"],
                                        symbol=symbol,
                                        signal_hour=current_hour,
                                        mt5_time=mt5_now,
                                        ticket=pos.ticket,
                                        fill_price=pos.real_entry_price,
                                        entry_slippage_pts=pos.slippage,
                                        be_level=pos.be_level,
                                        initial_sl=pos.sl,
                                        atr_signal=pos.atr,
                                        execution_ms=pos.exec_ms,
                                    )

                    else:
                        log.info(f"— Sin señal | reason={sig_reason}")
                        _write_audit("NO_SIGNAL", sig_reason, sig_reason=sig_reason)

                        if config["parity"]["enabled"]:
                            write_parity_row(
                                parity_file=config["paths"]["parity_file"],
                                bot_version="modular",
                                symbol=symbol,
                                signal_hour=current_hour,
                                mt5_time=mt5_now,
                                can_eval=True,
                                eval_reason=eval_decision.reason,
                                signal_found=False,
                                signal=None,
                                exec_allowed=False,
                                exec_reason="no_signal",
                            )

                    last_signal_check[key] = True

            # Heartbeat horario
            if should_emit_heartbeat(now, current_minute):
                info = get_account_info()
                balance = info.balance if info else 0.0
                log.info(
                    f"[{mt5_now.strftime('%H:%M')} MT5] "
                    f"Balance=${balance:.2f} | "
                    f"Posiciones: {len(open_positions)} | "
                    f"Trades hoy: {dict(trades_today)}"
                )
                if info is None:
                    log.warning(
                        "🔴 get_account_info() devolvió None — MT5 posiblemente desconectado"
                    )

                # ── PRIORITY 3: runtime/state/MT5 divergence check ────────────
                check_runtime_divergence(
                    open_positions=open_positions,
                    asset_params=asset_params,
                    bot_magic=config["bot"]["magic"],
                    logger=log,
                )

                # ── PRIORITY 1: OUT_OF_WINDOW audit row for non-signal hours ──
                if current_hour not in signal_hours:
                    for _sym, _p in asset_params.items():
                        if last_out_of_window_audit_hour.get(_sym) == current_hour:
                            continue
                        _df_s = daily_filters.get(_sym)
                        _hdf  = _p.get("low_vol_pct") is not None
                        write_signal_audit_row(
                            audit_file=config["paths"]["signal_audit_file"],
                            ts_local=now,
                            ts_mt5=mt5_now,
                            env=config["env_name"],
                            symbol=_sym,
                            scan_hour=current_hour,
                            scan_minute=current_minute,
                            allowed_hour=current_hour in _p["hours"],
                            allowed_dow=mt5_now.weekday() in _p["dow"],
                            daily_filter_loaded=(not _hdf) or (_df_s is not None),
                            daily_filter_pass=_df_s is None or mt5_now.date() in _df_s,
                            volume_filter_pass=None,
                            spread_points=None,
                            force_direction_applied=False,
                            raw_signal_detected=None,
                            final_signal_direction=None,
                            state_has_open_position=_sym in open_positions,
                            execution_enabled=bool(config["mode"].get("execution_enabled", False)),
                            action_taken="OUT_OF_WINDOW",
                            block_reason="out_of_window",
                            logger=log,
                        )
                        last_out_of_window_audit_hour[_sym] = current_hour

            time.sleep(config["bot"]["trailing_check_seconds"])

    except KeyboardInterrupt:
        log.info("Bot detenido manualmente.")

    except Exception as e:
        log.error(f"Error inesperado: {e}", exc_info=True)

        terminal_info = get_terminal_info()
        if not terminal_info:
            log.warning("MT5 desconectado. Intentando reconectar...")
            connect_mt5(
                logger=log,
                terminal_path=config["mt5"].get("terminal_path"),
            )

    finally:
        save_state(
            open_positions=open_positions,
            trades_today=trades_today,
            state_file=config["paths"]["state_file"],
            logger=log,
        )
        _write_runtime_event_rows("shutdown", notes="bot_stop")
        # session summary for the current (partial) day on any shutdown
        compute_and_write_session_summary(
            audit_file=config["paths"]["signal_audit_file"],
            summary_file=config["paths"]["session_summary_file"],
            for_date=last_date or get_mt5_now(offset_hours=config["mt5"]["offset_hours"]).date(),
            logger=log,
        )
        shutdown_mt5()
        log.info("Bot finalizado.")


if __name__ == "__main__":
    main()
