from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, Any

from src.shared.config_loader import load_app_config
from src.shared.logger_setup import setup_logger
from src.shared.mt5_connector import connect_mt5, get_account_info, get_mt5_now, get_terminal_info, shutdown_mt5
from src.london_bot.state_manager import load_state, save_state
from src.london_bot.validation_london import (
    ensure_audit_csv,
    reconcile_with_mt5,
    audit_closed_position,
)
from src.london_bot.signal_london import check_signal
from src.london_bot.order_manager_london import open_position
from src.london_bot.trade_manager_london import update_trailing
from src.london_bot.london_params import (
    build_live_asset_params,
    validate_symbol,
    init_volume_means,
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
from src.london_bot.parity_logger import ensure_parity_csv, write_parity_row
from src.core.execution import ExecutionDecision
from src.core.position import PositionState
from src.core.signal import Signal


def main() -> None:
    config = load_app_config("london_bot")

    # Anchor configured paths to project root — CWD-independent
    _root = config["project_root"]
    config["paths"]["state_file"]  = os.path.join(_root, config["paths"]["state_file"])
    config["paths"]["audit_file"]  = os.path.join(_root, config["paths"]["audit_file"])
    config["paths"]["log_dir"]     = os.path.join(_root, config["paths"]["log_dir"])
    config["paths"]["parity_file"] = os.path.join(_root, config["paths"]["parity_file"])

    log = setup_logger(
        module_name="london_bot",
        log_dir=config["paths"]["log_dir"],
        level=config["logging"]["level"],
    )

    log.info("╔════════════════════════════════════════════════════════════╗")
    log.info("║ LONDON BOT — STRUCTURED VERSION                          ║")
    log.info("╚════════════════════════════════════════════════════════════╝")

    ensure_audit_csv(
        audit_file=config["paths"]["audit_file"],
        audit_columns=config["audit"]["columns"],
        logger=log,
    )
    if config["parity"]["enabled"]:
        ensure_parity_csv(
            parity_file=config["paths"]["parity_file"],
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

    last_signal_check: Dict[str, bool] = {}
    last_date = None

    signal_hours = config["strategy"]["signal_hours_mt5"]
    scan_minutes = config["strategy"]["scan_minutes_after_signal"]

    log.info(
        f"Bot corriendo. Señal base: vela {signal_hours}; "
        f"escaneo minutos {scan_minutes} MT5..."
    )

    try:
        while True:
            now = datetime.now()
            mt5_now = get_mt5_now(offset_hours=config["mt5"]["offset_hours"])

            # Nuevo día MT5
            if is_new_mt5_day(last_date, mt5_now):
                last_date = mt5_now.date()
                trades_today = reset_daily_trades(asset_params)
                last_signal_check.clear()

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
                    state_changed = True

                elif (
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
                    if not config["parity"]["force_scan"] and not should_run_for_asset(current_hour, params):
                        continue

                    key = build_signal_key(symbol, mt5_now, current_hour)
                    if key in last_signal_check:
                        continue

                    if symbol in open_positions:
                        log.info(f"{symbol}: evaluación omitida | reason=ya existe posición abierta en este símbolo")
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

                    sig = check_signal(
                        symbol=symbol,
                        asset_params=params,
                        signal_hour=current_hour,
                        mt5_now=mt5_now,
                        volume_means=volume_means,
                        initial_capital_per_asset=config["bot"]["initial_capital_per_asset"],
                    )

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
                        else:
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


                    else:

                        log.info("— Sin señal")

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
        shutdown_mt5()
        log.info("Bot finalizado.")


if __name__ == "__main__":
    main()