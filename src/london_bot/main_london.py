from __future__ import annotations

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


def main() -> None:
    config = load_app_config("london_bot")

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
            if last_date != mt5_now.date():
                last_date = mt5_now.date()
                trades_today = {symbol: 0 for symbol in asset_params}
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
                digits = pos.get("digits", 2)
                sl_before = pos.get("sl", 0.0)
                be_before = pos.get("breakeven_hit", False)
                bp_before = pos.get("best_price", 0.0)

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
                    open_positions.pop(symbol)
                    state_changed = True

                elif (
                    pos.get("sl", 0.0) != sl_before
                    or pos.get("breakeven_hit", False) != be_before
                    or abs(pos.get("best_price", 0.0) - bp_before) > 10 ** (-digits)
                ):
                    state_changed = True

            if state_changed:
                save_state(
                    open_positions=open_positions,
                    trades_today=trades_today,
                    state_file=config["paths"]["state_file"],
                    logger=log,
                )

            current_hour = mt5_now.hour
            current_minute = mt5_now.minute

            # Escaneo de señales
            if current_minute in scan_minutes and current_hour in signal_hours:
                for symbol, params in asset_params.items():
                    if current_hour not in params["hours"]:
                        continue

                    key = f"{symbol}_{mt5_now.date()}_{current_hour}"
                    if key in last_signal_check:
                        continue

                    if symbol in open_positions:
                        last_signal_check[key] = True
                        continue

                    log.info(
                        f"🔍 Revisando vela {current_hour}:00 MT5 "
                        f"(scan {mt5_now.strftime('%H:%M')}) | "
                        f"capital=${config['bot']['initial_capital_per_asset']} | "
                        f"symbol={symbol}"
                    )

                    sig = check_signal(
                        symbol=symbol,
                        asset_params=params,
                        trades_today=trades_today.get(symbol, 0),
                        signal_hour=current_hour,
                        mt5_now=mt5_now,
                        volume_means=volume_means,
                        initial_capital_per_asset=config["bot"]["initial_capital_per_asset"],
                        max_trades_per_day_per_asset=config["bot"]["max_trades_per_day_per_asset"],
                    )

                    if sig:
                        log.info(
                            f"⚡ {sig['stype']} | "
                            f"lots={sig['lots']} | "
                            f"sl={sig['sl']:.{params['digits']}f} | "
                            f"lrr={sig['lrr']:.2f} | "
                            f"spread_signal={sig['spread_signal']:.{params['digits']}f} | "
                            f"tv={sig['signal_tick_volume']:.1f} vs "
                            f"vm={volume_means.get(symbol, 0.0):.1f}"
                        )

                        if not config["mode"]["execution_enabled"]:
                            log.info(
                                f"[SHADOW] Señal detectada pero execution_enabled=False | {symbol}"
                            )
                        else:
                            pos = open_position(
                                symbol=sig["symbol"],
                                direction=sig["direction"],
                                lots=sig["lots"],
                                sl=sig["sl"],
                                asset_params=params,
                                signal_price=sig["ep"],
                                spread_signal=sig["spread_signal"],
                                signal_type=sig["stype"],
                                bot_magic=config["bot"]["magic"],
                                deviation=config["execution"]["deviation"],
                                logger=log,
                            )

                            if pos:
                                pos["atr"] = sig["atr"]
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

                    last_signal_check[key] = True

            # Heartbeat horario
            if current_minute == 0 and now.second < 30:
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