#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEMO EXECUTION SMOKE TEST — USTEC
====================================
ONE-SHOT infrastructure validation. Sends a single market BUY to MT5 DEMO.

PURPOSE
  Verify the full execution stack works end-to-end in demo:
    - MT5 connection
    - Symbol specs / lot normalization
    - Order send + fill (filling-mode negotiation)
    - Slippage / spread capture
    - PositionState construction
    - State persistence

THIS IS NOT A STRATEGY SIGNAL.
The order is artificial, sized at minimum lots, and clearly labelled.

SAFETY GUARDS (hard-coded — cannot be overridden by flags or config):
  1. Aborts if MT5 account trade_mode != 0 (not demo)
  2. Symbol is hard-coded to USTEC — no other symbol accepted
  3. One-shot: script exits after the single attempt
  4. Uses isolated state + log files separate from all bot environments
  5. Uses distinct magic number (20269999) — smoke test orders identifiable in MT5

USAGE
  cd C:\\Users\\Dasty\\PycharmProjects\\Trading_Lab
  python -m src.london_bot.demo_smoke_test           # prompts for confirmation
  python -m src.london_bot.demo_smoke_test --yes     # skips confirmation prompt

AFTER THE TEST
  - Verify open position in MT5 terminal (ticket logged)
  - Close manually in MT5 when done
  - Delete data/state/smoke_test_state.json when done
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# ── repo root on path regardless of CWD ───────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import MetaTrader5 as mt5

from src.shared.logger_setup  import setup_logger
from src.shared.mt5_connector import (
    connect_mt5,
    ensure_symbol_selected,
    get_account_info,
    get_spread_points,
    get_symbol_info,
    get_tick,
    shutdown_mt5,
)
from src.london_bot.order_manager_london import open_position
from src.london_bot.state_manager        import save_state

# ── Hard-coded constants ───────────────────────────────────────────────────────
SYMBOL        = "USTEC"          # smoke test is USTEC only — not configurable
DIRECTION     = 1                # BUY
SL_PCT        = 0.003            # 0.3% SL — matches locked baseline
MAGIC         = 20269999         # distinct magic — smoke test orders only
DEVIATION     = 30               # max slippage ticks

SIGNAL_TYPE   = "[SMOKE TEST] FORCED BUY — NOT A STRATEGY SIGNAL"
STATE_FILE    = str(_ROOT / "data"  / "state" / "smoke_test_state.json")
LOG_DIR       = str(_ROOT / "logs"  / "smoke_test")
MT5_TERM_PATH = "C:/Program Files/MetaTrader 5/terminal64.exe"   # adjust if needed


# ── Guard: abort helper ────────────────────────────────────────────────────────

def _abort(log, reason: str) -> None:
    log.error(f"╔══ SMOKE TEST ABORTED ═══════════════════════════════╗")
    log.error(f"  REASON: {reason}")
    log.error(f"╚════════════════════════════════════════════════════╝")
    shutdown_mt5()
    sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo execution smoke test — one-shot USTEC BUY in MT5 demo account"
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    log = setup_logger(
        module_name="smoke_test",
        log_dir=LOG_DIR,
        level="DEBUG",
    )

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  EXECUTION SMOKE TEST — USTEC — DEMO ACCOUNT ONLY   ║")
    log.info("║  THIS IS NOT A STRATEGY SIGNAL                       ║")
    log.info(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 13 + "║")
    log.info("╚══════════════════════════════════════════════════════╝")

    # ── Connect to MT5 ────────────────────────────────────────────────────────
    if not connect_mt5(logger=log, terminal_path=MT5_TERM_PATH):
        _abort(log, "MT5 connection failed")

    # ── GUARD 1: must be DEMO account ─────────────────────────────────────────
    acct = get_account_info()
    if acct is None:
        _abort(log, "get_account_info() returned None after connection")

    log.info(
        f"Account: login={acct.login} | name={acct.name} | "
        f"balance={acct.balance:.2f} | currency={acct.currency} | "
        f"trade_mode={acct.trade_mode}  "
        f"(DEMO={mt5.ACCOUNT_TRADE_MODE_DEMO} "
        f"CONTEST={mt5.ACCOUNT_TRADE_MODE_CONTEST} "
        f"REAL={mt5.ACCOUNT_TRADE_MODE_REAL})"
    )

    if acct.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        _abort(
            log,
            f"Account trade_mode={acct.trade_mode} — NOT demo "
            f"(ACCOUNT_TRADE_MODE_DEMO={mt5.ACCOUNT_TRADE_MODE_DEMO}). "
            "Smoke test only runs on demo accounts."
        )
    log.info("GUARD 1 ✓ — demo account confirmed")

    # ── GUARD 2: symbol must exist ────────────────────────────────────────────
    ensure_symbol_selected(SYMBOL)
    sym_info = get_symbol_info(SYMBOL)
    if sym_info is None:
        _abort(log, f"{SYMBOL} not found or not visible in MT5")

    digits   = int(sym_info.digits)
    ml       = float(sym_info.volume_min)       if sym_info.volume_min       else 0.1
    step     = float(sym_info.volume_step)      if sym_info.volume_step      else 0.1
    cs       = int(sym_info.trade_contract_size) if sym_info.trade_contract_size else 1

    log.info(
        f"GUARD 2 ✓ — {SYMBOL} found | digits={digits} | "
        f"ml={ml} | step={step} | cs={cs} | "
        f"stops_level={getattr(sym_info,'trade_stops_level',None)} | "
        f"freeze_level={getattr(sym_info,'trade_freeze_level',None)}"
    )

    # ── Build minimal asset_params from live specs ────────────────────────────
    asset_params = {
        "digits":     digits,
        "ml":         ml,
        "step":       step,
        "cs":         cs,
        "comm":       0.0,
        "jpy":        False,
        "trail_mult": 3.0,
        "sl_pct":     SL_PCT,
    }

    # ── Entry price + SL ──────────────────────────────────────────────────────
    tick = get_tick(SYMBOL)
    if tick is None:
        _abort(log, f"{SYMBOL}: no tick available — market may be closed")

    entry_price = float(tick.ask)   # BUY entry at ask
    spread_pts  = get_spread_points(SYMBOL)
    stop_loss   = round(entry_price * (1.0 - SL_PCT), digits)
    lots        = ml                 # minimum lots — always

    log.info(
        f"Order plan: BUY {lots}L {SYMBOL} @ ~{entry_price:.{digits}f} | "
        f"SL={stop_loss:.{digits}f} | spread={spread_pts:.{digits}f} | "
        f"magic={MAGIC}"
    )

    # ── Confirmation prompt ───────────────────────────────────────────────────
    if not args.yes:
        print(f"\n  ┌─ SMOKE TEST CONFIRMATION ────────────────────────────")
        print(f"  │  Account : {acct.login} ({acct.name}) — DEMO")
        print(f"  │  Symbol  : {SYMBOL}")
        print(f"  │  Order   : BUY {lots}L @ ~{entry_price:.{digits}f}")
        print(f"  │  SL      : {stop_loss:.{digits}f}  (0.3% below entry)")
        print(f"  │  Magic   : {MAGIC}  (smoke test identifier)")
        print(f"  └──────────────────────────────────────────────────────")
        ans = input("  Type 'yes' to send order, anything else to cancel: ").strip().lower()
        if ans != "yes":
            log.info("Smoke test cancelled by user at confirmation prompt.")
            shutdown_mt5()
            sys.exit(0)

    # ── Send order — reuses full open_position pipeline ──────────────────────
    # open_position handles: filling mode negotiation, order_check, order_send,
    # slippage tracking, PositionState construction, retcode validation.
    log.info(f"Sending order: {SIGNAL_TYPE}")

    pos = open_position(
        symbol=SYMBOL,
        direction=DIRECTION,
        lots=lots,
        sl=stop_loss,
        asset_params=asset_params,
        signal_price=entry_price,
        spread_signal=spread_pts,
        signal_type=SIGNAL_TYPE,
        bot_magic=MAGIC,
        deviation=DEVIATION,
        logger=log,
    )

    # ── GUARD 3: verify fill ──────────────────────────────────────────────────
    if pos is None:
        _abort(log, "open_position returned None — order rejected by broker. Check logs above.")

    log.info(
        f"╔══ ORDER FILLED ═══════════════════════════════════════╗\n"
        f"  Ticket     : {pos.ticket}\n"
        f"  Symbol     : {pos.symbol}\n"
        f"  Direction  : {'BUY' if pos.direction == 1 else 'SELL'}\n"
        f"  Lots       : {pos.lots}\n"
        f"  Fill price : {pos.real_entry_price:.{digits}f}\n"
        f"  SL         : {pos.sl:.{digits}f}\n"
        f"  Slippage   : {pos.slippage:+.{digits}f} pts\n"
        f"  Spread     : {pos.spread:.{digits}f}\n"
        f"  Exec time  : {pos.exec_ms} ms\n"
        f"  Signal type: {pos.stype}\n"
        f"╚═══════════════════════════════════════════════════════╝"
    )

    # ── Save state (isolated file — does NOT affect paper/demo bot state) ─────
    save_state(
        open_positions={SYMBOL: pos},
        trades_today={SYMBOL: 1},
        state_file=STATE_FILE,
        logger=log,
    )
    log.info(f"State saved: {STATE_FILE}")

    # ── Final instructions ────────────────────────────────────────────────────
    print(f"\n  ┌─ SMOKE TEST COMPLETE ─────────────────────────────────")
    print(f"  │  Ticket  : {pos.ticket}")
    print(f"  │  Fill    : {pos.real_entry_price:.{digits}f}")
    print(f"  │  SL      : {pos.sl:.{digits}f}")
    print(f"  │  Magic   : {MAGIC}  ← find in MT5 positions tab")
    print(f"  │")
    print(f"  │  Next steps:")
    print(f"  │    1. Verify open position in MT5 terminal (magic={MAGIC})")
    print(f"  │    2. Check logs at: {LOG_DIR}")
    print(f"  │    3. Check state at: {STATE_FILE}")
    print(f"  │    4. Close position manually in MT5 when done")
    print(f"  │    5. Delete {STATE_FILE} to clean up")
    print(f"  └──────────────────────────────────────────────────────")

    log.info("Smoke test finished. MT5 connection closing.")
    shutdown_mt5()


if __name__ == "__main__":
    main()
