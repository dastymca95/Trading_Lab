from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# ── Column schemas ─────────────────────────────────────────────────────────────

SIGNAL_AUDIT_COLUMNS = [
    "ts_local",
    "ts_mt5",
    "env",
    "symbol",
    "timeframe",
    "scan_hour",
    "scan_minute",
    "allowed_hour",
    "allowed_dow",
    "daily_filter_loaded",
    "daily_filter_pass",
    "volume_filter_pass",
    "spread_points",
    "force_direction_applied",
    "raw_signal_detected",
    "final_signal_direction",
    "state_has_open_position",
    "execution_enabled",
    "action_taken",
    "block_reason",
    "notes",
]

SESSION_SUMMARY_COLUMNS = [
    "date",
    "symbol",
    "total_checks",
    "in_window_checks",
    "out_of_window_checks",
    "daily_filter_blocks",
    "volume_filter_blocks",
    "no_signal_count",
    "signal_count",
    "already_open_blocks",
    "execution_disabled_blocks",
    "would_send_order_count",
    "runtime_error_count",
]

# ── action_taken values ────────────────────────────────────────────────────────
#   OUT_OF_WINDOW        – bot running, scan window inactive (logged at heartbeat)
#   EVAL_BLOCKED         – can_evaluate_signal denied (hour/dow/max_trades)
#   POSITION_ALREADY_OPEN – symbol has open position, evaluation skipped
#   NO_SIGNAL            – evaluation ran, check_signal returned None
#   SIGNAL_ONLY          – signal found, execution_enabled=False (paper shadow)
#   ORDER_SENT           – signal found, order submitted to MT5
#   ORDER_BLOCKED        – signal found, exec check failed for other reason
#   RUNTIME_ERROR        – exception during signal evaluation

# ── block_reason ───────────────────────────────────────────────────────────────
#   ""                   – no block (ORDER_SENT, SIGNAL_ONLY, OUT_OF_WINDOW)
#   out_of_window        – scan window inactive
#   out_of_hour          – current hour not in asset.hours
#   out_of_dow           – weekday not allowed
#   max_trades_reached   – daily trade cap hit
#   already_in_position  – open position exists
#   daily_filter_fail    – regime filter rejected today
#   volume_filter_fail   – tick_volume below historical mean
#   lrr_fail             – LRR below threshold
#   atr_zero             – ATR unavailable
#   context_unavailable  – no candle context from MT5
#   no_direction         – no breakout / large-candle detected
#   stop_distance_zero   – SL distance collapsed to zero
#   lots_zero            – position sizing returned 0
#   execution_disabled   – paper mode, execution_enabled=False
#   runtime_error        – exception during evaluation


def ensure_signal_audit_csv(audit_file: str) -> None:
    """Creates the signal audit CSV with headers if it does not exist."""
    path = Path(audit_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(SIGNAL_AUDIT_COLUMNS)


def write_signal_audit_row(
    audit_file: str,
    ts_local: datetime,
    ts_mt5: datetime,
    env: str,
    symbol: str,
    scan_hour: int,
    scan_minute: int,
    allowed_hour: bool,
    allowed_dow: bool,
    daily_filter_loaded: bool,
    daily_filter_pass: bool,
    volume_filter_pass: Optional[bool],   # None = not evaluated (eval stopped earlier)
    spread_points: Optional[float],        # None = tick unavailable
    force_direction_applied: bool,
    raw_signal_detected: Optional[bool],  # None = not evaluated (eval stopped before direction check)
    final_signal_direction: Optional[int],
    state_has_open_position: bool,
    execution_enabled: bool,
    action_taken: str,
    block_reason: str,
    notes: str = "",
    logger=None,
) -> None:
    """
    Appends one row to the signal audit CSV.
    Silent on write errors — audit failure must never crash the trading loop.
    """
    try:
        path = Path(audit_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts_local":                ts_local.isoformat(),
            "ts_mt5":                  ts_mt5.isoformat(),
            "env":                     env,
            "symbol":                  symbol,
            "timeframe":               "M2",
            "scan_hour":               scan_hour,
            "scan_minute":             scan_minute,
            "allowed_hour":            int(allowed_hour),
            "allowed_dow":             int(allowed_dow),
            "daily_filter_loaded":     int(daily_filter_loaded),
            "daily_filter_pass":       int(daily_filter_pass),
            "volume_filter_pass":      "" if volume_filter_pass is None else int(volume_filter_pass),
            "spread_points":           "" if spread_points is None else f"{spread_points:.5f}",
            "force_direction_applied": int(force_direction_applied),
            "raw_signal_detected":     "" if raw_signal_detected is None else int(raw_signal_detected),
            "final_signal_direction":  "" if final_signal_direction is None else final_signal_direction,
            "state_has_open_position": int(state_has_open_position),
            "execution_enabled":       int(execution_enabled),
            "action_taken":            action_taken,
            "block_reason":            block_reason,
            "notes":                   notes,
        }
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SIGNAL_AUDIT_COLUMNS).writerow(row)
    except Exception as exc:
        if logger is not None:
            logger.warning(f"signal audit append failed | file={audit_file} | error={exc}")


def compute_and_write_session_summary(
    audit_file: str,
    summary_file: str,
    for_date: date,
    logger,
) -> None:
    """
    Reads the signal audit CSV, computes per-symbol daily statistics for
    for_date, and appends rows to the session summary CSV.

    Called at day-rollover and at shutdown so every trading day gets a summary.
    """
    try:
        import pandas as pd

        audit_path = Path(audit_file)
        if not audit_path.exists():
            return

        df = pd.read_csv(audit_path)
        if df.empty:
            return

        df["_date"] = pd.to_datetime(df["ts_mt5"], errors="coerce").dt.date
        day_df = df[df["_date"] == for_date].copy()

        if day_df.empty:
            logger.info(f"Session summary: no audit rows for {for_date}")
            return

        # Reasons that represent a technical "no signal" (not blocked by filters)
        _no_sig_reasons = {
            "context_unavailable", "atr_zero", "lrr_fail",
            "no_direction", "stop_distance_zero", "lots_zero",
        }

        rows = []
        for symbol, grp in day_df.groupby("symbol"):
            ops = grp[grp["action_taken"] != "RUNTIME_EVENT"]
            rows.append({
                "date":                      for_date.isoformat(),
                "symbol":                    symbol,
                "total_checks":              len(ops),
                "in_window_checks":          int((ops["action_taken"] != "OUT_OF_WINDOW").sum()),
                "out_of_window_checks":      int((ops["action_taken"] == "OUT_OF_WINDOW").sum()),
                "daily_filter_blocks":       int((ops["block_reason"] == "daily_filter_fail").sum()),
                "volume_filter_blocks":      int((ops["block_reason"] == "volume_filter_fail").sum()),
                "no_signal_count":           int(ops["block_reason"].isin(_no_sig_reasons).sum()),
                "signal_count":              int(ops["action_taken"].isin(
                                                 ["SIGNAL_ONLY", "ORDER_SENT", "ORDER_BLOCKED"]).sum()),
                "already_open_blocks":       int((ops["block_reason"] == "already_in_position").sum()),
                "execution_disabled_blocks": int((ops["block_reason"] == "execution_disabled").sum()),
                "would_send_order_count":    int((ops["action_taken"] == "SIGNAL_ONLY").sum()),
                "runtime_error_count":       int((ops["block_reason"] == "runtime_error").sum()),
            })

        summary_path = Path(summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        new_rows = pd.DataFrame(rows, columns=SESSION_SUMMARY_COLUMNS)

        if summary_path.exists():
            existing = pd.read_csv(summary_path)
            if not existing.empty:
                existing["date"] = existing["date"].astype(str)
                existing["symbol"] = existing["symbol"].astype(str)
                new_keys = set(zip(new_rows["date"], new_rows["symbol"]))
                existing = existing[
                    ~existing.apply(lambda r: (str(r["date"]), str(r["symbol"])) in new_keys, axis=1)
                ]
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows

        combined.to_csv(summary_path, index=False)
        logger.info(f"Session summary written | file={summary_file} | date={for_date}")

        for r in rows:
            logger.info(
                f"📊 SESSION [{r['date']}] {r['symbol']} | "
                f"in_window={r['in_window_checks']} "
                f"signal={r['signal_count']} "
                f"would_order={r['would_send_order_count']} "
                f"df_block={r['daily_filter_blocks']} "
                f"vol_block={r['volume_filter_blocks']} "
                f"errors={r['runtime_error_count']}"
            )

    except Exception as e:
        logger.error(f"compute_session_summary failed: {e}")
