#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 1 research runner:
clock integrity + fixed-time baselines only.

Scope
-----
- Session-based fixed-time holds.
- Long and short only.
- Explicit spread and commission costs.
- Honest ALL / TRAIN / TEST governance.
- Calendar-year and rolling-window diagnostics.

Non-goals
---------
- No conditioners.
- No breakout / ORB / trailing / stop logic.
- No parameter sweeps.
- No operational strategy claims.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKTEST_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(os.path.dirname(_BACKTEST_DIR))
if _BACKTEST_DIR not in sys.path:
    sys.path.insert(0, _BACKTEST_DIR)

from backtest_config import INITIAL_PER_ASSET, TEST_START  # noqa: E402
from backtest_data import load_price_data  # noqa: E402
from backtest_specs import resolve_asset_params  # noqa: E402
from research.session_clock import MT5_TZ_DEFAULT, label_sessions  # noqa: E402


DATA_DIR = os.path.join(_REPO_ROOT, "data", "backtesting")
REPORTS_ROOT = os.path.join(_REPO_ROOT, "reports", "research_phase1")
ROLL_MONTHS = 12
ROLL_STEP_MONTHS = 3


@dataclass(frozen=True)
class WindowSpec:
    key: str
    title: str
    start_hhmm: str
    end_hhmm: str
    start_min: int
    end_min: int


def _hhmm_to_min(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


WINDOW_SPECS: dict[str, WindowSpec] = {
    "ASIA_EU_CONC": WindowSpec(
        key="ASIA_EU_CONC",
        title="Asia-close / EU-open concentration",
        start_hhmm="07:00",
        end_hhmm="09:59",
        start_min=_hhmm_to_min("07:00"),
        end_min=_hhmm_to_min("09:59"),
    ),
    "EU_OPEN": WindowSpec(
        key="EU_OPEN",
        title="EU open",
        start_hhmm="08:00",
        end_hhmm="09:59",
        start_min=_hhmm_to_min("08:00"),
        end_min=_hhmm_to_min("09:59"),
    ),
    "US_OPEN": WindowSpec(
        key="US_OPEN",
        title="US open",
        start_hhmm="14:30",
        end_hhmm="15:59",
        start_min=_hhmm_to_min("14:30"),
        end_min=_hhmm_to_min("15:59"),
    ),
    "US_MID": WindowSpec(
        key="US_MID",
        title="US mid-session",
        start_hhmm="16:00",
        end_hhmm="19:59",
        start_min=_hhmm_to_min("16:00"),
        end_min=_hhmm_to_min("19:59"),
    ),
    "US_LATE": WindowSpec(
        key="US_LATE",
        title="US late-session",
        start_hhmm="20:00",
        end_hhmm="21:59",
        start_min=_hhmm_to_min("20:00"),
        end_min=_hhmm_to_min("21:59"),
    ),
}

PHASE1_UNIVERSE: dict[str, list[str]] = {
    "US500": ["EU_OPEN", "ASIA_EU_CONC", "US_MID", "US_LATE"],
    "USTEC": ["EU_OPEN", "ASIA_EU_CONC", "US_MID", "US_LATE"],
    "DE40": ["EU_OPEN", "US_OPEN", "US_MID"],
    "US30": ["US_OPEN", "US_MID"],
}

DIRECTIONS: dict[str, int] = {
    "LONG": 1,
    "SHORT": -1,
}


def _pct(x: float | int | np.floating | None) -> float:
    return round(float(x), 2) if pd.notna(x) else np.nan


def _metric(x: float | int | np.floating | None, digits: int = 4) -> float:
    return round(float(x), digits) if pd.notna(x) else np.nan


def _sample_bounds(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    sample: str,
) -> tuple[date | None, date | None]:
    test_start_date = TEST_START.normalize().date()
    start_date = data_start.date()
    end_date = data_end.date()

    if sample == "ALL":
        return start_date, end_date
    if sample == "TRAIN":
        train_end = min(end_date, (TEST_START - pd.Timedelta(days=1)).date())
        return start_date, train_end
    if sample == "TEST":
        test_start = max(start_date, test_start_date)
        return test_start, end_date
    raise ValueError(f"Unknown sample: {sample}")


def _subset_dates(
    df: pd.DataFrame,
    start_date: date | None,
    end_exclusive: date | None,
) -> pd.DataFrame:
    out = df
    if start_date is not None:
        out = out[out["trade_date"] >= start_date]
    if end_exclusive is not None:
        out = out[out["trade_date"] < end_exclusive]
    return out


def prepare_asset_df(asset: str, digits: int) -> tuple[pd.DataFrame, dict]:
    df, _, _, qc = load_price_data(asset, DATA_DIR, digits)
    if df is None:
        raise FileNotFoundError(f"No price data found for {asset}")

    df = label_sessions(df, MT5_TZ_DEFAULT)
    df["trade_date"] = df["time_utc"].dt.date
    df["utc_min"] = df["time_utc"].dt.hour * 60 + df["time_utc"].dt.minute
    return df, qc


def build_window_days(df: pd.DataFrame, window: WindowSpec) -> pd.DataFrame:
    sub = df[df["utc_min"].between(window.start_min, window.end_min)].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for trade_date, grp in sub.groupby("trade_date", sort=True):
        grp = grp.sort_values("time_utc")
        entry_row = grp.iloc[0]
        exit_row = grp.iloc[-1]
        rows.append(
            {
                "trade_date": trade_date,
                "entry_time_utc": entry_row["time_utc"],
                "exit_time_utc": exit_row["time_utc"],
                "entry_px": float(entry_row["open"]),
                "exit_px": float(exit_row["close"]),
                "entry_spread_px": (
                    float(entry_row["spread_px"])
                    if pd.notna(entry_row["spread_px"]) and float(entry_row["spread_px"]) > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def simulate_fixed_time_baseline(
    days_df: pd.DataFrame,
    direction_name: str,
    params: dict,
    cap_start: float,
    start_date: date | None = None,
    end_exclusive: date | None = None,
) -> pd.DataFrame:
    trades = _subset_dates(days_df, start_date, end_exclusive).copy()
    if trades.empty:
        return pd.DataFrame()

    direction = DIRECTIONS[direction_name]
    lots = float(params["ml"])
    cs = float(params["cs"])
    comm_rt = float(params["comm"])
    fallback_sp = float(params["sp"])
    is_jpy = bool(params.get("jpy", False))

    spread_px = trades["entry_spread_px"].where(
        trades["entry_spread_px"].notna() & (trades["entry_spread_px"] > 0),
        fallback_sp,
    )
    avg_px = (trades["entry_px"] + trades["exit_px"]) / 2.0
    multiplier = np.where(is_jpy, cs / np.maximum(avg_px, 1.0), cs)

    gross = (trades["exit_px"] - trades["entry_px"]) * direction * lots * multiplier
    spread_cost = spread_px * lots * cs
    commission = lots * comm_rt
    net = gross - spread_cost - commission
    equity = cap_start + np.cumsum(net)
    result = np.where(net > 0.01, "WIN", np.where(net < -0.01, "LOSS", "BE"))

    trades["direction"] = direction_name
    trades["lots"] = lots
    trades["gross_usd"] = gross
    trades["spread_cost_usd"] = spread_cost
    trades["commission_usd"] = commission
    trades["net_usd"] = net
    trades["result"] = result
    trades["cap"] = equity
    return trades


def compute_metrics(trades: pd.DataFrame, cap_start: float) -> dict:
    if trades.empty:
        return {
            "n": 0,
            "wr": np.nan,
            "pf": np.nan,
            "exp": np.nan,
            "ret_pct": np.nan,
            "mdd": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
        }

    wins = trades.loc[trades["result"] == "WIN", "net_usd"]
    losses = trades.loc[trades["result"] == "LOSS", "net_usd"]
    gross_w = float(wins.sum())
    gross_l = abs(float(losses.sum()))
    pf = gross_w / gross_l if gross_l > 0 else np.nan

    caps = np.array([cap_start] + trades["cap"].tolist(), dtype=float)
    peaks = np.maximum.accumulate(caps)
    dd = (caps - peaks) / np.where(peaks > 0, peaks, 1.0)

    return {
        "n": int(len(trades)),
        "wr": _pct((trades["result"] == "WIN").mean() * 100),
        "pf": _metric(pf, 3),
        "exp": _metric(trades["net_usd"].mean(), 4),
        "ret_pct": _pct(trades["net_usd"].sum() / cap_start * 100),
        "mdd": _pct(dd.min() * 100),
        "avg_win": _metric(wins.mean(), 4),
        "avg_loss": _metric(losses.mean(), 4),
    }


def iter_calendar_years(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
) -> list[dict]:
    out: list[dict] = []
    for year in range(data_start.year, data_end.year + 1):
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")
        actual_start = max(year_start, data_start)
        actual_end = min(year_end, data_end)
        is_partial = actual_start > year_start or actual_end < year_end
        out.append(
            {
                "label": f"{year}{' [PARTIAL]' if is_partial else ''}",
                "year": year,
                "start_date": actual_start.date(),
                "end_date": actual_end.date(),
                "is_partial": is_partial,
                "bucket": (
                    "TRAIN"
                    if actual_end.date() < TEST_START.date()
                    else ("TEST" if actual_start.date() >= TEST_START.date() else "CROSS")
                ),
            }
        )
    return out


def iter_rolling_windows(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    months_len: int = ROLL_MONTHS,
    months_step: int = ROLL_STEP_MONTHS,
) -> list[dict]:
    windows: list[dict] = []
    start = data_start.to_period("M").to_timestamp()
    hard_end = data_end.to_period("M").to_timestamp() + pd.offsets.MonthEnd(0)

    while start <= hard_end:
        requested_end = start + pd.DateOffset(months=months_len) - pd.Timedelta(days=1)
        actual_start = max(start, data_start)
        actual_end = min(requested_end, data_end)
        if actual_start > actual_end:
            break
        is_partial = actual_start > start or actual_end < requested_end
        label = (
            f"{start.strftime('%Y-%m')}->{requested_end.strftime('%Y-%m')}"
            f"{' [PARTIAL]' if is_partial else ''}"
        )
        if actual_end.date() < TEST_START.date():
            bucket = "TRAIN"
        elif actual_start.date() >= TEST_START.date():
            bucket = "TEST"
        else:
            bucket = "CROSS"
        windows.append(
            {
                "label": label,
                "requested_start": start.date(),
                "requested_end": requested_end.date(),
                "start_date": actual_start.date(),
                "end_date": actual_end.date(),
                "is_partial": is_partial,
                "bucket": bucket,
            }
        )
        start = start + pd.DateOffset(months=months_step)
    return windows


def _print_table(title: str, df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> None:
    print()
    print(title)
    if df.empty:
        print("  (empty)")
        return
    print(df[columns].head(max_rows).to_string(index=False))


def main() -> None:
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPORTS_ROOT, f"run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    params_by_asset = resolve_asset_params(DATA_DIR)
    summary_rows: list[dict] = []
    yearly_rows: list[dict] = []
    rolling_rows: list[dict] = []

    print("PHASE 1 RESEARCH - SESSION BASELINES V2")
    print(f"MT5 timezone : {MT5_TZ_DEFAULT}")
    print(f"ALL/TRAIN/TEST split : TEST starts {TEST_START.date()}")
    print(f"Output dir : {run_dir}")

    for asset, window_keys in PHASE1_UNIVERSE.items():
        if asset not in params_by_asset:
            print(f"  [skip] {asset}: asset params not resolved")
            continue

        params = params_by_asset[asset]
        df, _qc = prepare_asset_df(asset, int(params["digits"]))
        asset_trade_start = pd.Timestamp(df["trade_date"].min())
        asset_trade_end = pd.Timestamp(df["trade_date"].max())

        print()
        print(f"ASSET {asset} | trade dates {asset_trade_start.date()} -> {asset_trade_end.date()} | "
              f"comm_rt={params['comm']:.2f} | spread_fallback={params['sp']}")

        for window_key in window_keys:
            window = WINDOW_SPECS[window_key]
            days_df = build_window_days(df, window)
            if days_df.empty:
                continue
            data_start = pd.Timestamp(days_df["trade_date"].min())
            data_end = pd.Timestamp(days_df["trade_date"].max())

            for direction_name in DIRECTIONS:
                all_trades = simulate_fixed_time_baseline(
                    days_df,
                    direction_name=direction_name,
                    params=params,
                    cap_start=INITIAL_PER_ASSET,
                )

                for sample in ["ALL", "TRAIN", "TEST"]:
                    sample_start, sample_end = _sample_bounds(data_start, data_end, sample)
                    sample_end_exclusive = (
                        sample_end + timedelta(days=1) if sample_end is not None else None
                    )
                    trades = simulate_fixed_time_baseline(
                        days_df,
                        direction_name=direction_name,
                        params=params,
                        cap_start=INITIAL_PER_ASSET,
                        start_date=sample_start,
                        end_exclusive=sample_end_exclusive,
                    )
                    metrics = compute_metrics(trades, INITIAL_PER_ASSET)
                    summary_rows.append(
                        {
                            "asset": asset,
                            "family": window_key,
                            "family_title": window.title,
                            "direction": direction_name,
                            "sample": sample,
                            "window_start_utc": window.start_hhmm,
                            "window_end_utc": window.end_hhmm,
                            "sample_start": sample_start,
                            "sample_end": sample_end,
                            "n": metrics["n"],
                            "wr": metrics["wr"],
                            "pf": metrics["pf"],
                            "exp": metrics["exp"],
                            "ret_pct": metrics["ret_pct"],
                            "mdd": metrics["mdd"],
                            "avg_win": metrics["avg_win"],
                            "avg_loss": metrics["avg_loss"],
                            "lots": float(params["ml"]),
                            "spread_fallback_pts": float(params["sp"]),
                            "comm_rt": float(params["comm"]),
                            "cost_model": "entry_spread_from_bar_else_fallback + roundtrip_commission",
                        }
                    )

                for yr in iter_calendar_years(data_start, data_end):
                    trades = simulate_fixed_time_baseline(
                        days_df,
                        direction_name=direction_name,
                        params=params,
                        cap_start=INITIAL_PER_ASSET,
                        start_date=yr["start_date"],
                        end_exclusive=yr["end_date"] + timedelta(days=1),
                    )
                    metrics = compute_metrics(trades, INITIAL_PER_ASSET)
                    yearly_rows.append(
                        {
                            "asset": asset,
                            "family": window_key,
                            "family_title": window.title,
                            "direction": direction_name,
                            "year": yr["year"],
                            "label": yr["label"],
                            "is_partial": yr["is_partial"],
                            "bucket": yr["bucket"],
                            "sample_start": yr["start_date"],
                            "sample_end": yr["end_date"],
                            **metrics,
                        }
                    )

                for rw in iter_rolling_windows(data_start, data_end):
                    trades = simulate_fixed_time_baseline(
                        days_df,
                        direction_name=direction_name,
                        params=params,
                        cap_start=INITIAL_PER_ASSET,
                        start_date=rw["start_date"],
                        end_exclusive=rw["end_date"] + timedelta(days=1),
                    )
                    metrics = compute_metrics(trades, INITIAL_PER_ASSET)
                    rolling_rows.append(
                        {
                            "asset": asset,
                            "family": window_key,
                            "family_title": window.title,
                            "direction": direction_name,
                            "label": rw["label"],
                            "is_partial": rw["is_partial"],
                            "bucket": rw["bucket"],
                            "requested_start": rw["requested_start"],
                            "requested_end": rw["requested_end"],
                            "sample_start": rw["start_date"],
                            "sample_end": rw["end_date"],
                            **metrics,
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    yearly_df = pd.DataFrame(yearly_rows)
    rolling_df = pd.DataFrame(rolling_rows)

    pivot = summary_df.pivot_table(
        index=["asset", "family", "family_title", "direction", "window_start_utc", "window_end_utc"],
        columns="sample",
        values=["n", "wr", "pf", "exp", "ret_pct", "mdd", "avg_win", "avg_loss"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{sample}" for metric, sample in pivot.columns]
    ranking_df = pivot.reset_index()

    yearly_full = yearly_df[~yearly_df["is_partial"]].copy()
    if not yearly_full.empty:
        yearly_stats = (
            yearly_full.groupby(["asset", "family", "family_title", "direction"], dropna=False)
            .agg(
                full_years=("year", "count"),
                full_years_pf_gt1=("pf", lambda s: int((s > 1.0).sum())),
                full_years_exp_pos=("exp", lambda s: int((s > 0).sum())),
                full_years_nonneg_ret=("ret_pct", lambda s: int((s >= 0).sum())),
            )
            .reset_index()
        )
        ranking_df = ranking_df.merge(
            yearly_stats,
            on=["asset", "family", "family_title", "direction"],
            how="left",
        )

    rolling_full = rolling_df[~rolling_df["is_partial"]].copy()
    if not rolling_full.empty:
        rolling_stats = (
            rolling_full.groupby(["asset", "family", "family_title", "direction"], dropna=False)
            .agg(
                rolling_windows=("label", "count"),
                rolling_pf_gt1=("pf", lambda s: int((s > 1.0).sum())),
                rolling_exp_pos=("exp", lambda s: int((s > 0).sum())),
            )
            .reset_index()
        )
        ranking_df = ranking_df.merge(
            rolling_stats,
            on=["asset", "family", "family_title", "direction"],
            how="left",
        )

    ranking_df = ranking_df.sort_values(
        by=["pf_TEST", "exp_TEST", "n_TEST"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    summary_path = os.path.join(run_dir, "phase1_summary.csv")
    yearly_path = os.path.join(run_dir, "phase1_yearly.csv")
    rolling_path = os.path.join(run_dir, "phase1_rolling.csv")
    ranking_path = os.path.join(run_dir, "phase1_ranking.csv")
    config_path = os.path.join(run_dir, "phase1_config.json")

    summary_df.to_csv(summary_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)
    rolling_df.to_csv(rolling_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": "Phase 1 - clock integrity + fixed-time baselines",
                "mt5_timezone": MT5_TZ_DEFAULT,
                "data_dir": DATA_DIR,
                "test_start": str(TEST_START.date()),
                "all_definition": "all available history",
                "train_definition": "trade_date < TEST_START",
                "test_definition": "trade_date >= TEST_START",
                "rolling_months": ROLL_MONTHS,
                "rolling_step_months": ROLL_STEP_MONTHS,
                "capital_mode": "fixed minimum-lot research baseline referenced to INITIAL_PER_ASSET; ret_pct and MDD may exceed -100 if cumulative PnL does",
                "windows": {
                    k: {
                        "title": v.title,
                        "start_utc": v.start_hhmm,
                        "end_utc": v.end_hhmm,
                    }
                    for k, v in WINDOW_SPECS.items()
                },
                "universe": PHASE1_UNIVERSE,
                "directions": list(DIRECTIONS.keys()),
                "cost_model": "gross PnL minus entry spread cost (dynamic bar spread if available else fallback spread) minus roundtrip commission",
            },
            f,
            indent=2,
        )

    _print_table(
        "SUMMARY (sample rows)",
        summary_df.sort_values(["asset", "family", "direction", "sample"]),
        [
            "asset", "family", "direction", "sample",
            "n", "wr", "pf", "exp", "ret_pct", "mdd",
        ],
        max_rows=40,
    )
    _print_table(
        "RANKING (sorted by TEST PF / expectancy / sample size)",
        ranking_df,
        [
            "asset", "family", "direction",
            "n_TEST", "pf_TEST", "exp_TEST", "ret_pct_TEST",
            "pf_TRAIN", "exp_TRAIN",
            "full_years", "full_years_pf_gt1",
            "rolling_windows", "rolling_pf_gt1",
        ],
        max_rows=30,
    )

    print()
    print("OUTPUTS")
    print(f"  {summary_path}")
    print(f"  {yearly_path}")
    print(f"  {rolling_path}")
    print(f"  {ranking_path}")
    print(f"  {config_path}")
    print("  Note: fixed minimum-lot baseline; ret_pct and MDD are referenced to cap0 and are not operational sizing metrics.")


if __name__ == "__main__":
    main()
