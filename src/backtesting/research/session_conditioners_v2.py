#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 research runner:
one-conditioner session research over the Phase 1 shortlist.

Scope
-----
- Shortlisted fixed-time baselines only.
- Baseline reference + exactly one conditioner at a time.
- FOMC day suppression.
- Single causal volatility gate.
- Honest ALL / TRAIN / TEST, yearly, and rolling diagnostics.

Non-goals
---------
- No breakout / ORB / operational runner translation.
- No stacked conditioners.
- No parameter sweeps.
- No new alpha families.
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
from backtest_specs import resolve_asset_params  # noqa: E402
from research.research_runner import _FOMC_DATES  # noqa: E402
from research.session_baselines_v2 import (  # noqa: E402
    DATA_DIR,
    ROLL_MONTHS,
    ROLL_STEP_MONTHS,
    WINDOW_SPECS,
    build_window_days,
    compute_metrics,
    iter_calendar_years,
    iter_rolling_windows,
    prepare_asset_df,
    simulate_fixed_time_baseline,
)
from research.session_clock import MT5_TZ_DEFAULT  # noqa: E402


REPORTS_ROOT = os.path.join(_REPO_ROOT, "reports", "research_phase2")
VOL_WINDOW_DAYS = 90
VOL_PERCENTILE = 50
MIN_RETENTION_WARN = 50.0


@dataclass(frozen=True)
class Candidate:
    tier: str
    asset: str
    family: str
    direction: str


SHORTLIST: list[Candidate] = [
    Candidate("TIER1", "DE40", "US_MID", "LONG"),
    Candidate("TIER1", "US30", "US_OPEN", "LONG"),
    Candidate("TIER2", "DE40", "US_OPEN", "LONG"),
    Candidate("TIER2", "DE40", "EU_OPEN", "SHORT"),
    Candidate("TIER2", "USTEC", "ASIA_EU_CONC", "LONG"),
    Candidate("BORDERLINE", "USTEC", "US_MID", "LONG"),
    Candidate("BORDERLINE", "US500", "US_MID", "LONG"),
]


def build_causal_low_vol_dates(
    df: pd.DataFrame,
    window_days: int = VOL_WINDOW_DAYS,
    percentile: int = VOL_PERCENTILE,
) -> frozenset[date]:
    """
    Causal volatility gate.

    Trade day D is eligible only if the last fully completed day ATR (D-1)
    is <= rolling percentile threshold computed over prior completed days.
    """
    daily_atr = df.groupby("trade_date")["atr14"].mean().sort_index()
    atr_ref = daily_atr.shift(1)
    rolling_thr = (
        atr_ref
        .rolling(window_days, min_periods=window_days // 2)
        .quantile(percentile / 100.0)
    )
    return frozenset(
        d for d in daily_atr.index
        if pd.notna(atr_ref.loc[d])
        and pd.notna(rolling_thr.loc[d])
        and atr_ref.loc[d] <= rolling_thr.loc[d]
    )


def apply_conditioner(
    days_df: pd.DataFrame,
    conditioner: str,
    low_vol_dates: frozenset[date],
) -> pd.DataFrame:
    if conditioner == "BASELINE":
        return days_df
    if conditioner == "NO_FOMC":
        return days_df[~days_df["trade_date"].isin(_FOMC_DATES)]
    if conditioner == "VOL_LOW_P50W90":
        return days_df[days_df["trade_date"].isin(low_vol_dates)]
    raise ValueError(f"Unknown conditioner: {conditioner}")


def _sample_bounds(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    sample: str,
) -> tuple[date, date]:
    if sample == "ALL":
        return data_start.date(), data_end.date()
    if sample == "TRAIN":
        return data_start.date(), min(data_end.date(), (TEST_START - pd.Timedelta(days=1)).date())
    if sample == "TEST":
        return max(data_start.date(), TEST_START.date()), data_end.date()
    raise ValueError(f"Unknown sample: {sample}")


def _simulate_slice(
    days_df: pd.DataFrame,
    candidate: Candidate,
    params: dict,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    end_exclusive = end_date + timedelta(days=1) if end_date is not None else None
    return simulate_fixed_time_baseline(
        days_df,
        direction_name=candidate.direction,
        params=params,
        cap_start=INITIAL_PER_ASSET,
        start_date=start_date,
        end_exclusive=end_exclusive,
    )


def _row_prefix(candidate: Candidate, conditioner: str, window_title: str) -> dict:
    return {
        "tier": candidate.tier,
        "asset": candidate.asset,
        "family": candidate.family,
        "family_title": window_title,
        "direction": candidate.direction,
        "conditioner": conditioner,
    }


def _metrics_with_retention(
    conditioned_trades: pd.DataFrame,
    baseline_n: int,
) -> dict:
    metrics = compute_metrics(conditioned_trades, INITIAL_PER_ASSET)
    metrics["retention_pct"] = (
        round(metrics["n"] / baseline_n * 100, 2)
        if baseline_n > 0
        else np.nan
    )
    metrics["retention_warning"] = (
        bool(pd.notna(metrics["retention_pct"]) and metrics["retention_pct"] < MIN_RETENTION_WARN)
    )
    return metrics


def _add_delta_columns(ranking_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "tier", "asset", "family", "family_title", "direction",
        "n_TEST", "pf_TEST", "exp_TEST", "ret_pct_TEST", "mdd_TEST",
    ]
    baseline = (
        ranking_df[ranking_df["conditioner"] == "BASELINE"][base_cols]
        .rename(
            columns={
                "n_TEST": "baseline_n_TEST",
                "pf_TEST": "baseline_pf_TEST",
                "exp_TEST": "baseline_exp_TEST",
                "ret_pct_TEST": "baseline_ret_pct_TEST",
                "mdd_TEST": "baseline_mdd_TEST",
            }
        )
    )
    out = ranking_df.merge(
        baseline,
        on=["tier", "asset", "family", "family_title", "direction"],
        how="left",
    )
    out["delta_pf_TEST"] = (out["pf_TEST"] - out["baseline_pf_TEST"]).round(3)
    out["delta_exp_TEST"] = (out["exp_TEST"] - out["baseline_exp_TEST"]).round(4)
    out["delta_mdd_TEST"] = (out["mdd_TEST"] - out["baseline_mdd_TEST"]).round(2)
    return out


def _print_table(title: str, df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> None:
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
    asset_cache: dict[str, tuple[pd.DataFrame, dict]] = {}

    summary_rows: list[dict] = []
    yearly_rows: list[dict] = []
    rolling_rows: list[dict] = []

    print("PHASE 2 RESEARCH - ONE-CONDITIONER SESSION RESEARCH")
    print(f"MT5 timezone : {MT5_TZ_DEFAULT}")
    print(f"ALL/TRAIN/TEST split : TEST starts {TEST_START.date()}")
    print("Conditioners : BASELINE | NO_FOMC | VOL_LOW_P50W90")
    print(f"Output dir : {run_dir}")

    for candidate in SHORTLIST:
        if candidate.asset not in params_by_asset:
            print(f"  [skip] {candidate.asset}: asset params not resolved")
            continue

        params = params_by_asset[candidate.asset]
        if candidate.asset not in asset_cache:
            asset_cache[candidate.asset] = prepare_asset_df(candidate.asset, int(params["digits"]))
        df, _qc = asset_cache[candidate.asset]

        window = WINDOW_SPECS[candidate.family]
        days_df = build_window_days(df, window)
        if days_df.empty:
            print(f"  [skip] {candidate.asset} {candidate.family}: no window days")
            continue

        data_start = pd.Timestamp(days_df["trade_date"].min())
        data_end = pd.Timestamp(days_df["trade_date"].max())
        low_vol_dates = build_causal_low_vol_dates(df)

        print()
        print(
            f"{candidate.tier} | {candidate.asset} | {candidate.family} | "
            f"{candidate.direction} | {window.start_hhmm}-{window.end_hhmm} UTC"
        )

        baseline_days = apply_conditioner(days_df, "BASELINE", low_vol_dates)

        for conditioner in ["BASELINE", "NO_FOMC", "VOL_LOW_P50W90"]:
            conditioned_days = apply_conditioner(days_df, conditioner, low_vol_dates)

            for sample in ["ALL", "TRAIN", "TEST"]:
                sample_start, sample_end = _sample_bounds(data_start, data_end, sample)
                baseline_trades = _simulate_slice(
                    baseline_days,
                    candidate,
                    params,
                    sample_start,
                    sample_end,
                )
                conditioned_trades = _simulate_slice(
                    conditioned_days,
                    candidate,
                    params,
                    sample_start,
                    sample_end,
                )
                metrics = _metrics_with_retention(conditioned_trades, len(baseline_trades))
                summary_rows.append(
                    {
                        **_row_prefix(candidate, conditioner, window.title),
                        "sample": sample,
                        "window_start_utc": window.start_hhmm,
                        "window_end_utc": window.end_hhmm,
                        "sample_start": sample_start,
                        "sample_end": sample_end,
                        **metrics,
                    }
                )

            for yr in iter_calendar_years(data_start, data_end):
                baseline_trades = _simulate_slice(
                    baseline_days,
                    candidate,
                    params,
                    yr["start_date"],
                    yr["end_date"],
                )
                conditioned_trades = _simulate_slice(
                    conditioned_days,
                    candidate,
                    params,
                    yr["start_date"],
                    yr["end_date"],
                )
                metrics = _metrics_with_retention(conditioned_trades, len(baseline_trades))
                yearly_rows.append(
                    {
                        **_row_prefix(candidate, conditioner, window.title),
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
                baseline_trades = _simulate_slice(
                    baseline_days,
                    candidate,
                    params,
                    rw["start_date"],
                    rw["end_date"],
                )
                conditioned_trades = _simulate_slice(
                    conditioned_days,
                    candidate,
                    params,
                    rw["start_date"],
                    rw["end_date"],
                )
                metrics = _metrics_with_retention(conditioned_trades, len(baseline_trades))
                rolling_rows.append(
                    {
                        **_row_prefix(candidate, conditioner, window.title),
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
        index=["tier", "asset", "family", "family_title", "direction", "conditioner"],
        columns="sample",
        values=[
            "n", "wr", "pf", "exp", "ret_pct", "mdd",
            "avg_win", "avg_loss", "retention_pct",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{sample}" for metric, sample in pivot.columns]
    ranking_df = pivot.reset_index()

    yearly_full = yearly_df[~yearly_df["is_partial"]].copy()
    if not yearly_full.empty:
        yearly_stats = (
            yearly_full.groupby(
                ["tier", "asset", "family", "family_title", "direction", "conditioner"],
                dropna=False,
            )
            .agg(
                full_years=("year", "count"),
                full_years_pf_gt1=("pf", lambda s: int((s > 1.0).sum())),
                full_years_exp_pos=("exp", lambda s: int((s > 0).sum())),
                full_years_retention_warning=("retention_warning", lambda s: int(s.sum())),
            )
            .reset_index()
        )
        ranking_df = ranking_df.merge(
            yearly_stats,
            on=["tier", "asset", "family", "family_title", "direction", "conditioner"],
            how="left",
        )

    rolling_full = rolling_df[~rolling_df["is_partial"]].copy()
    if not rolling_full.empty:
        rolling_stats = (
            rolling_full.groupby(
                ["tier", "asset", "family", "family_title", "direction", "conditioner"],
                dropna=False,
            )
            .agg(
                rolling_windows=("label", "count"),
                rolling_pf_gt1=("pf", lambda s: int((s > 1.0).sum())),
                rolling_exp_pos=("exp", lambda s: int((s > 0).sum())),
                rolling_retention_warning=("retention_warning", lambda s: int(s.sum())),
            )
            .reset_index()
        )
        ranking_df = ranking_df.merge(
            rolling_stats,
            on=["tier", "asset", "family", "family_title", "direction", "conditioner"],
            how="left",
        )

    ranking_df = _add_delta_columns(ranking_df)
    ranking_df = ranking_df.sort_values(
        by=["asset", "family", "direction", "conditioner"],
    ).reset_index(drop=True)

    conditioner_ranking = (
        ranking_df[ranking_df["conditioner"] != "BASELINE"]
        .sort_values(
            by=["delta_pf_TEST", "delta_exp_TEST", "retention_pct_TEST"],
            ascending=[False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    summary_path = os.path.join(run_dir, "phase2_summary.csv")
    yearly_path = os.path.join(run_dir, "phase2_yearly.csv")
    rolling_path = os.path.join(run_dir, "phase2_rolling.csv")
    ranking_path = os.path.join(run_dir, "phase2_ranking.csv")
    conditioner_ranking_path = os.path.join(run_dir, "phase2_conditioner_ranking.csv")
    config_path = os.path.join(run_dir, "phase2_config.json")

    summary_df.to_csv(summary_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)
    rolling_df.to_csv(rolling_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)
    conditioner_ranking.to_csv(conditioner_ranking_path, index=False)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": "Phase 2 - one-conditioner session research",
                "mt5_timezone": MT5_TZ_DEFAULT,
                "data_dir": DATA_DIR,
                "test_start": str(TEST_START.date()),
                "shortlist": [c.__dict__ for c in SHORTLIST],
                "conditioners": {
                    "BASELINE": "Pure Phase 1 baseline reference.",
                    "NO_FOMC": "Exclude FOMC decision dates only. No other macro calendar added.",
                    "VOL_LOW_P50W90": (
                        "Causal daily volatility gate: prior completed day ATR <= "
                        "rolling p50 threshold over 90 completed trading days."
                    ),
                },
                "rolling_months": ROLL_MONTHS,
                "rolling_step_months": ROLL_STEP_MONTHS,
                "min_retention_warning_pct": MIN_RETENTION_WARN,
                "cost_model": "Same fixed minimum-lot cost model as Phase 1.",
                "non_goals": [
                    "No stacked conditioners",
                    "No parameter sweeps",
                    "No breakout or ORB",
                    "No operational runner translation",
                ],
            },
            f,
            indent=2,
        )

    _print_table(
        "BASELINE REFERENCES",
        ranking_df[ranking_df["conditioner"] == "BASELINE"].sort_values(
            by=["pf_TEST", "exp_TEST"],
            ascending=[False, False],
        ),
        [
            "tier", "asset", "family", "direction", "n_TEST", "pf_TEST",
            "exp_TEST", "ret_pct_TEST", "mdd_TEST", "full_years_pf_gt1",
            "rolling_pf_gt1",
        ],
    )

    _print_table(
        "CONDITIONER RANKING (delta vs baseline, TEST primary)",
        conditioner_ranking,
        [
            "tier", "asset", "family", "direction", "conditioner",
            "n_TEST", "retention_pct_TEST", "pf_TEST", "delta_pf_TEST",
            "exp_TEST", "delta_exp_TEST", "mdd_TEST", "delta_mdd_TEST",
            "full_years_pf_gt1", "rolling_pf_gt1",
        ],
        max_rows=40,
    )

    print()
    print("OUTPUTS")
    print(f"  {summary_path}")
    print(f"  {yearly_path}")
    print(f"  {rolling_path}")
    print(f"  {ranking_path}")
    print(f"  {conditioner_ranking_path}")
    print(f"  {config_path}")
    print("  Note: fixed minimum-lot research baseline; this is not an operational runner.")


if __name__ == "__main__":
    main()
