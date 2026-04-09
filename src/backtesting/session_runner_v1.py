#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 4 session candidate runner v1.

Purpose
-------
Translate the Phase 3 approved fixed-time research candidates into a minimal,
separate, runner-like backtest layer.

Scope
-----
- Fixed-time entry at the first bar of the approved window.
- Fixed-time exit at the last bar of the approved window.
- Fixed direction.
- Explicit spread and commission costs from project specs.
- Optional NO_FOMC comparison only where Phase 3 allowed it.
- Honest ALL / TRAIN / TEST, yearly, and rolling diagnostics.

Non-goals
---------
- No breakout / ORB / stop / trailing / take-profit logic.
- No runtime, paper, live, MT5 order, or london_bot integration.
- No new filters, windows, assets, or parameter search.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

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
)


REPORTS_ROOT = os.path.join(REPO_ROOT, "reports", "research_phase4")
PHASE3_ROOT = os.path.join(REPO_ROOT, "reports", "research_phase3")

DIRECTIONS = {"LONG": 1, "SHORT": -1}
VARIANT_BASELINE = "BASELINE"
VARIANT_NO_FOMC = "NO_FOMC"


@dataclass(frozen=True)
class SessionCandidateSpec:
    candidate_id: str
    asset: str
    family: str
    direction: str
    variants: tuple[str, ...]
    phase3_decision: str
    runner_like_notes: str
    remaining_translation_risk: str


SESSION_CANDIDATES: tuple[SessionCandidateSpec, ...] = (
    SessionCandidateSpec(
        candidate_id="DE40_US_MID_LONG",
        asset="DE40",
        family="US_MID",
        direction="LONG",
        variants=(VARIANT_BASELINE, VARIANT_NO_FOMC),
        phase3_decision="PROMOTE",
        runner_like_notes=(
            "Fixed UTC session entry/exit, fixed long direction, one position per trade date, "
            "spread and commission included."
        ),
        remaining_translation_risk=(
            "Future runtime translation still needs scheduler/order semantics, holiday handling, "
            "calendar source governance for NO_FOMC, live spread/slippage validation, and state reconciliation."
        ),
    ),
    SessionCandidateSpec(
        candidate_id="US30_US_OPEN_LONG",
        asset="US30",
        family="US_OPEN",
        direction="LONG",
        variants=(VARIANT_BASELINE,),
        phase3_decision="PROMOTE WITH CAUTION",
        runner_like_notes=(
            "Fixed UTC US-open entry/exit, fixed long direction, one position per trade date, "
            "spread and commission included."
        ),
        remaining_translation_risk=(
            "Future runtime translation still needs conservative US30 slippage review, live spread validation, "
            "scheduler/order semantics, and additional caution because TRAIN was weak."
        ),
    ),
    SessionCandidateSpec(
        candidate_id="DE40_US_OPEN_LONG",
        asset="DE40",
        family="US_OPEN",
        direction="LONG",
        variants=(VARIANT_BASELINE, VARIANT_NO_FOMC),
        phase3_decision="PROMOTE WITH CAUTION",
        runner_like_notes=(
            "Fixed UTC US-open entry/exit, fixed long direction, one position per trade date, "
            "spread and commission included."
        ),
        remaining_translation_risk=(
            "Future runtime translation still needs scheduler/order semantics, live spread/slippage validation, "
            "and proof that weak TRAIN behavior is not masked by TEST-only strength."
        ),
    ),
)


def _sample_bounds(data_start: pd.Timestamp, data_end: pd.Timestamp, sample: str) -> tuple[date, date]:
    start_date = data_start.date()
    end_date = data_end.date()
    test_start_date = TEST_START.normalize().date()
    if sample == "ALL":
        return start_date, end_date
    if sample == "TRAIN":
        return start_date, min(end_date, (TEST_START - pd.Timedelta(days=1)).date())
    if sample == "TEST":
        return max(start_date, test_start_date), end_date
    raise ValueError(f"Unknown sample: {sample}")


def _subset_dates(
    df: pd.DataFrame,
    start_date: date | None = None,
    end_exclusive: date | None = None,
) -> pd.DataFrame:
    out = df
    if start_date is not None:
        out = out[out["trade_date"] >= start_date]
    if end_exclusive is not None:
        out = out[out["trade_date"] < end_exclusive]
    return out.copy()


def _apply_variant(days_df: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == VARIANT_BASELINE:
        return days_df.copy()
    if variant == VARIANT_NO_FOMC:
        return days_df[~days_df["trade_date"].isin(_FOMC_DATES)].copy()
    raise ValueError(f"Unknown candidate variant: {variant}")


def _sample_name(trade_date: date) -> str:
    return "TEST" if trade_date >= TEST_START.date() else "TRAIN"


def simulate_session_candidate(
    days_df: pd.DataFrame,
    spec: SessionCandidateSpec,
    variant: str,
    params: dict,
    cap_start: float,
    start_date: date | None = None,
    end_exclusive: date | None = None,
) -> pd.DataFrame:
    if spec.direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction: {spec.direction}")
    if variant not in spec.variants:
        raise ValueError(f"{spec.candidate_id} does not allow variant {variant}")

    trades = _apply_variant(days_df, variant)
    trades = _subset_dates(trades, start_date, end_exclusive)
    if trades.empty:
        return pd.DataFrame()

    direction = DIRECTIONS[spec.direction]
    lots = float(params["ml"])
    contract_size = float(params["cs"])
    commission_rt = float(params["comm"])
    fallback_spread_px = float(params["sp"])
    is_jpy = bool(params.get("jpy", False))

    spread_px = trades["entry_spread_px"].where(
        trades["entry_spread_px"].notna() & (trades["entry_spread_px"] > 0),
        fallback_spread_px,
    )
    avg_px = (trades["entry_px"] + trades["exit_px"]) / 2.0
    multiplier = np.where(is_jpy, contract_size / np.maximum(avg_px, 1.0), contract_size)

    if direction == 1:
        entry_exec_px = trades["entry_px"] + spread_px
        exit_exec_px = trades["exit_px"]
        gross_usd = (exit_exec_px - entry_exec_px) * lots * multiplier
    else:
        entry_exec_px = trades["entry_px"] - spread_px
        exit_exec_px = trades["exit_px"]
        gross_usd = (entry_exec_px - exit_exec_px) * lots * multiplier

    # The spread is represented directly in entry_exec_px. Keep spread_cost_usd
    # explicit for audit without subtracting it twice.
    spread_cost_usd = spread_px * lots * contract_size
    commission_usd = lots * commission_rt
    net_usd = gross_usd - commission_usd
    equity = cap_start + np.cumsum(net_usd)
    result = np.where(net_usd > 0.01, "WIN", np.where(net_usd < -0.01, "LOSS", "BE"))

    out = trades.copy()
    out.insert(0, "candidate_id", spec.candidate_id)
    out["asset"] = spec.asset
    out["family"] = spec.family
    out["direction"] = spec.direction
    out["variant"] = variant
    out["sample_bucket"] = out["trade_date"].map(_sample_name)
    out["entry_rule"] = "first_bar_open_in_window"
    out["exit_rule"] = "last_bar_close_in_window"
    out["lots"] = lots
    out["contract_size"] = contract_size
    out["commission_rt_per_lot"] = commission_rt
    out["fallback_spread_px"] = fallback_spread_px
    out["spread_px_used"] = spread_px.astype(float)
    out["entry_exec_px"] = entry_exec_px.astype(float)
    out["exit_exec_px"] = exit_exec_px.astype(float)
    out["gross_usd"] = gross_usd.astype(float)
    out["spread_cost_usd"] = spread_cost_usd.astype(float)
    out["commission_usd"] = commission_usd
    out["net_usd"] = net_usd.astype(float)
    out["result"] = result
    out["cap"] = equity.astype(float)
    return out


def _metric_rows_for_samples(
    spec: SessionCandidateSpec,
    variant: str,
    days_df: pd.DataFrame,
    params: dict,
) -> list[dict]:
    rows: list[dict] = []
    data_start = pd.Timestamp(days_df["trade_date"].min())
    data_end = pd.Timestamp(days_df["trade_date"].max())
    baseline_n: dict[str, int] = {}

    for sample in ["ALL", "TRAIN", "TEST"]:
        sample_start, sample_end = _sample_bounds(data_start, data_end, sample)
        end_exclusive = sample_end + timedelta(days=1)
        baseline_trades = simulate_session_candidate(
            days_df,
            spec=spec,
            variant=VARIANT_BASELINE,
            params=params,
            cap_start=INITIAL_PER_ASSET,
            start_date=sample_start,
            end_exclusive=end_exclusive,
        )
        baseline_n[sample] = int(len(baseline_trades))

        trades = simulate_session_candidate(
            days_df,
            spec=spec,
            variant=variant,
            params=params,
            cap_start=INITIAL_PER_ASSET,
            start_date=sample_start,
            end_exclusive=end_exclusive,
        )
        metrics = compute_metrics(trades, INITIAL_PER_ASSET)
        rows.append(
            {
                **_row_prefix(spec, variant),
                "sample": sample,
                "sample_start": sample_start,
                "sample_end": sample_end,
                "baseline_n": baseline_n[sample],
                "retention_pct": round(metrics["n"] / baseline_n[sample] * 100, 2)
                if baseline_n[sample] > 0 else np.nan,
                **metrics,
                "lots": float(params["ml"]),
                "contract_size": float(params["cs"]),
                "commission_rt_per_lot": float(params["comm"]),
                "fallback_spread_px": float(params["sp"]),
                "cost_model": "entry_exec_price includes spread; roundtrip commission subtracted",
            }
        )
    return rows


def _yearly_rows(
    spec: SessionCandidateSpec,
    variant: str,
    days_df: pd.DataFrame,
    params: dict,
) -> list[dict]:
    rows: list[dict] = []
    data_start = pd.Timestamp(days_df["trade_date"].min())
    data_end = pd.Timestamp(days_df["trade_date"].max())
    for yr in iter_calendar_years(data_start, data_end):
        trades = simulate_session_candidate(
            days_df,
            spec=spec,
            variant=variant,
            params=params,
            cap_start=INITIAL_PER_ASSET,
            start_date=yr["start_date"],
            end_exclusive=yr["end_date"] + timedelta(days=1),
        )
        metrics = compute_metrics(trades, INITIAL_PER_ASSET)
        rows.append(
            {
                **_row_prefix(spec, variant),
                "year": yr["year"],
                "label": yr["label"],
                "is_partial": yr["is_partial"],
                "bucket": yr["bucket"],
                "sample_start": yr["start_date"],
                "sample_end": yr["end_date"],
                **metrics,
            }
        )
    return rows


def _rolling_rows(
    spec: SessionCandidateSpec,
    variant: str,
    days_df: pd.DataFrame,
    params: dict,
) -> list[dict]:
    rows: list[dict] = []
    data_start = pd.Timestamp(days_df["trade_date"].min())
    data_end = pd.Timestamp(days_df["trade_date"].max())
    for rw in iter_rolling_windows(data_start, data_end, ROLL_MONTHS, ROLL_STEP_MONTHS):
        trades = simulate_session_candidate(
            days_df,
            spec=spec,
            variant=variant,
            params=params,
            cap_start=INITIAL_PER_ASSET,
            start_date=rw["start_date"],
            end_exclusive=rw["end_date"] + timedelta(days=1),
        )
        metrics = compute_metrics(trades, INITIAL_PER_ASSET)
        rows.append(
            {
                **_row_prefix(spec, variant),
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
    return rows


def _row_prefix(spec: SessionCandidateSpec, variant: str) -> dict:
    window = WINDOW_SPECS[spec.family]
    return {
        "candidate_id": spec.candidate_id,
        "asset": spec.asset,
        "family": spec.family,
        "family_title": window.title,
        "direction": spec.direction,
        "variant": variant,
        "window_start_utc": window.start_hhmm,
        "window_end_utc": window.end_hhmm,
        "phase3_decision": spec.phase3_decision,
    }


def _latest_run(root: str, required_file: str) -> str | None:
    if not os.path.isdir(root):
        return None
    candidates = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, required_file)):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _build_ranking(summary_df: pd.DataFrame, yearly_df: pd.DataFrame, rolling_df: pd.DataFrame) -> pd.DataFrame:
    pivot = summary_df.pivot_table(
        index=[
            "candidate_id", "asset", "family", "family_title", "direction",
            "variant", "window_start_utc", "window_end_utc", "phase3_decision",
        ],
        columns="sample",
        values=["n", "wr", "pf", "exp", "ret_pct", "mdd", "avg_win", "avg_loss", "retention_pct"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{sample}" for metric, sample in pivot.columns]
    ranking_df = pivot.reset_index()

    yearly_full = yearly_df[~yearly_df["is_partial"]].copy()
    if not yearly_full.empty:
        yearly_stats = (
            yearly_full.groupby(["candidate_id", "asset", "family", "direction", "variant"], dropna=False)
            .agg(
                full_years=("year", "count"),
                full_years_pf_gt1=("pf", lambda s: int((s > 1.0).sum())),
                full_years_exp_pos=("exp", lambda s: int((s > 0).sum())),
            )
            .reset_index()
        )
        ranking_df = ranking_df.merge(
            yearly_stats,
            on=["candidate_id", "asset", "family", "direction", "variant"],
            how="left",
        )

    rolling_full = rolling_df[~rolling_df["is_partial"]].copy()
    if not rolling_full.empty:
        rolling_stats = (
            rolling_full.groupby(["candidate_id", "asset", "family", "direction", "variant"], dropna=False)
            .agg(
                rolling_windows=("label", "count"),
                rolling_pf_gt1=("pf", lambda s: int((s > 1.0).sum())),
                rolling_exp_pos=("exp", lambda s: int((s > 0).sum())),
            )
            .reset_index()
        )
        ranking_df = ranking_df.merge(
            rolling_stats,
            on=["candidate_id", "asset", "family", "direction", "variant"],
            how="left",
        )

    return ranking_df.sort_values(
        by=["pf_TEST", "exp_TEST", "n_TEST"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _write_specs_json(run_dir: str, params_by_asset: dict) -> str:
    path = os.path.join(run_dir, "phase4_candidate_specs.json")
    specs = []
    for spec in SESSION_CANDIDATES:
        window = WINDOW_SPECS[spec.family]
        params = params_by_asset[spec.asset]
        specs.append(
            {
                **spec.__dict__,
                "window": {
                    "source": "research.session_baselines_v2.WINDOW_SPECS",
                    "start_utc": window.start_hhmm,
                    "end_utc": window.end_hhmm,
                    "entry_rule": "first available M2 bar open inside window",
                    "exit_rule": "last available M2 bar close inside window",
                },
                "costs": {
                    "spread": "dynamic entry bar spread when available else fallback_spread_px",
                    "commission": "roundtrip commission per lot from resolved specs",
                    "lots": float(params["ml"]),
                    "contract_size": float(params["cs"]),
                    "fallback_spread_px": float(params["sp"]),
                    "commission_rt_per_lot": float(params["comm"]),
                },
            }
        )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(specs, fh, indent=2)
    return path


def _write_config_json(run_dir: str, specs_path: str, phase3_run: str | None, qc_by_asset: dict) -> str:
    path = os.path.join(run_dir, "phase4_config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "phase": "Phase 4 - session candidate runner v1",
                "data_dir": DATA_DIR,
                "phase3_source_run": phase3_run,
                "candidate_specs_file": specs_path,
                "test_start": str(TEST_START.date()),
                "all_definition": "all available candidate trade dates",
                "train_definition": "trade_date < TEST_START",
                "test_definition": "trade_date >= TEST_START",
                "rolling_months": ROLL_MONTHS,
                "rolling_step_months": ROLL_STEP_MONTHS,
                "capital_mode": (
                    "fixed minimum-lot candidate runner referenced to INITIAL_PER_ASSET; "
                    "not production sizing"
                ),
                "allowed_variants": {
                    "BASELINE": "No conditioner.",
                    "NO_FOMC": "Exclude FOMC decision dates only; optional comparison where Phase 3 allowed it.",
                },
                "explicit_non_goals": [
                    "no breakout",
                    "no ORB",
                    "no stop loss",
                    "no trailing",
                    "no take profit",
                    "no runtime/paper/live integration",
                    "no new filters",
                    "no parameter search",
                ],
                "data_qc": qc_by_asset,
            },
            fh,
            indent=2,
            default=str,
        )
    return path


def _write_memo(
    run_dir: str,
    ranking_df: pd.DataFrame,
    specs_path: str,
    phase3_run: str | None,
) -> str:
    path = os.path.join(run_dir, "phase4_translation_memo.md")
    lines: list[str] = []
    lines.append("# Phase 4 - Session Candidate Runner V1")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Fixed-time operational-candidate backtest only.")
    lines.append("- No runtime, paper/live, breakout, ORB, stops, trailing, or new filters.")
    lines.append("- Costs are explicit: spread at entry execution plus roundtrip commission.")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Candidate specs: `{specs_path}`")
    lines.append(f"- Phase 3 source run: `{phase3_run or 'not found'}`")
    lines.append("")
    lines.append("## Candidate Results")
    for row in ranking_df.sort_values(["candidate_id", "variant"]).itertuples(index=False):
        lines.append(f"### {row.candidate_id} | {row.variant}")
        lines.append(
            f"- TEST: PF {row.pf_TEST:.3f}, expectancy {row.exp_TEST:.4f}, "
            f"n {int(row.n_TEST)}, retention {row.retention_pct_TEST:.2f}%, MDD {row.mdd_TEST:.2f}"
        )
        lines.append(
            f"- TRAIN: PF {row.pf_TRAIN:.3f}, expectancy {row.exp_TRAIN:.4f}, n {int(row.n_TRAIN)}"
        )
        lines.append(
            f"- Full years PF>1: {int(row.full_years_pf_gt1)}/{int(row.full_years)}; "
            f"rolling PF>1: {int(row.rolling_pf_gt1)}/{int(row.rolling_windows)}"
        )
        lines.append("")
    lines.append("## Runner-Like Coverage")
    lines.append("- Aligned: deterministic entry/exit schedule, fixed direction, one candidate trade per eligible day, costs explicit, trade ledger emitted.")
    lines.append("- Still simplified: no live order queue, no slippage model beyond spread, no holiday/calendar provider, no broker execution validation, no runtime state machine.")
    lines.append("")
    lines.append("## Translation Guardrail")
    lines.append("- This is candidate research infrastructure. It is not production, paper, live, or london_bot integration.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _print_table(title: str, df: pd.DataFrame, columns: Iterable[str]) -> None:
    print()
    print(title)
    if df.empty:
        print("  (empty)")
        return
    print(df[list(columns)].to_string(index=False))


def main() -> None:
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPORTS_ROOT, f"run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    params_by_asset = resolve_asset_params(DATA_DIR)
    required_assets = sorted({spec.asset for spec in SESSION_CANDIDATES})
    missing_assets = [asset for asset in required_assets if asset not in params_by_asset]
    if missing_assets:
        raise RuntimeError(f"Missing resolved specs for assets: {missing_assets}")

    phase3_run = _latest_run(PHASE3_ROOT, "phase3_translation_candidates.csv")
    qc_by_asset: dict[str, dict] = {}
    asset_cache: dict[str, tuple[pd.DataFrame, dict]] = {}
    summary_rows: list[dict] = []
    yearly_rows: list[dict] = []
    rolling_rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []

    print("PHASE 4 - SESSION CANDIDATE RUNNER V1")
    print(f"Output dir: {run_dir}")
    print(f"Phase 3 source: {phase3_run or 'not found'}")
    print("")

    for spec in SESSION_CANDIDATES:
        params = params_by_asset[spec.asset]
        if spec.asset not in asset_cache:
            df, qc = prepare_asset_df(spec.asset, int(params["digits"]))
            asset_cache[spec.asset] = (df, qc)
            qc_by_asset[spec.asset] = qc
        df, _qc = asset_cache[spec.asset]
        window = WINDOW_SPECS[spec.family]
        days_df = build_window_days(df, window)
        if days_df.empty:
            raise RuntimeError(f"No window days for {spec.candidate_id}")

        print(
            f"{spec.candidate_id}: {spec.asset} {spec.family} {spec.direction} "
            f"{window.start_hhmm}-{window.end_hhmm} UTC | variants={list(spec.variants)}"
        )

        for variant in spec.variants:
            all_trades = simulate_session_candidate(
                days_df,
                spec=spec,
                variant=variant,
                params=params,
                cap_start=INITIAL_PER_ASSET,
            )
            trade_frames.append(all_trades)
            summary_rows.extend(_metric_rows_for_samples(spec, variant, days_df, params))
            yearly_rows.extend(_yearly_rows(spec, variant, days_df, params))
            rolling_rows.extend(_rolling_rows(spec, variant, days_df, params))

    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    yearly_df = pd.DataFrame(yearly_rows)
    rolling_df = pd.DataFrame(rolling_rows)
    ranking_df = _build_ranking(summary_df, yearly_df, rolling_df)

    specs_path = _write_specs_json(run_dir, params_by_asset)
    trades_path = os.path.join(run_dir, "phase4_trades.csv")
    summary_path = os.path.join(run_dir, "phase4_summary.csv")
    yearly_path = os.path.join(run_dir, "phase4_yearly.csv")
    rolling_path = os.path.join(run_dir, "phase4_rolling.csv")
    ranking_path = os.path.join(run_dir, "phase4_ranking.csv")

    trades_df.to_csv(trades_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)
    rolling_df.to_csv(rolling_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    config_path = _write_config_json(run_dir, specs_path, phase3_run, qc_by_asset)
    memo_path = _write_memo(run_dir, ranking_df, specs_path, phase3_run)

    _print_table(
        "CANDIDATE RANKING (TEST primary)",
        ranking_df,
        [
            "candidate_id", "variant", "n_TEST", "pf_TEST", "exp_TEST", "ret_pct_TEST",
            "mdd_TEST", "pf_TRAIN", "exp_TRAIN", "full_years", "full_years_pf_gt1",
            "rolling_windows", "rolling_pf_gt1",
        ],
    )

    print()
    print("OUTPUTS")
    print(f"  {specs_path}")
    print(f"  {trades_path}")
    print(f"  {summary_path}")
    print(f"  {yearly_path}")
    print(f"  {rolling_path}")
    print(f"  {ranking_path}")
    print(f"  {memo_path}")
    print(f"  {config_path}")


if __name__ == "__main__":
    main()
