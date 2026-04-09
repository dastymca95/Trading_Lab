#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 5 pre-operational evaluation.

Purpose
-------
Stress the Phase 4 session candidates with a small, governed set of
pre-operational perturbations.

Scope
-----
- Three Phase 4 candidates only.
- Cost/friction inflation.
- Simple timing and hold-period perturbations.
- Optional NO_FOMC sensitivity only where Phase 4 already allowed it.

Non-goals
---------
- No new filters, assets, families, directions, or conditioners.
- No breakout / ORB / stop / trailing / sizing optimization.
- No runtime, paper, live, or london_bot integration.
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
from research.session_baselines_v2 import (  # noqa: E402
    DATA_DIR,
    ROLL_MONTHS,
    ROLL_STEP_MONTHS,
    WINDOW_SPECS,
    compute_metrics,
    iter_calendar_years,
    iter_rolling_windows,
    prepare_asset_df,
)
from session_runner_v1 import (  # noqa: E402
    SESSION_CANDIDATES,
    VARIANT_BASELINE,
    VARIANT_NO_FOMC,
    simulate_session_candidate,
)


REPORTS_ROOT = os.path.join(REPO_ROOT, "reports", "research_phase5")
PHASE4_ROOT = os.path.join(REPO_ROOT, "reports", "research_phase4")

MIN_OK_PF = 1.05
MIN_RETENTION_PCT = 85.0


@dataclass(frozen=True)
class ScenarioSpec:
    scenario: str
    category: str
    variant: str
    spread_mult: float = 1.0
    commission_mult: float = 1.0
    start_shift_min: int = 0
    end_shift_min: int = 0
    description: str = ""


@dataclass(frozen=True)
class ResolvedWindow:
    key: str
    title: str
    start_hhmm: str
    end_hhmm: str
    start_min: int
    end_min: int


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        scenario="BASELINE",
        category="BASELINE",
        variant=VARIANT_BASELINE,
        description="Original Phase 4 candidate window and baseline variant.",
    ),
    ScenarioSpec(
        scenario="COST_UP_1",
        category="COST",
        variant=VARIANT_BASELINE,
        spread_mult=1.25,
        commission_mult=1.25,
        description="Spread and commission multiplied by 1.25.",
    ),
    ScenarioSpec(
        scenario="COST_UP_2",
        category="COST",
        variant=VARIANT_BASELINE,
        spread_mult=1.50,
        commission_mult=1.50,
        description="Spread and commission multiplied by 1.50.",
    ),
    ScenarioSpec(
        scenario="TIME_SHIFT_EARLY_30",
        category="TIMING",
        variant=VARIANT_BASELINE,
        start_shift_min=-30,
        end_shift_min=-30,
        description="Entry and exit shifted 30 minutes earlier.",
    ),
    ScenarioSpec(
        scenario="TIME_SHIFT_LATE_30",
        category="TIMING",
        variant=VARIANT_BASELINE,
        start_shift_min=30,
        end_shift_min=30,
        description="Entry and exit shifted 30 minutes later.",
    ),
    ScenarioSpec(
        scenario="HOLD_SHORTER_30",
        category="HOLD",
        variant=VARIANT_BASELINE,
        end_shift_min=-30,
        description="Same entry, exit 30 minutes earlier.",
    ),
    ScenarioSpec(
        scenario="HOLD_LONGER_30",
        category="HOLD",
        variant=VARIANT_BASELINE,
        end_shift_min=30,
        description="Same entry, exit 30 minutes later.",
    ),
    ScenarioSpec(
        scenario="NO_FOMC",
        category="EVENT",
        variant=VARIANT_NO_FOMC,
        description="Exclude FOMC decision dates only; allowed where Phase 4 already approved NO_FOMC.",
    ),
)


def _min_to_hhmm(value: int) -> str:
    hour = value // 60
    minute = value % 60
    return f"{hour:02d}:{minute:02d}"


def _resolve_window(family: str, scenario: ScenarioSpec) -> ResolvedWindow:
    base = WINDOW_SPECS[family]
    start_min = int(base.start_min) + scenario.start_shift_min
    end_min = int(base.end_min) + scenario.end_shift_min
    if start_min < 0 or end_min > 23 * 60 + 59 or start_min > end_min:
        raise ValueError(
            f"Invalid scenario window for {family} {scenario.scenario}: "
            f"{start_min}->{end_min}"
        )
    return ResolvedWindow(
        key=f"{family}_{scenario.scenario}",
        title=f"{base.title} | {scenario.scenario}",
        start_hhmm=_min_to_hhmm(start_min),
        end_hhmm=_min_to_hhmm(end_min),
        start_min=start_min,
        end_min=end_min,
    )


def build_window_days_for_scenario(df: pd.DataFrame, window: ResolvedWindow) -> pd.DataFrame:
    sub = df[df["utc_min"].between(window.start_min, window.end_min)].copy()
    if sub.empty:
        return pd.DataFrame()

    rows: list[dict] = []
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


def apply_cost_inflation(days_df: pd.DataFrame, params: dict, scenario: ScenarioSpec) -> tuple[pd.DataFrame, dict]:
    out_days = days_df.copy()
    out_params = params.copy()
    if scenario.spread_mult != 1.0:
        out_days["entry_spread_px"] = out_days["entry_spread_px"] * scenario.spread_mult
        out_params["sp"] = float(out_params["sp"]) * scenario.spread_mult
    if scenario.commission_mult != 1.0:
        out_params["comm"] = float(out_params["comm"]) * scenario.commission_mult
    return out_days, out_params


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


def _row_prefix(spec, scenario: ScenarioSpec, window: ResolvedWindow) -> dict:
    return {
        "candidate_id": spec.candidate_id,
        "asset": spec.asset,
        "family": spec.family,
        "direction": spec.direction,
        "scenario": scenario.scenario,
        "category": scenario.category,
        "variant": scenario.variant,
        "window_start_utc": window.start_hhmm,
        "window_end_utc": window.end_hhmm,
        "spread_mult": scenario.spread_mult,
        "commission_mult": scenario.commission_mult,
        "start_shift_min": scenario.start_shift_min,
        "end_shift_min": scenario.end_shift_min,
        "phase4_decision": spec.phase3_decision,
    }


def _allowed_scenarios_for_candidate(spec) -> list[ScenarioSpec]:
    out: list[ScenarioSpec] = []
    for scenario in SCENARIOS:
        if scenario.variant == VARIANT_NO_FOMC and VARIANT_NO_FOMC not in spec.variants:
            continue
        out.append(scenario)
    return out


def _simulate(
    spec,
    scenario: ScenarioSpec,
    df: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, ResolvedWindow, dict]:
    window = _resolve_window(spec.family, scenario)
    days_df = build_window_days_for_scenario(df, window)
    if days_df.empty:
        return pd.DataFrame(), window, params.copy()
    scenario_days, scenario_params = apply_cost_inflation(days_df, params, scenario)
    trades = simulate_session_candidate(
        scenario_days,
        spec=spec,
        variant=scenario.variant,
        params=scenario_params,
        cap_start=INITIAL_PER_ASSET,
    )
    if not trades.empty:
        trades["scenario"] = scenario.scenario
        trades["category"] = scenario.category
        trades["window_start_utc"] = window.start_hhmm
        trades["window_end_utc"] = window.end_hhmm
    return trades, window, scenario_params


def _metrics_by_sample(
    spec,
    scenario: ScenarioSpec,
    days_df: pd.DataFrame,
    params: dict,
    window: ResolvedWindow,
    baseline_counts: dict[str, int],
) -> list[dict]:
    rows: list[dict] = []
    data_start = pd.Timestamp(days_df["trade_date"].min())
    data_end = pd.Timestamp(days_df["trade_date"].max())
    scenario_days, scenario_params = apply_cost_inflation(days_df, params, scenario)

    for sample in ["ALL", "TRAIN", "TEST"]:
        sample_start, sample_end = _sample_bounds(data_start, data_end, sample)
        trades = simulate_session_candidate(
            scenario_days,
            spec=spec,
            variant=scenario.variant,
            params=scenario_params,
            cap_start=INITIAL_PER_ASSET,
            start_date=sample_start,
            end_exclusive=sample_end + timedelta(days=1),
        )
        metrics = compute_metrics(trades, INITIAL_PER_ASSET)
        baseline_n = baseline_counts.get(sample, 0)
        rows.append(
            {
                **_row_prefix(spec, scenario, window),
                "sample": sample,
                "sample_start": sample_start,
                "sample_end": sample_end,
                "baseline_n": baseline_n,
                "retention_pct": round(metrics["n"] / baseline_n * 100, 2)
                if baseline_n > 0 else np.nan,
                **metrics,
            }
        )
    return rows


def _yearly_rows(spec, scenario: ScenarioSpec, days_df: pd.DataFrame, params: dict, window: ResolvedWindow) -> list[dict]:
    rows: list[dict] = []
    data_start = pd.Timestamp(days_df["trade_date"].min())
    data_end = pd.Timestamp(days_df["trade_date"].max())
    scenario_days, scenario_params = apply_cost_inflation(days_df, params, scenario)
    for yr in iter_calendar_years(data_start, data_end):
        trades = simulate_session_candidate(
            scenario_days,
            spec=spec,
            variant=scenario.variant,
            params=scenario_params,
            cap_start=INITIAL_PER_ASSET,
            start_date=yr["start_date"],
            end_exclusive=yr["end_date"] + timedelta(days=1),
        )
        rows.append(
            {
                **_row_prefix(spec, scenario, window),
                "year": yr["year"],
                "label": yr["label"],
                "is_partial": yr["is_partial"],
                "bucket": yr["bucket"],
                "sample_start": yr["start_date"],
                "sample_end": yr["end_date"],
                **compute_metrics(trades, INITIAL_PER_ASSET),
            }
        )
    return rows


def _rolling_rows(spec, scenario: ScenarioSpec, days_df: pd.DataFrame, params: dict, window: ResolvedWindow) -> list[dict]:
    rows: list[dict] = []
    data_start = pd.Timestamp(days_df["trade_date"].min())
    data_end = pd.Timestamp(days_df["trade_date"].max())
    scenario_days, scenario_params = apply_cost_inflation(days_df, params, scenario)
    for rw in iter_rolling_windows(data_start, data_end, ROLL_MONTHS, ROLL_STEP_MONTHS):
        trades = simulate_session_candidate(
            scenario_days,
            spec=spec,
            variant=scenario.variant,
            params=scenario_params,
            cap_start=INITIAL_PER_ASSET,
            start_date=rw["start_date"],
            end_exclusive=rw["end_date"] + timedelta(days=1),
        )
        rows.append(
            {
                **_row_prefix(spec, scenario, window),
                "label": rw["label"],
                "is_partial": rw["is_partial"],
                "bucket": rw["bucket"],
                "requested_start": rw["requested_start"],
                "requested_end": rw["requested_end"],
                "sample_start": rw["start_date"],
                "sample_end": rw["end_date"],
                **compute_metrics(trades, INITIAL_PER_ASSET),
            }
        )
    return rows


def _build_ranking(summary_df: pd.DataFrame, yearly_df: pd.DataFrame, rolling_df: pd.DataFrame) -> pd.DataFrame:
    pivot = summary_df.pivot_table(
        index=[
            "candidate_id", "asset", "family", "direction", "scenario", "category", "variant",
            "window_start_utc", "window_end_utc", "spread_mult", "commission_mult",
            "start_shift_min", "end_shift_min", "phase4_decision",
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
            yearly_full.groupby(["candidate_id", "scenario"], dropna=False)
            .agg(
                full_years=("year", "count"),
                full_years_pf_gt1=("pf", lambda s: int((s > 1.0).sum())),
                full_years_exp_pos=("exp", lambda s: int((s > 0).sum())),
            )
            .reset_index()
        )
        ranking_df = ranking_df.merge(yearly_stats, on=["candidate_id", "scenario"], how="left")

    rolling_full = rolling_df[~rolling_df["is_partial"]].copy()
    if not rolling_full.empty:
        rolling_stats = (
            rolling_full.groupby(["candidate_id", "scenario"], dropna=False)
            .agg(
                rolling_windows=("label", "count"),
                rolling_pf_gt1=("pf", lambda s: int((s > 1.0).sum())),
                rolling_exp_pos=("exp", lambda s: int((s > 0).sum())),
            )
            .reset_index()
        )
        ranking_df = ranking_df.merge(rolling_stats, on=["candidate_id", "scenario"], how="left")

    ranking_df["scenario_pass"] = (
        (ranking_df["pf_TEST"] >= MIN_OK_PF)
        & (ranking_df["exp_TEST"] > 0)
        & (ranking_df["retention_pct_TEST"] >= MIN_RETENTION_PCT)
    )
    return ranking_df.sort_values(
        by=["candidate_id", "category", "scenario"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def _pct(num: float, den: float) -> float:
    return round(num / den * 100.0, 2) if den else 0.0


def _classify_candidate(group: pd.DataFrame) -> dict:
    candidate_id = str(group["candidate_id"].iloc[0])
    baseline = group[group["scenario"] == "BASELINE"].iloc[0]
    perturb = group[group["category"].isin(["COST", "TIMING", "HOLD"])].copy()
    cost = group[group["category"] == "COST"].copy()
    timing_hold = group[group["category"].isin(["TIMING", "HOLD"])].copy()
    event = group[group["category"] == "EVENT"].copy()

    baseline_ok = bool(
        baseline["pf_TEST"] >= MIN_OK_PF
        and baseline["exp_TEST"] > 0
        and baseline["retention_pct_TEST"] >= MIN_RETENTION_PCT
    )
    cost_pass = int(cost["scenario_pass"].sum())
    timing_hold_pass = int(timing_hold["scenario_pass"].sum())
    perturb_pass = int(perturb["scenario_pass"].sum())
    perturb_total = int(len(perturb))
    train_ok = bool(baseline["pf_TRAIN"] >= 1.0 and baseline["exp_TRAIN"] > 0)
    full_ratio = _pct(float(baseline["full_years_pf_gt1"]), float(baseline["full_years"]))
    rolling_ratio = _pct(float(baseline["rolling_pf_gt1"]), float(baseline["rolling_windows"]))
    min_perturb_pf = float(perturb["pf_TEST"].min()) if not perturb.empty else np.nan
    min_perturb_exp = float(perturb["exp_TEST"].min()) if not perturb.empty else np.nan

    if not baseline_ok:
        readiness = "NOT TRANSLATION-READY"
    elif (
        cost_pass == len(cost)
        and timing_hold_pass >= 3
        and train_ok
        and full_ratio >= 55.0
        and rolling_ratio >= 55.0
    ):
        readiness = "ROBUST"
    elif (
        cost_pass >= 1
        and timing_hold_pass >= 2
        and full_ratio >= 45.0
        and rolling_ratio >= 45.0
    ):
        readiness = "ACCEPTABLE"
    elif cost_pass >= 1 or timing_hold_pass >= 1:
        readiness = "FRAGILE"
    else:
        readiness = "NOT TRANSLATION-READY"

    event_note = "not applicable"
    if not event.empty:
        row = event.iloc[0]
        event_note = (
            f"NO_FOMC TEST PF {row['pf_TEST']:.3f}, exp {row['exp_TEST']:.4f}, "
            f"retention {row['retention_pct_TEST']:.2f}%"
        )

    return {
        "candidate_id": candidate_id,
        "asset": baseline["asset"],
        "family": baseline["family"],
        "direction": baseline["direction"],
        "readiness": readiness,
        "baseline_pf_TEST": round(float(baseline["pf_TEST"]), 3),
        "baseline_exp_TEST": round(float(baseline["exp_TEST"]), 4),
        "baseline_pf_TRAIN": round(float(baseline["pf_TRAIN"]), 3),
        "cost_pass": cost_pass,
        "cost_total": int(len(cost)),
        "timing_hold_pass": timing_hold_pass,
        "timing_hold_total": int(len(timing_hold)),
        "perturb_pass": perturb_pass,
        "perturb_total": perturb_total,
        "min_perturb_pf_TEST": round(min_perturb_pf, 3) if pd.notna(min_perturb_pf) else np.nan,
        "min_perturb_exp_TEST": round(min_perturb_exp, 4) if pd.notna(min_perturb_exp) else np.nan,
        "full_years_pf_gt1": int(baseline["full_years_pf_gt1"]),
        "full_years": int(baseline["full_years"]),
        "rolling_pf_gt1": int(baseline["rolling_pf_gt1"]),
        "rolling_windows": int(baseline["rolling_windows"]),
        "event_note": event_note,
    }


def _build_fragility(ranking_df: pd.DataFrame) -> pd.DataFrame:
    rows = [_classify_candidate(group) for _, group in ranking_df.groupby("candidate_id", sort=False)]
    order = {
        "ROBUST": 0,
        "ACCEPTABLE": 1,
        "FRAGILE": 2,
        "NOT TRANSLATION-READY": 3,
    }
    out = pd.DataFrame(rows)
    out["rank_key"] = out["readiness"].map(order)
    return out.sort_values(
        by=["rank_key", "baseline_pf_TEST", "perturb_pass"],
        ascending=[True, False, False],
    ).drop(columns=["rank_key"]).reset_index(drop=True)


def _write_scenarios_json(run_dir: str, phase4_run: str | None) -> str:
    path = os.path.join(run_dir, "phase5_scenarios.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "phase": "Phase 5 - pre-operational evaluation",
                "phase4_source_run": phase4_run,
                "test_start": str(TEST_START.date()),
                "scenarios": [scenario.__dict__ for scenario in SCENARIOS],
                "readiness_rules": {
                    "scenario_pass": f"TEST PF >= {MIN_OK_PF}, TEST expectancy > 0, TEST retention >= {MIN_RETENTION_PCT}%",
                    "ROBUST": "baseline passes, all cost tests pass, >=3/4 timing+hold pass, TRAIN baseline positive, full/rolling ratios >=55%",
                    "ACCEPTABLE": "baseline passes, >=1/2 cost tests pass, >=2/4 timing+hold pass, full/rolling ratios >=45%",
                    "FRAGILE": "baseline passes but perturbation support is limited",
                    "NOT_TRANSLATION_READY": "baseline fails or perturbations do not support paper-candidate review",
                },
                "retention_note": (
                    "retention_pct compares each scenario sample count to the original BASELINE sample count; "
                    "timing-shift scenarios can exceed 100% at dataset boundaries."
                ),
                "non_goals": [
                    "no new filters",
                    "no new conditioners",
                    "no parameter search",
                    "no runtime integration",
                    "no USTEC review",
                ],
            },
            fh,
            indent=2,
        )
    return path


def _write_memo(run_dir: str, ranking_df: pd.DataFrame, fragility_df: pd.DataFrame, phase4_run: str | None) -> str:
    path = os.path.join(run_dir, "phase5_preop_memo.md")
    lines: list[str] = []
    lines.append("# Phase 5 - Pre-Operational Evaluation")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Three Phase 4 candidates only.")
    lines.append("- Perturbations: cost inflation, 30-minute timing shifts, 30-minute hold changes, NO_FOMC where already approved.")
    lines.append("- No new filters, families, directions, conditioners, or runtime integration.")
    lines.append("- Retention compares each scenario to original BASELINE sample count; timing shifts can exceed 100% at data boundaries.")
    lines.append("")
    lines.append("## Input")
    lines.append(f"- Phase 4 source run: `{phase4_run or 'not found'}`")
    lines.append("")
    lines.append("## Readiness")
    for row in fragility_df.itertuples(index=False):
        lines.append(f"### {row.candidate_id}")
        lines.append(f"- Classification: {row.readiness}")
        lines.append(
            f"- Baseline TEST: PF {row.baseline_pf_TEST:.3f}, exp {row.baseline_exp_TEST:.4f}; "
            f"TRAIN PF {row.baseline_pf_TRAIN:.3f}"
        )
        lines.append(
            f"- Perturbation pass: cost {row.cost_pass}/{row.cost_total}, "
            f"timing+hold {row.timing_hold_pass}/{row.timing_hold_total}, total {row.perturb_pass}/{row.perturb_total}"
        )
        lines.append(
            f"- Worst perturbation TEST: PF {row.min_perturb_pf_TEST:.3f}, "
            f"exp {row.min_perturb_exp_TEST:.4f}"
        )
        lines.append(
            f"- Full years PF>1 {row.full_years_pf_gt1}/{row.full_years}; "
            f"rolling PF>1 {row.rolling_pf_gt1}/{row.rolling_windows}"
        )
        lines.append(f"- Event sensitivity: {row.event_note}")
        lines.append("")
    lines.append("## Scenario Ranking")
    for row in ranking_df.sort_values(["candidate_id", "category", "scenario"]).itertuples(index=False):
        lines.append(
            f"- {row.candidate_id} | {row.scenario}: TEST PF {row.pf_TEST:.3f}, "
            f"exp {row.exp_TEST:.4f}, n {int(row.n_TEST)}, pass={bool(row.scenario_pass)}"
        )
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
    phase4_run = _latest_run(PHASE4_ROOT, "phase4_ranking.csv")
    asset_cache: dict[str, tuple[pd.DataFrame, dict]] = {}
    baseline_counts_by_candidate: dict[str, dict[str, int]] = {}
    summary_rows: list[dict] = []
    yearly_rows: list[dict] = []
    rolling_rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []

    print("PHASE 5 - PRE-OPERATIONAL EVALUATION")
    print(f"Output dir: {run_dir}")
    print(f"Phase 4 source: {phase4_run or 'not found'}")
    print("")

    for spec in SESSION_CANDIDATES:
        params = params_by_asset[spec.asset]
        if spec.asset not in asset_cache:
            df, qc = prepare_asset_df(spec.asset, int(params["digits"]))
            asset_cache[spec.asset] = (df, qc)
        df, _qc = asset_cache[spec.asset]

        baseline_scenario = SCENARIOS[0]
        baseline_window = _resolve_window(spec.family, baseline_scenario)
        baseline_days = build_window_days_for_scenario(df, baseline_window)
        if baseline_days.empty:
            raise RuntimeError(f"No baseline days for {spec.candidate_id}")
        baseline_counts: dict[str, int] = {}
        data_start = pd.Timestamp(baseline_days["trade_date"].min())
        data_end = pd.Timestamp(baseline_days["trade_date"].max())
        for sample in ["ALL", "TRAIN", "TEST"]:
            sample_start, sample_end = _sample_bounds(data_start, data_end, sample)
            sample_days = _subset_dates(baseline_days, sample_start, sample_end + timedelta(days=1))
            baseline_counts[sample] = int(len(sample_days))
        baseline_counts_by_candidate[spec.candidate_id] = baseline_counts

        print(f"{spec.candidate_id}: scenarios={len(_allowed_scenarios_for_candidate(spec))}")

        for scenario in _allowed_scenarios_for_candidate(spec):
            window = _resolve_window(spec.family, scenario)
            days_df = build_window_days_for_scenario(df, window)
            if days_df.empty:
                print(f"  [skip] {spec.candidate_id} {scenario.scenario}: no window days")
                continue
            scenario_days, scenario_params = apply_cost_inflation(days_df, params, scenario)
            trades = simulate_session_candidate(
                scenario_days,
                spec=spec,
                variant=scenario.variant,
                params=scenario_params,
                cap_start=INITIAL_PER_ASSET,
            )
            if not trades.empty:
                trades["scenario"] = scenario.scenario
                trades["category"] = scenario.category
                trades["window_start_utc"] = window.start_hhmm
                trades["window_end_utc"] = window.end_hhmm
                trades["spread_mult"] = scenario.spread_mult
                trades["commission_mult"] = scenario.commission_mult
                trade_frames.append(trades)

            summary_rows.extend(
                _metrics_by_sample(spec, scenario, days_df, params, window, baseline_counts)
            )
            yearly_rows.extend(_yearly_rows(spec, scenario, days_df, params, window))
            rolling_rows.extend(_rolling_rows(spec, scenario, days_df, params, window))

    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    yearly_df = pd.DataFrame(yearly_rows)
    rolling_df = pd.DataFrame(rolling_rows)
    ranking_df = _build_ranking(summary_df, yearly_df, rolling_df)
    fragility_df = _build_fragility(ranking_df)

    trades_path = os.path.join(run_dir, "phase5_trades.csv")
    summary_path = os.path.join(run_dir, "phase5_summary.csv")
    yearly_path = os.path.join(run_dir, "phase5_yearly.csv")
    rolling_path = os.path.join(run_dir, "phase5_rolling.csv")
    ranking_path = os.path.join(run_dir, "phase5_ranking.csv")
    fragility_path = os.path.join(run_dir, "phase5_fragility.csv")
    scenarios_path = _write_scenarios_json(run_dir, phase4_run)
    memo_path = _write_memo(run_dir, ranking_df, fragility_df, phase4_run)

    trades_df.to_csv(trades_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)
    rolling_df.to_csv(rolling_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)
    fragility_df.to_csv(fragility_path, index=False)

    _print_table(
        "FRAGILITY / READINESS",
        fragility_df,
        [
            "candidate_id", "readiness", "baseline_pf_TEST", "baseline_exp_TEST",
            "cost_pass", "cost_total", "timing_hold_pass", "timing_hold_total",
            "min_perturb_pf_TEST", "min_perturb_exp_TEST",
        ],
    )
    _print_table(
        "SCENARIO RANKING",
        ranking_df.sort_values(["candidate_id", "category", "scenario"]),
        [
            "candidate_id", "scenario", "category", "n_TEST", "pf_TEST", "exp_TEST",
            "mdd_TEST", "retention_pct_TEST", "scenario_pass",
        ],
    )

    print()
    print("OUTPUTS")
    print(f"  {trades_path}")
    print(f"  {summary_path}")
    print(f"  {yearly_path}")
    print(f"  {rolling_path}")
    print(f"  {ranking_path}")
    print(f"  {fragility_path}")
    print(f"  {scenarios_path}")
    print(f"  {memo_path}")


if __name__ == "__main__":
    main()
