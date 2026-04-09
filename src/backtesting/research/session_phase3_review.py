#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 3 research review:
cross-asset comparison + translation candidacy over the governed Phase 2 shortlist.

Scope
-----
- Consume existing Phase 1 and Phase 2 outputs.
- Review only the Phase 2 survivor shortlist.
- Produce decision scorecards, cross-asset comparison, and translation memo.

Non-goals
---------
- No new families.
- No new conditioners.
- No backtest rerun.
- No operational runner / live / paper implementation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(BACKTEST_DIR))

PHASE1_ROOT = os.path.join(REPO_ROOT, "reports", "research_phase1")
PHASE2_ROOT = os.path.join(REPO_ROOT, "reports", "research_phase2")
PHASE3_ROOT = os.path.join(REPO_ROOT, "reports", "research_phase3")


@dataclass(frozen=True)
class CandidateMeta:
    tier: str
    asset: str
    family: str
    direction: str
    structural_note: str
    translation_difficulty: str
    translation_risk: str


PHASE3_SHORTLIST: list[CandidateMeta] = [
    CandidateMeta(
        "TIER1",
        "DE40",
        "US_MID",
        "LONG",
        "Late European / US overlap fixed-time drift; broad enough to avoid micro-breakout dependency.",
        "MEDIUM",
        "Needs runner translation of fixed-time entry/exit, session clock, spread/commission, and optional FOMC calendar.",
    ),
    CandidateMeta(
        "TIER1",
        "US30",
        "US_OPEN",
        "LONG",
        "US cash-open directional concentration; simple but higher-volatility hard-mode asset.",
        "MEDIUM",
        "Needs conservative cost/slippage review and train/test parity checks before operational translation.",
    ),
    CandidateMeta(
        "TIER2",
        "DE40",
        "US_OPEN",
        "LONG",
        "US-open spillover into DE40; structurally plausible cross-market session effect.",
        "MEDIUM",
        "Needs proof that weak TRAIN behavior is not masked by TEST-only improvement.",
    ),
    CandidateMeta(
        "WATCHLIST",
        "DE40",
        "EU_OPEN",
        "SHORT",
        "EU-open mean/drift candidate; plausible session effect but less stable than DE40 long candidates.",
        "HIGH",
        "EU-open execution is sensitive to opening volatility and conditioner sample collapse.",
    ),
    CandidateMeta(
        "WATCHLIST",
        "USTEC",
        "ASIA_EU_CONC",
        "LONG",
        "Asia-close / EU-open concentration candidate; weak after trust reset and not a USTEC rescue thesis.",
        "HIGH",
        "Needs stronger clean baseline evidence before any runner translation should be considered.",
    ),
]

CONDITIONERS = ["BASELINE", "NO_FOMC", "VOL_LOW_P50W90"]
MIN_CLEAN_RETENTION = 90.0
SAMPLE_COLLAPSE_RETENTION = 50.0


def latest_run(root: str, required_file: str) -> str:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Missing reports root: {root}")
    candidates = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, required_file)):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No run under {root} contains {required_file}")
    return max(candidates, key=os.path.getmtime)


def read_csv(run_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(run_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _row(
    df: pd.DataFrame,
    meta: CandidateMeta,
    conditioner: str | None = None,
) -> pd.Series | None:
    mask = (
        (df["asset"] == meta.asset)
        & (df["family"] == meta.family)
        & (df["direction"] == meta.direction)
    )
    if conditioner is not None:
        mask &= df["conditioner"] == conditioner
    rows = df[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def _float(row: pd.Series | None, col: str, default: float = 0.0) -> float:
    if row is None or col not in row or pd.isna(row[col]):
        return default
    return float(row[col])


def _int(row: pd.Series | None, col: str, default: int = 0) -> int:
    if row is None or col not in row or pd.isna(row[col]):
        return default
    return int(row[col])


def ratio(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return round(100.0 * num / den, 2)


def conditioner_note(row: pd.Series | None) -> str:
    if row is None:
        return "missing"
    retention = _float(row, "retention_pct_TEST")
    delta_pf = _float(row, "delta_pf_TEST")
    delta_exp = _float(row, "delta_exp_TEST")
    delta_mdd = _float(row, "delta_mdd_TEST")
    if retention < SAMPLE_COLLAPSE_RETENTION:
        return (
            f"sample collapse: retention {retention:.2f}%, "
            f"delta_pf {delta_pf:.3f}, delta_exp {delta_exp:.4f}, delta_mdd {delta_mdd:.2f}"
        )
    if retention >= MIN_CLEAN_RETENTION and delta_pf > 0 and delta_exp >= 0:
        mdd_text = "MDD improved" if delta_mdd >= 0 else "MDD worsened"
        return (
            f"clean modest improvement: retention {retention:.2f}%, "
            f"delta_pf {delta_pf:.3f}, delta_exp {delta_exp:.4f}, {mdd_text} {delta_mdd:.2f}"
        )
    if retention >= MIN_CLEAN_RETENTION and (delta_pf < 0 or delta_exp < 0):
        return (
            f"not helpful: retention {retention:.2f}%, "
            f"delta_pf {delta_pf:.3f}, delta_exp {delta_exp:.4f}"
        )
    return (
        f"mixed: retention {retention:.2f}%, "
        f"delta_pf {delta_pf:.3f}, delta_exp {delta_exp:.4f}, delta_mdd {delta_mdd:.2f}"
    )


def score_baseline(base: pd.Series) -> int:
    pf_test = _float(base, "pf_TEST")
    exp_test = _float(base, "exp_TEST")
    pf_train = _float(base, "pf_TRAIN")
    exp_train = _float(base, "exp_TRAIN")
    full_ratio = ratio(_int(base, "full_years_pf_gt1"), _int(base, "full_years"))
    rolling_ratio = ratio(_int(base, "rolling_pf_gt1"), _int(base, "rolling_windows"))
    n_test = _int(base, "n_TEST")

    score = 0
    if pf_test >= 1.25:
        score += 3
    elif pf_test >= 1.15:
        score += 2
    elif pf_test >= 1.05:
        score += 1

    if exp_test > 0:
        score += 2
    if pf_train >= 1.0:
        score += 2
    else:
        score -= 1
    if exp_train > 0:
        score += 1
    else:
        score -= 1
    if full_ratio >= 60:
        score += 2
    elif full_ratio >= 50:
        score += 1
    else:
        score -= 1
    if rolling_ratio >= 60:
        score += 2
    elif rolling_ratio >= 50:
        score += 1
    elif rolling_ratio < 45:
        score -= 1
    if n_test >= 250:
        score += 1
    return score


def promotion_decision(base: pd.Series, no_fomc: pd.Series | None, vol_low: pd.Series | None) -> str:
    pf_test = _float(base, "pf_TEST")
    exp_test = _float(base, "exp_TEST")
    pf_train = _float(base, "pf_TRAIN")
    full_ratio = ratio(_int(base, "full_years_pf_gt1"), _int(base, "full_years"))
    rolling_ratio = ratio(_int(base, "rolling_pf_gt1"), _int(base, "rolling_windows"))
    n_test = _int(base, "n_TEST")

    clean_no_fomc = (
        no_fomc is not None
        and _float(no_fomc, "retention_pct_TEST") >= MIN_CLEAN_RETENTION
        and _float(no_fomc, "delta_pf_TEST") > 0
        and _float(no_fomc, "delta_exp_TEST") >= 0
    )
    vol_sample_collapse = (
        vol_low is not None
        and _float(vol_low, "retention_pct_TEST") < SAMPLE_COLLAPSE_RETENTION
        and _float(vol_low, "delta_pf_TEST") > 0
    )

    if (
        pf_test >= 1.25
        and exp_test > 0
        and pf_train >= 1.0
        and full_ratio >= 60
        and rolling_ratio >= 60
        and n_test >= 250
    ):
        return "PROMOTE"
    if (
        pf_test >= 1.14
        and exp_test > 0
        and n_test >= 250
        and full_ratio >= 50
        and rolling_ratio >= 44
        and not vol_sample_collapse
    ):
        return "PROMOTE WITH CAUTION"
    if (
        pf_test >= 1.14
        and exp_test > 0
        and n_test >= 250
        and full_ratio >= 50
        and rolling_ratio >= 44
        and vol_sample_collapse
    ):
        return "WATCHLIST"
    if clean_no_fomc and pf_test >= 1.10 and exp_test > 0:
        return "WATCHLIST"
    if pf_test >= 1.10 and exp_test > 0:
        return "WATCHLIST"
    return "KILL"


def strengths_and_weaknesses(base: pd.Series, no_fomc: pd.Series | None, vol_low: pd.Series | None) -> tuple[str, str]:
    strengths: list[str] = []
    weaknesses: list[str] = []
    pf_test = _float(base, "pf_TEST")
    exp_test = _float(base, "exp_TEST")
    pf_train = _float(base, "pf_TRAIN")
    full_ratio = ratio(_int(base, "full_years_pf_gt1"), _int(base, "full_years"))
    rolling_ratio = ratio(_int(base, "rolling_pf_gt1"), _int(base, "rolling_windows"))

    if pf_test > 1.15 and exp_test > 0:
        strengths.append("positive TEST PF/expectancy")
    if pf_train >= 1.0:
        strengths.append("TRAIN PF above 1")
    else:
        weaknesses.append("TRAIN PF below 1")
    if full_ratio >= 50:
        strengths.append(f"full-year PF>1 ratio {full_ratio:.2f}%")
    else:
        weaknesses.append(f"weak full-year PF>1 ratio {full_ratio:.2f}%")
    if rolling_ratio >= 50:
        strengths.append(f"rolling PF>1 ratio {rolling_ratio:.2f}%")
    else:
        weaknesses.append(f"weak rolling PF>1 ratio {rolling_ratio:.2f}%")

    if no_fomc is not None and _float(no_fomc, "retention_pct_TEST") >= MIN_CLEAN_RETENTION:
        if _float(no_fomc, "delta_pf_TEST") > 0 and _float(no_fomc, "delta_exp_TEST") >= 0:
            strengths.append("NO_FOMC improves cleanly")
        elif _float(no_fomc, "delta_pf_TEST") < 0 or _float(no_fomc, "delta_exp_TEST") < 0:
            weaknesses.append("NO_FOMC does not help")

    if vol_low is not None and _float(vol_low, "retention_pct_TEST") < SAMPLE_COLLAPSE_RETENTION:
        if _float(vol_low, "delta_pf_TEST") > 0 or _float(vol_low, "delta_exp_TEST") > 0:
            weaknesses.append("VOL_LOW improvement is sample-collapse sensitive")
        else:
            weaknesses.append("VOL_LOW is not helpful and cuts sample below 50%")

    if not strengths:
        strengths.append("none strong enough for promotion")
    if not weaknesses:
        weaknesses.append("no major Phase 3 weakness")
    return "; ".join(strengths), "; ".join(weaknesses)


def phase1_match(phase1_df: pd.DataFrame, meta: CandidateMeta, base: pd.Series) -> str:
    mask = (
        (phase1_df["asset"] == meta.asset)
        & (phase1_df["family"] == meta.family)
        & (phase1_df["direction"] == meta.direction)
    )
    rows = phase1_df[mask]
    if rows.empty:
        return "missing"
    p1 = rows.iloc[0]
    pf_ok = abs(_float(p1, "pf_TEST") - _float(base, "pf_TEST")) <= 0.001
    exp_ok = abs(_float(p1, "exp_TEST") - _float(base, "exp_TEST")) <= 0.0001
    return "ok" if pf_ok and exp_ok else "mismatch"


def build_scorecards(phase1_df: pd.DataFrame, phase2_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for meta in PHASE3_SHORTLIST:
        base = _row(phase2_df, meta, "BASELINE")
        if base is None:
            raise RuntimeError(f"Missing baseline row for {meta.asset} {meta.family} {meta.direction}")
        no_fomc = _row(phase2_df, meta, "NO_FOMC")
        vol_low = _row(phase2_df, meta, "VOL_LOW_P50W90")
        strengths, weaknesses = strengths_and_weaknesses(base, no_fomc, vol_low)
        decision = promotion_decision(base, no_fomc, vol_low)
        final_variant = "BASELINE"
        if decision in {"PROMOTE", "PROMOTE WITH CAUTION"}:
            if no_fomc is not None and "clean modest improvement" in conditioner_note(no_fomc):
                final_variant = "BASELINE + optional NO_FOMC review"
        rows.append(
            {
                "tier": meta.tier,
                "asset": meta.asset,
                "family": meta.family,
                "direction": meta.direction,
                "baseline_pf_TEST": _float(base, "pf_TEST"),
                "baseline_exp_TEST": _float(base, "exp_TEST"),
                "baseline_pf_TRAIN": _float(base, "pf_TRAIN"),
                "baseline_exp_TRAIN": _float(base, "exp_TRAIN"),
                "baseline_n_TEST": _int(base, "n_TEST"),
                "baseline_mdd_TEST": _float(base, "mdd_TEST"),
                "baseline_wr_TEST": _float(base, "wr_TEST"),
                "full_years": _int(base, "full_years"),
                "full_years_pf_gt1": _int(base, "full_years_pf_gt1"),
                "full_years_pf_gt1_pct": ratio(_int(base, "full_years_pf_gt1"), _int(base, "full_years")),
                "rolling_windows": _int(base, "rolling_windows"),
                "rolling_pf_gt1": _int(base, "rolling_pf_gt1"),
                "rolling_pf_gt1_pct": ratio(_int(base, "rolling_pf_gt1"), _int(base, "rolling_windows")),
                "phase1_baseline_match": phase1_match(phase1_df, meta, base),
                "no_fomc_note": conditioner_note(no_fomc),
                "vol_low_note": conditioner_note(vol_low),
                "structural_note": meta.structural_note,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "translation_difficulty": meta.translation_difficulty,
                "translation_risk": meta.translation_risk,
                "score": score_baseline(base),
                "promotion_decision": decision,
                "phase4_variant": final_variant if decision in {"PROMOTE", "PROMOTE WITH CAUTION"} else "not promoted",
            }
        )
    scorecards = pd.DataFrame(rows)
    decision_order = {
        "PROMOTE": 0,
        "PROMOTE WITH CAUTION": 1,
        "WATCHLIST": 2,
        "KILL": 3,
    }
    scorecards["decision_rank"] = scorecards["promotion_decision"].map(decision_order)
    return scorecards.sort_values(
        by=["decision_rank", "score", "baseline_pf_TEST", "baseline_exp_TEST"],
        ascending=[True, False, False, False],
    ).drop(columns=["decision_rank"])


def build_cross_asset(scorecards: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for asset, group in scorecards.groupby("asset", sort=True):
        best = group.sort_values(
            by=["score", "baseline_pf_TEST", "baseline_exp_TEST"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(
            {
                "asset": asset,
                "candidate_count": int(len(group)),
                "best_family": best["family"],
                "best_direction": best["direction"],
                "best_decision": best["promotion_decision"],
                "best_pf_TEST": best["baseline_pf_TEST"],
                "best_exp_TEST": best["baseline_exp_TEST"],
                "total_n_TEST_reviewed": int(group["baseline_n_TEST"].sum()),
                "promote_count": int((group["promotion_decision"] == "PROMOTE").sum()),
                "caution_count": int((group["promotion_decision"] == "PROMOTE WITH CAUTION").sum()),
                "watchlist_count": int((group["promotion_decision"] == "WATCHLIST").sum()),
                "kill_count": int((group["promotion_decision"] == "KILL").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["promote_count", "caution_count", "best_pf_TEST"],
        ascending=[False, False, False],
    )


def build_translation_candidates(scorecards: pd.DataFrame) -> pd.DataFrame:
    promoted = scorecards[
        scorecards["promotion_decision"].isin(["PROMOTE", "PROMOTE WITH CAUTION"])
    ].copy()
    promoted = promoted.sort_values(
        by=["promotion_decision", "score", "baseline_pf_TEST"],
        ascending=[True, False, False],
    )
    promoted.insert(0, "phase4_rank", range(1, len(promoted) + 1))
    return promoted[
        [
            "phase4_rank",
            "asset",
            "family",
            "direction",
            "promotion_decision",
            "phase4_variant",
            "baseline_pf_TEST",
            "baseline_exp_TEST",
            "baseline_pf_TRAIN",
            "full_years_pf_gt1",
            "full_years",
            "rolling_pf_gt1",
            "rolling_windows",
            "translation_difficulty",
            "translation_risk",
        ]
    ]


def write_memo(
    path: str,
    phase1_run: str,
    phase2_run: str,
    scorecards: pd.DataFrame,
    cross_asset: pd.DataFrame,
    translation: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 3 Review - Cross-Asset Comparison + Translation Candidacy")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Phase 1 run: `{phase1_run}`")
    lines.append(f"- Phase 2 run: `{phase2_run}`")
    lines.append("- Scope: governed Phase 2 shortlist only; no rerun, no new filters, no runner translation.")
    lines.append("")
    lines.append("## Candidate Scorecards")
    for row in scorecards.itertuples(index=False):
        lines.append(f"### {row.asset} | {row.family} | {row.direction}")
        lines.append(f"- Decision: {row.promotion_decision}")
        lines.append(
            "- Baseline TEST: "
            f"PF {row.baseline_pf_TEST:.3f}, expectancy {row.baseline_exp_TEST:.4f}, "
            f"n {row.baseline_n_TEST}, MDD {row.baseline_mdd_TEST:.2f}"
        )
        lines.append(
            "- Consistency: "
            f"TRAIN PF {row.baseline_pf_TRAIN:.3f}; "
            f"full years PF>1 {row.full_years_pf_gt1}/{row.full_years}; "
            f"rolling PF>1 {row.rolling_pf_gt1}/{row.rolling_windows}"
        )
        lines.append(f"- Conditioner notes: NO_FOMC = {row.no_fomc_note}; VOL_LOW = {row.vol_low_note}")
        lines.append(f"- Strengths: {row.strengths}")
        lines.append(f"- Weaknesses: {row.weaknesses}")
        lines.append(f"- Translation difficulty: {row.translation_difficulty}")
        lines.append(f"- Translation risk: {row.translation_risk}")
        lines.append("")
    lines.append("## Cross-Asset Comparison")
    for row in cross_asset.itertuples(index=False):
        lines.append(
            f"- {row.asset}: best {row.best_family} {row.best_direction}, "
            f"decision {row.best_decision}, TEST PF {row.best_pf_TEST:.3f}, "
            f"promote {row.promote_count}, caution {row.caution_count}, watchlist {row.watchlist_count}."
        )
    lines.append("")
    lines.append("## Final Phase 4 Translation Shortlist")
    if translation.empty:
        lines.append("- No candidates promoted.")
    else:
        for row in translation.itertuples(index=False):
            lines.append(
                f"{row.phase4_rank}. {row.asset} | {row.family} | {row.direction} | "
                f"{row.promotion_decision} | {row.phase4_variant}"
            )
    lines.append("")
    lines.append("## Phase 3 Doctrine Check")
    lines.append("- Research baseline remains research baseline, not an operational runner.")
    lines.append("- VOL_LOW sample-collapse improvements are not used as promotion basis.")
    lines.append("- No new families, conditioners, thresholds, or parameter search were introduced.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    phase1_run = latest_run(PHASE1_ROOT, "phase1_ranking.csv")
    phase2_run = latest_run(PHASE2_ROOT, "phase2_ranking.csv")

    phase1_df = read_csv(phase1_run, "phase1_ranking.csv")
    phase2_df = read_csv(phase2_run, "phase2_ranking.csv")

    missing_conditioners = set(CONDITIONERS) - set(phase2_df["conditioner"].unique())
    if missing_conditioners:
        raise RuntimeError(f"Phase 2 ranking is missing conditioners: {sorted(missing_conditioners)}")

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(PHASE3_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    scorecards = build_scorecards(phase1_df, phase2_df)
    cross_asset = build_cross_asset(scorecards)
    translation = build_translation_candidates(scorecards)

    scorecards_path = os.path.join(run_dir, "phase3_scorecards.csv")
    cross_asset_path = os.path.join(run_dir, "phase3_cross_asset.csv")
    translation_path = os.path.join(run_dir, "phase3_translation_candidates.csv")
    memo_path = os.path.join(run_dir, "phase3_memo.md")
    config_path = os.path.join(run_dir, "phase3_config.json")

    scorecards.to_csv(scorecards_path, index=False)
    cross_asset.to_csv(cross_asset_path, index=False)
    translation.to_csv(translation_path, index=False)
    write_memo(memo_path, phase1_run, phase2_run, scorecards, cross_asset, translation)

    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "phase": "Phase 3 - cross-asset comparison + translation candidacy",
                "phase1_run": phase1_run,
                "phase2_run": phase2_run,
                "shortlist": [meta.__dict__ for meta in PHASE3_SHORTLIST],
                "conditioners_reviewed": CONDITIONERS,
                "promotion_guardrails": {
                    "no_new_families": True,
                    "no_new_conditioners": True,
                    "vol_low_sample_collapse_not_promotion_basis": True,
                    "research_baseline_not_operational_runner": True,
                },
            },
            fh,
            indent=2,
        )

    print("PHASE 3 REVIEW - CROSS-ASSET COMPARISON + TRANSLATION CANDIDACY")
    print(f"Phase 1 input: {phase1_run}")
    print(f"Phase 2 input: {phase2_run}")
    print(f"Output dir   : {run_dir}")
    print("")
    print("SCORECARDS")
    print(
        scorecards[
            [
                "asset",
                "family",
                "direction",
                "baseline_pf_TEST",
                "baseline_exp_TEST",
                "baseline_pf_TRAIN",
                "full_years_pf_gt1",
                "full_years",
                "rolling_pf_gt1",
                "rolling_windows",
                "promotion_decision",
                "translation_difficulty",
            ]
        ].to_string(index=False)
    )
    print("")
    print("FINAL PHASE 4 TRANSLATION SHORTLIST")
    if translation.empty:
        print("No candidates promoted.")
    else:
        print(
            translation[
                [
                    "phase4_rank",
                    "asset",
                    "family",
                    "direction",
                    "promotion_decision",
                    "phase4_variant",
                ]
            ].to_string(index=False)
        )
    print("")
    print("Wrote:")
    print(f"  {scorecards_path}")
    print(f"  {cross_asset_path}")
    print(f"  {translation_path}")
    print(f"  {memo_path}")
    print(f"  {config_path}")


if __name__ == "__main__":
    main()
