#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading_Lab — Smoke Validation Suite  (N2.1)

Covers:
  A — Path integrity         (filesystem, no execution)
  B — Config loading         (imports only)
  C — Data loading           (reads parquets from data/backtesting/)
  D — Engine unit            (run_backtest + calc_metrics, one asset)
  G — London bot config      (config resolution, state file, stale-dir check)

Deferred to N2.1.runners:
  E — Backtest full run      (requires USE_LIVE_SPECS=False override + longer run)
  F — Risk runner full run   (requires USE_LIVE_SPECS=False override + longer run)
  H — Artifact placement     (depends on E/F output)

Usage (from repo root):
    .venv\\Scripts\\python.exe -X utf8 tests\\smoke_check.py

Expected duration: ~30–90 seconds  (D.4 runs Monte Carlo + bootstrap)
Exit code: 0 if all checks pass, 1 if any check fails.
"""

import os
import sys
import glob
import json
import time

# ─── Path bootstrap ───────────────────────────────────────────────────────────
# Must run before any project imports.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)                                      # for src.shared.*
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "backtesting"))  # for backtest_* bare imports

# ─── UTF-8 output (prevents codec errors on Windows with Unicode chars) ───────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Check runner ─────────────────────────────────────────────────────────────
_results: list = []


def check(name: str, fn) -> bool:
    """Execute fn(), print [PASS]/[FAIL], record result. Never raises."""
    try:
        fn()
        print(f"  [PASS] {name}")
        _results.append((name, True, ""))
        return True
    except AssertionError as e:
        print(f"  [FAIL] {name} — {e}")
        _results.append((name, False, str(e)))
        return False
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"  [FAIL] {name} — {msg}")
        _results.append((name, False, msg))
        return False


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK A — Path integrity
# ═════════════════════════════════════════════════════════════════════════════

def run_block_a() -> None:
    print("\n── BLOCK A: Path integrity ─────────────────────────────────────────")

    D_BT    = os.path.join(REPO_ROOT, "data", "backtesting")
    D_RUNS  = os.path.join(D_BT, "pipeline_runs")
    D_STATE = os.path.join(REPO_ROOT, "data", "state")
    L_BOT   = os.path.join(REPO_ROOT, "logs", "london_bot")
    R_BT    = os.path.join(REPO_ROOT, "reports", "backtests")
    R_RISK  = os.path.join(REPO_ROOT, "reports", "risk_analysis")
    R_AUDIT = os.path.join(REPO_ROOT, "reports", "execution_audit")
    R_BOT   = os.path.join(REPO_ROOT, "reports", "london_bot")

    def a1():
        missing = [d for d in [D_BT, D_RUNS, D_STATE, L_BOT,
                                R_BT, R_RISK, R_AUDIT, R_BOT]
                   if not os.path.isdir(d)]
        assert not missing, f"missing canonical dirs: {missing}"

    def a2():
        import backtest_config
        assets = (list(backtest_config.ASSET_PARAMS_BASE.keys())
                  + list(backtest_config.OPTIONAL_PARAMS.keys()))
        missing = [a for a in assets
                   if not os.path.exists(os.path.join(D_BT, f"{a}_Data.parquet"))]
        assert not missing, f"no parquet for: {missing}"

    def a3():
        jsons = glob.glob(os.path.join(D_BT, "especificaciones_*.json"))
        assert jsons, f"no especificaciones_*.json in {D_BT}"

    def a4():
        src_dir  = os.path.join(REPO_ROOT, "src")
        patterns = ["**/*.parquet", "**/*.log", "**/*.xlsx",
                    "**/*.csv",     "**/*.json"]
        stale = []
        for pat in patterns:
            for m in glob.glob(os.path.join(src_dir, pat), recursive=True):
                if "__pycache__" not in m:
                    stale.append(os.path.relpath(m, REPO_ROOT))
        assert not stale, f"artifacts found in src/ (first 5): {stale[:5]}"

    def a5():
        assert os.path.isdir(D_RUNS), f"pipeline_runs/ missing: {D_RUNS}"

    check("A.1 — canonical directories exist", a1)
    check("A.2 — parquet for every configured asset", a2)
    check("A.3 — especificaciones_*.json present in data/backtesting/", a3)
    check("A.4 — no stale artifacts under src/", a4)
    check("A.5 — pipeline_runs/ subdir exists inside data/backtesting/", a5)


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK B — Config loading
# ═════════════════════════════════════════════════════════════════════════════

def run_block_b() -> None:
    print("\n── BLOCK B: Config loading ─────────────────────────────────────────")

    def b1():
        from src.shared.config_loader import load_app_config
        cfg = load_app_config("london_bot")
        assert isinstance(cfg, dict) and cfg, "load_app_config returned empty or non-dict"
        assert "paths" in cfg, "'paths' key missing from config"

    def b2():
        from src.shared.config_loader import load_app_config
        cfg = load_app_config("london_bot")
        root = cfg["project_root"]
        assert os.path.isabs(root),  f"project_root is not absolute: {root}"
        assert os.path.isdir(root),  f"project_root does not exist: {root}"
        assert os.path.samefile(root, REPO_ROOT), \
            f"project_root mismatch: config={root} expected={REPO_ROOT}"

    def b3():
        from src.shared.config_loader import load_app_config
        cfg = load_app_config("london_bot")
        _root = cfg["project_root"]
        for key in ["state_file", "audit_file", "log_dir", "parity_file"]:
            raw      = cfg["paths"][key]
            resolved = os.path.join(_root, raw)
            assert os.path.isabs(resolved), \
                f"{key} not absolute after join: {resolved}"
            rel = os.path.relpath(resolved, REPO_ROOT)
            assert not rel.startswith("src"), \
                f"{key} resolves inside src/: {resolved}"

    def b4():
        import backtest_config
        assert isinstance(backtest_config.ASSET_PARAMS_BASE, dict), \
            "ASSET_PARAMS_BASE not a dict"
        assert len(backtest_config.ASSET_PARAMS_BASE) > 0, \
            "ASSET_PARAMS_BASE is empty"
        for attr in ["USE_LIVE_SPECS", "TEST_START",
                     "INITIAL_PER_ASSET", "GLOBAL_RISK_PCT"]:
            assert hasattr(backtest_config, attr), f"backtest_config missing {attr}"

    check("B.1 — load_app_config('london_bot') succeeds and has 'paths'", b1)
    check("B.2 — project_root is absolute and matches repo root", b2)
    check("B.3 — all 4 bot paths anchor to absolute non-src locations", b3)
    check("B.4 — backtest_config imports and has required constants", b4)


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK C — Data loading
# ═════════════════════════════════════════════════════════════════════════════

def run_block_c() -> dict:
    """Returns fixtures dict (df, lr, vm, qc) for use by Block D."""
    print("\n── BLOCK C: Data loading ───────────────────────────────────────────")
    DATA_DIR = os.path.join(REPO_ROOT, "data", "backtesting")
    fixtures: dict = {}

    def c1():
        import backtest_data
        df, lr, vm, qc = backtest_data.load_price_data("XAUUSD", DATA_DIR, digits=2)
        assert df is not None and len(df) > 0, "df is empty or None"
        assert lr is not None and len(lr) > 0, "london-range df is empty or None"
        assert isinstance(vm, (int, float)) and vm > 0, f"volume mean invalid: {vm}"
        assert isinstance(qc, dict), "qc is not a dict"
        fixtures.update({"df": df, "lr": lr, "vm": vm, "qc": qc})

    def c2():
        assert "df" in fixtures, "C.1 must pass first"
        df = fixtures["df"]
        required = ["time", "open", "high", "low", "close", "tick_volume", "atr14"]
        missing  = [c for c in required if c not in df.columns]
        assert not missing, f"df missing columns: {missing}"

    def c3():
        assert "lr" in fixtures, "C.1 must pass first"
        lr  = fixtures["lr"]
        qc  = fixtures["qc"]
        # London-range columns
        lr_required = ["date", "lh", "ll", "lrr"]
        lr_missing  = [c for c in lr_required if c not in lr.columns]
        assert not lr_missing, f"lr missing columns: {lr_missing}"
        # Volume mean
        assert fixtures["vm"] > 0, "volume mean is not positive"
        # QC dict keys
        qc_required = ["Asset", "Rows Final", "Has Spread Column"]
        qc_missing  = [k for k in qc_required if k not in qc]
        assert not qc_missing, f"qc missing keys: {qc_missing}"

    check("C.1 — load_price_data('XAUUSD') returns (df, lr, vm, qc)", c1)
    check("C.2 — df has required OHLCV + derived columns", c2)
    check("C.3 — lr has london-range cols; vm > 0; qc has expected keys", c3)
    return fixtures


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK D — Engine unit  (one asset, one full run)
# ═════════════════════════════════════════════════════════════════════════════

def run_block_d(c_fixtures: dict) -> None:
    print("\n── BLOCK D: Engine unit ────────────────────────────────────────────")
    fixtures: dict = {}

    def d1():
        import backtest_config, backtest_runner
        df = c_fixtures.get("df")
        lr = c_fixtures.get("lr")
        vm = c_fixtures.get("vm")
        assert df is not None, "C.1 must have passed (df not available)"
        p = backtest_config.ASSET_PARAMS_BASE["XAUUSD"].copy()
        p["risk_pct"] = 0.02
        trades, equity = backtest_runner.run_backtest(
            "XAUUSD", df, lr, vm, p, 250.0, None, None, "FULL"
        )
        assert hasattr(trades, "columns"), \
            f"run_backtest first return is not a DataFrame: {type(trades)}"
        assert hasattr(equity, "__len__"), \
            f"run_backtest second return is not array-like: {type(equity)}"
        fixtures["trades"] = trades
        fixtures["equity"] = equity

    def d2():
        assert "trades" in fixtures, "D.1 must pass first"
        trades   = fixtures["trades"]
        required = ["PnL Neto USD", "Resultado", "Date", "Mes",
                    "Comisión USD", "PnL Bruto USD", "Fecha Apertura"]
        missing  = [c for c in required if c not in trades.columns]
        assert not missing, f"trades_df missing columns consumed by risk runner: {missing}"

    def d3():
        assert "trades" in fixtures and "equity" in fixtures, "D.1 must pass first"
        t, e = fixtures["trades"], fixtures["equity"]
        assert len(e) == len(t) + 1, \
            f"equity len {len(e)} != trades len {len(t)} + 1"

    def d4():
        import backtest_stats
        assert "trades" in fixtures and "equity" in fixtures, "D.1 must pass first"
        t, e = fixtures["trades"], fixtures["equity"]
        if len(t) == 0:
            # Empty trades → empty dict is expected early-exit behavior
            m = backtest_stats.calc_metrics(t, e, 250.0)
            assert isinstance(m, dict), "calc_metrics did not return dict for empty trades"
            return
        m = backtest_stats.calc_metrics(t, e, 250.0)
        assert isinstance(m, dict) and m, \
            "calc_metrics returned empty dict for non-empty trades"
        required_keys = ["n", "ret", "mdd", "sharpe", "calmar", "pf", "wr",
                         "worst_day_usd", "best_day_usd", "mc_dd_p5",
                         "boot_lo", "boot_hi", "sign_p"]
        missing = [k for k in required_keys if k not in m]
        assert not missing, f"calc_metrics missing keys (incl. N1.5 additions): {missing}"

    check("D.1 — run_backtest(XAUUSD, FULL) returns (trades_df, equity_ndarray)", d1)
    check("D.2 — trades_df has the 7 columns consumed by risk runner", d2)
    check("D.3 — equity array length == len(trades) + 1", d3)
    check("D.4 — calc_metrics returns all expected keys incl. worst/best_day_usd", d4)


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK G — London bot config smoke
# ═════════════════════════════════════════════════════════════════════════════

def run_block_g() -> None:
    print("\n── BLOCK G: London bot config smoke ────────────────────────────────")

    def g1():
        from src.shared.config_loader import load_app_config
        cfg = load_app_config("london_bot")
        assert cfg, "config is empty"
        assert "paths" in cfg, "'paths' key missing"
        assert "assets" in cfg, "'assets' key missing"

    def g2():
        from src.shared.config_loader import load_app_config
        cfg   = load_app_config("london_bot")
        _root = cfg["project_root"]
        for key in ["state_file", "audit_file", "log_dir", "parity_file"]:
            assert key in cfg["paths"], f"paths.{key} missing from config"
            resolved = os.path.join(_root, cfg["paths"][key])
            assert os.path.isabs(resolved), \
                f"{key} not absolute after join: {resolved}"
            rel = os.path.relpath(resolved, REPO_ROOT)
            assert not rel.startswith("src"), \
                f"{key} resolves inside src/: {resolved}"

    def g3():
        state_path = os.path.join(REPO_ROOT, "data", "state", "london_bot_state.json")
        assert os.path.exists(state_path), f"state file missing: {state_path}"
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        for key in ["open_positions", "trades_today"]:
            assert key in state, f"state missing key: '{key}'"

    def g4():
        stale = [
            os.path.join(REPO_ROOT, "src", "london_bot", "data"),
            os.path.join(REPO_ROOT, "src", "london_bot", "logs"),
            os.path.join(REPO_ROOT, "src", "london_bot", "reports"),
        ]
        present = [d for d in stale if os.path.isdir(d)]
        assert not present, \
            f"stale artifact dirs still present under src/london_bot/: {present}"

    check("G.1 — load_app_config('london_bot') returns valid config", g1)
    check("G.2 — all 4 bot paths anchor to absolute non-src locations", g2)
    check("G.3 — london_bot_state.json exists and is valid JSON", g3)
    check("G.4 — no stale data/logs/reports dirs under src/london_bot/", g4)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  Trading_Lab — Smoke Validation Suite  (N2.1)            ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  Repo root: {REPO_ROOT}")

    run_block_a()
    run_block_b()
    c_fixtures = run_block_c()
    run_block_d(c_fixtures)
    run_block_g()

    # ── Summary ───────────────────────────────────────────────────────
    total   = len(_results)
    passed  = sum(1 for _, ok, _ in _results if ok)
    failed  = total - passed
    elapsed = time.time() - t0

    print(f"\n{'═' * 61}")
    print(f"  Total: {total}  |  Passed: {passed}  |  Failed: {failed}  |  {elapsed:.1f}s")
    if failed:
        print(f"\n  FAILED checks:")
        for name, ok, msg in _results:
            if not ok:
                print(f"    ✗ {name}")
                if msg:
                    print(f"      → {msg}")
    print(f"{'═' * 61}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
