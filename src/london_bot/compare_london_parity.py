from __future__ import annotations

from pathlib import Path
import pandas as pd


MODULAR_FILE = Path(r"C:\Users\Dasty\PycharmProjects\Trading_Lab\reports\london_bot\parity_log.csv")
LEGACY_FILE = Path(r"C:\Users\Dasty\PycharmProjects\Trading_Lab\reports\london_bot\parity_log_legacy.csv")
OUTPUT_FILE = Path(r"C:\Users\Dasty\PycharmProjects\Trading_Lab\reports\london_bot\parity_comparison.csv")


def eq(a, b, tol=1e-9):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol

    return a == b


def main() -> None:
    if not MODULAR_FILE.exists():
        print(f"No existe: {MODULAR_FILE}")
        return

    if not LEGACY_FILE.exists():
        print(f"No existe: {LEGACY_FILE}")
        return

    mod = pd.read_csv(MODULAR_FILE)
    leg = pd.read_csv(LEGACY_FILE)

    print("=== DEBUG INICIAL ===")
    print(f"Filas modular: {len(mod)}")
    print(f"Filas legacy:  {len(leg)}")
    print(f"Columnas modular: {list(mod.columns)}")
    print(f"Columnas legacy:  {list(leg.columns)}")
    print()

    if mod.empty or leg.empty:
        print("Uno o ambos parity logs están vacíos. No se puede comparar todavía.")
        return

    if "mt5_time" in mod.columns and "mt5_time" in leg.columns:
        mod["mt5_time"] = pd.to_datetime(mod["mt5_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        leg["mt5_time"] = pd.to_datetime(leg["mt5_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        key_cols = ["symbol", "signal_hour", "mt5_time"]
    else:
        key_cols = ["symbol", "signal_hour"]

    mod = mod.rename(columns={c: f"{c}_modular" for c in mod.columns if c not in key_cols})
    leg = leg.rename(columns={c: f"{c}_legacy" for c in leg.columns if c not in key_cols})

    merged = mod.merge(leg, on=key_cols, how="outer")

    print(f"Filas merged: {len(merged)}")
    print()

    if merged.empty:
        print("No hay filas para comparar después del merge.")
        return

    compare_pairs = [
        ("can_eval_modular", "can_eval_legacy", "match_can_eval"),
        ("eval_reason_modular", "eval_reason_legacy", "match_eval_reason"),
        ("signal_found_modular", "signal_found_legacy", "match_signal_found"),
        ("signal_type_modular", "signal_type_legacy", "match_signal_type"),
        ("direction_modular", "direction_legacy", "match_direction"),
        ("entry_price_modular", "entry_price_legacy", "match_entry_price"),
        ("stop_loss_modular", "stop_loss_legacy", "match_stop_loss"),
        ("stop_distance_modular", "stop_distance_legacy", "match_stop_distance"),
        ("lots_modular", "lots_legacy", "match_lots"),
        ("lrr_modular", "lrr_legacy", "match_lrr"),
        ("spread_signal_modular", "spread_signal_legacy", "match_spread_signal"),
    ]

    for a, b, out in compare_pairs:
        if a in merged.columns and b in merged.columns:
            merged[out] = merged.apply(lambda r: eq(r[a], r[b]), axis=1)

    match_cols = [c for _, _, c in compare_pairs if c in merged.columns]
    if match_cols:
        merged["all_match"] = merged[match_cols].all(axis=1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"Comparación guardada en: {OUTPUT_FILE}")

    if match_cols:
        print("\nResumen de coincidencias:")
        for c in match_cols:
            print(f"{c}: {merged[c].mean():.2%}")

    if "all_match" in merged.columns:
        mismatches = merged[~merged["all_match"]]
        print(f"\nFilas con mismatch total/parcial: {len(mismatches)}")


if __name__ == "__main__":
    main()