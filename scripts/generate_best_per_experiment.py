"""Generate xlsx tables showing only the best model per horizon for each experiment.

Output format per file:
    Horizon (h) | Algoritma | NSE | RMSE (m) | MAE (m)

Usage:
    python scripts/generate_best_per_experiment.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "tables"
PER_EXP = REPORTS / "per_experiment"
OUT_DIR = REPORTS / "per_experiment_best"
OUT_DIR.mkdir(exist_ok=True)

HORIZON_LABELS = {1: "+1 Jam", 2: "+2 Jam", 3: "+3 Jam", 4: "+4 Jam", 5: "+5 Jam"}
HORIZON_ORDER = ["+1 Jam", "+2 Jam", "+3 Jam", "+4 Jam", "+5 Jam"]


def best_per_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """From a full experiment table, pick the best model (highest NSE) per horizon."""
    rows = []
    for h in HORIZON_ORDER:
        sub = df[df["Horizon (h)"] == h]
        if sub.empty:
            continue
        best = sub.loc[sub["NSE"].idxmax()]
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def process_existing_xlsx():
    """Process all per-experiment xlsx files."""
    for f in sorted(PER_EXP.glob("exp*.xlsx")):
        df = pd.read_excel(f)
        best = best_per_horizon(df)
        out_name = f.stem + "_best.xlsx"
        out_path = OUT_DIR / out_name
        best.to_excel(out_path, index=False)
        print(f"  {out_name} ({len(best)} rows)")


def gen_combined_abc_best():
    """Generate best-per-horizon for Exp A/B/C from xls_13."""
    df = pd.read_excel(REPORTS / "xls_13_combined_experiment_comparison.xlsx")

    for label, nse_col, rmse_col, model_col in [
        ("exp2_A_train2022_test2023", "A: NSE", "A: RMSE", "A: Model"),
        ("exp3_B_combined_training", "B: NSE", "B: RMSE", "B: Model"),
        ("exp4_C_combined_rainfall", "C: NSE", "C: RMSE", "C: Model"),
    ]:
        out = pd.DataFrame({
            "Horizon (h)": df["Horizon"],
            "Algoritma": df[model_col],
            "NSE": df[nse_col],
            "RMSE (m)": df[rmse_col],
        })
        out_path = OUT_DIR / f"{label}_best.xlsx"
        out.to_excel(out_path, index=False)
        print(f"  {label}_best.xlsx ({len(out)} rows)")


def main():
    print("Generating best-per-horizon tables...\n")

    print("[Exp 2-4] A vs B vs C (from xls_13)")
    gen_combined_abc_best()

    print("\n[Exp 1, 5, 6, 7] From per-experiment tables")
    process_existing_xlsx()

    print(f"\nDone! Files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
