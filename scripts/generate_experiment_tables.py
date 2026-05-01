"""Generate per-experiment xlsx tables in standardized format.

Output format per file:
    Horizon (h) | Algoritma | NSE | RMSE (m) | MAE (m)

Usage:
    python scripts/generate_experiment_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "tables"
OUT_DIR = REPORTS / "per_experiment"
OUT_DIR.mkdir(exist_ok=True)

HORIZON_LABELS = {1: "+1 Jam", 2: "+2 Jam", 3: "+3 Jam", 4: "+4 Jam", 5: "+5 Jam"}


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column order and sort."""
    df = df[["Horizon (h)", "Algoritma", "NSE", "RMSE (m)", "MAE (m)"]].copy()
    horizon_order = ["+1 Jam", "+2 Jam", "+3 Jam", "+4 Jam", "+5 Jam"]
    df["_h_sort"] = df["Horizon (h)"].map({v: i for i, v in enumerate(horizon_order)})
    df = df.sort_values(["_h_sort", "Algoritma"]).drop(columns="_h_sort").reset_index(drop=True)
    return df


def gen_2022_only():
    """Eksperimen 1: Training 2022 Only."""
    df = pd.read_excel(REPORTS / "xls_11_model_comparison_final.xlsx")
    out = df[["Horizon (h)", "Algoritma", "NSE", "RMSE (m)", "MAE (m)"]].copy()
    return fmt(out)


def gen_progressive_features():
    """Eksperimen 5: Progressive Features (A, B1, B2, B3, B4) — one file each."""
    df = pd.read_excel(REPORTS / "experiment_progressive_features.xlsx")
    results = {}
    for exp in df["experiment"].unique():
        sub = df[df["experiment"] == exp].copy()
        out = pd.DataFrame({
            "Horizon (h)": sub["horizon"].map(HORIZON_LABELS),
            "Algoritma": sub["model"],
            "NSE": sub["test_NSE"],
            "RMSE (m)": sub["test_RMSE"],
            "MAE (m)": sub["test_MAE"],
        })
        results[exp] = fmt(out)
    return results


def gen_delta_vs_abs():
    """Eksperimen 6: Delta vs Absolute — one file per mode."""
    df = pd.read_excel(REPORTS / "experiment_delta_vs_abs.xlsx")
    results = {}
    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode].copy()
        out = pd.DataFrame({
            "Horizon (h)": sub["horizon"].map(HORIZON_LABELS),
            "Algoritma": sub["model"],
            "NSE": sub["test_NSE"],
            "RMSE (m)": sub["test_RMSE"],
            "MAE (m)": sub["test_MAE"],
        })
        results[mode] = fmt(out)
    return results


def gen_smoothing():
    """Eksperimen 7: Target Smoothing — one file per config."""
    df = pd.read_excel(REPORTS / "experiment_target_smoothing.xlsx")
    results = {}
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg].copy()
        out = pd.DataFrame({
            "Horizon (h)": sub["horizon"].map(HORIZON_LABELS),
            "Algoritma": sub["model"],
            "NSE": sub["NSE_vs_raw"],
            "RMSE (m)": sub["RMSE_vs_raw"],
            "MAE (m)": sub["MAE_vs_raw"],
        })
        results[cfg] = fmt(out)
    return results


def save(df: pd.DataFrame, name: str):
    path = OUT_DIR / f"{name}.xlsx"
    df.to_excel(path, index=False)
    print(f"  {path.name} ({len(df)} rows)")


def main():
    print("Generating per-experiment tables...\n")

    # 1. Training 2022 Only
    print("[1] Training 2022 Only")
    save(gen_2022_only(), "exp1_training_2022_only")

    # 5. Progressive Features
    print("[5] Progressive Features")
    for exp_name, df in gen_progressive_features().items():
        safe_name = exp_name.replace("+", "plus_").replace(" ", "_")
        save(df, f"exp5_{safe_name}")

    # 6. Delta vs Absolute
    print("[6] Delta vs Absolute Target")
    for mode, df in gen_delta_vs_abs().items():
        save(df, f"exp6_target_{mode.lower()}")

    # 7. Target Smoothing
    print("[7] Target Smoothing")
    for cfg, df in gen_smoothing().items():
        save(df, f"exp7_smoothing_{cfg.lower()}")

    print(f"\nDone! Files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
