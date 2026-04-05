"""Validate and compare data-clean.csv vs Data generated 2023.xlsx.

Run before training to ensure both datasets are compatible.

Usage
-----
    python training/validate_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from dhompo.data.loader import (
    ALL_STATIONS,
    RAINFALL_COLUMN,
    load_combined_data,
    load_data,
    load_generated_data,
)


def validate_schema(df_clean: pd.DataFrame, df_gen: pd.DataFrame) -> bool:
    """Check that both datasets have the expected station columns."""
    ok = True
    for st in ALL_STATIONS:
        if st not in df_clean.columns:
            print(f"  [WARN] Station '{st}' missing in clean data")
            ok = False
        if st not in df_gen.columns:
            print(f"  [WARN] Station '{st}' missing in generated data")
            ok = False
    return ok


def check_frequency(df: pd.DataFrame, label: str) -> None:
    """Verify 30-minute frequency."""
    if df.index.freq is None:
        diffs = df.index.to_series().diff().dropna()
        mode_freq = diffs.mode()[0] if len(diffs.mode()) > 0 else "unknown"
        print(f"  [{label}] No freq set. Most common interval: {mode_freq}")
    else:
        print(f"  [{label}] Frequency: {df.index.freq}")

    n_missing = df.isna().sum()
    total = len(df)
    cols_with_missing = n_missing[n_missing > 0]
    if len(cols_with_missing) > 0:
        print(f"  [{label}] Missing values:")
        for col, cnt in cols_with_missing.items():
            print(f"    {col}: {cnt}/{total} ({100*cnt/total:.1f}%)")
    else:
        print(f"  [{label}] No missing values")


def compare_statistics(df_clean: pd.DataFrame, df_gen: pd.DataFrame) -> None:
    """Compare per-station descriptive statistics."""
    common = [c for c in ALL_STATIONS if c in df_clean.columns and c in df_gen.columns]
    rows = []
    for st in common:
        rows.append({
            "Station": st,
            "Clean Mean": df_clean[st].mean(),
            "Gen Mean": df_gen[st].mean(),
            "Clean Std": df_clean[st].std(),
            "Gen Std": df_gen[st].std(),
            "Clean Min": df_clean[st].min(),
            "Gen Min": df_gen[st].min(),
            "Clean Max": df_clean[st].max(),
            "Gen Max": df_gen[st].max(),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format="%.3f"))


def ks_test_per_station(df_clean: pd.DataFrame, df_gen: pd.DataFrame) -> None:
    """Run KS-test per station to check distribution similarity."""
    common = [c for c in ALL_STATIONS if c in df_clean.columns and c in df_gen.columns]
    print(f"\n{'Station':25s} {'KS Stat':>10s} {'p-value':>12s} {'Similar?':>10s}")
    print("-" * 60)
    for st in common:
        s1 = df_clean[st].dropna().values
        s2 = df_gen[st].dropna().values
        ks_stat, p_val = stats.ks_2samp(s1, s2)
        similar = "YES" if p_val > 0.05 else "NO"
        print(f"  {st:23s} {ks_stat:10.4f} {p_val:12.4e} {similar:>10s}")


def report_rainfall(rainfall: pd.Series) -> None:
    """Report distribution of the rainfall column."""
    print(f"\n  Total observations: {len(rainfall)}")
    print(f"  Non-zero count: {(rainfall > 0).sum()} ({100*(rainfall > 0).mean():.1f}%)")
    print(f"  Zero count: {(rainfall == 0).sum()} ({100*(rainfall == 0).mean():.1f}%)")
    print(f"  Mean: {rainfall.mean():.3f}")
    print(f"  Std: {rainfall.std():.3f}")
    print(f"  Max: {rainfall.max():.3f}")
    q = rainfall.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print(f"  Quantiles: {q.to_dict()}")


def main() -> None:
    clean_path = PROJECT_ROOT / "data" / "data-clean.csv"
    gen_path = PROJECT_ROOT / "data" / "Data generated 2023.xlsx"

    if not clean_path.exists():
        print(f"ERROR: {clean_path} not found")
        sys.exit(1)
    if not gen_path.exists():
        print(f"ERROR: {gen_path} not found")
        sys.exit(1)

    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    # Load data
    print("\n1. Loading datasets...")
    df_clean = load_data(clean_path)
    gen = load_generated_data(gen_path)
    df_gen = gen.stations

    print(f"  Clean: {df_clean.shape} rows x cols, {df_clean.index.min()} to {df_clean.index.max()}")
    print(f"  Generated: {df_gen.shape} rows x cols, {df_gen.index.min()} to {df_gen.index.max()}")

    # Schema
    print("\n2. Schema validation (14 canonical stations)...")
    schema_ok = validate_schema(df_clean, df_gen)
    print(f"  Result: {'PASS' if schema_ok else 'WARN — some stations missing'}")

    # Frequency & missing
    print("\n3. Frequency and missing values...")
    check_frequency(df_clean, "Clean")
    check_frequency(df_gen, "Generated")

    # Statistics
    print("\n4. Descriptive statistics comparison...")
    compare_statistics(df_clean, df_gen)

    # KS test
    print("\n5. KS-test (distribution similarity, alpha=0.05)...")
    ks_test_per_station(df_clean, df_gen)

    # Rainfall
    print("\n6. Rainfall distribution (Curah Hujan)...")
    report_rainfall(gen.rainfall)

    # Segments
    print("\n7. Segment loading test...")
    segments = load_combined_data(clean_path, gen_path)
    for seg in segments:
        print(f"  [{seg.label}] {len(seg.df)} rows, cols={list(seg.df.columns[:3])}...")
        print(f"    Range: {seg.df.index.min()} to {seg.df.index.max()}")
        if seg.rainfall is not None:
            print(f"    Rainfall: {len(seg.rainfall)} values")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
