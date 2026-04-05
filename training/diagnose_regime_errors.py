"""Diagnostic: dekomposisi error model terbaik per regime muka air.

Hipotesis: RMSE model saat ini didominasi oleh error di regime flood (y >= 12).
Kalau regime normal (y < 10) sudah RMSE < 0.10, maka solusi regime-aware akan
langsung menembus target.

Usage
-----
    python training/diagnose_regime_errors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from dhompo.config import load_yaml_config, resolve_path_from_config
from dhompo.data.features import (
    align_features_targets,
    build_features_from_segments,
    build_targets,
)
from dhompo.data.loader import RAINFALL_COLUMN, UPSTREAM_STATIONS, load_combined_data
from dhompo.models.sklearn_models import HORIZONS, get_model_definitions
from training.evaluate import calc_metrics


REGIMES = [
    ("Normal (y<10)",    lambda y: y < 10.0),
    ("Elevated (10-12)", lambda y: (y >= 10.0) & (y < 12.0)),
    ("Flood (y>=12)",    lambda y: y >= 12.0),
]


def main() -> None:
    train_cfg = load_yaml_config("configs/training.yaml")
    model_cfg = load_yaml_config("configs/sklearn_model.yaml")
    train_split = train_cfg.get("train_split", 0.8)
    horizons = train_cfg.get("horizons", HORIZONS)
    horizon_steps = {int(h): int(h) * 2 for h in horizons}
    target_station = train_cfg.get("target_station", "Dhompo")
    extra_columns = [RAINFALL_COLUMN]  # match baseline A

    # Load segments
    source_paths = {}
    for src in train_cfg["data_sources"]:
        source_paths[src["label"]] = resolve_path_from_config(
            "configs/training.yaml", src["path"]
        )
    segments = load_combined_data(
        clean_path=source_paths["2022_clean"],
        generated_path=source_paths["2023_generated"],
    )

    X_features = build_features_from_segments(
        segments, upstream_stations=UPSTREAM_STATIONS, target=target_station,
        extra_columns=extra_columns,
    )
    y_parts = []
    for seg in segments:
        ys = build_targets(seg.df, horizons, horizon_steps, target=target_station)
        y_parts.append(pd.concat({h: s for h, s in ys.items()}, axis=1))
    y_all = pd.concat(y_parts, axis=0)
    y_horizons = {h: y_all[h] for h in horizons}
    X_full, y_horizons = align_features_targets(X_features, y_horizons)

    seg_2023_start = segments[1].df.index.min()
    idx_2023 = X_full.index >= seg_2023_start
    n_2023 = idx_2023.sum()
    n_2023_train = int(n_2023 * train_split)
    split_idx = (~idx_2023).sum() + n_2023_train

    X_train_raw = X_full.iloc[:split_idx]
    X_test_raw = X_full.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=X_train_raw.columns, index=X_train_raw.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X_test_raw.columns, index=X_test_raw.index,
    )

    # Use CatBoost (winner of most horizons) for diagnostic
    model_defs = get_model_definitions(model_cfg)
    catboost_template, use_scaled_cat = model_defs["CatBoost"]
    elasticnet_template, use_scaled_el = model_defs["ElasticNet"]

    # Train CatBoost on every horizon, ElasticNet on h5 only
    rows = []
    train_regime_summary = []

    # Test-set regime distribution
    y_test_current = X_test_raw["Dhompo_t0"]
    print("=" * 72)
    print("TEST SET regime distribution (by current level at time t)")
    print("=" * 72)
    for label, mask_fn in REGIMES:
        m = mask_fn(y_test_current)
        n = m.sum()
        pct = m.mean() * 100
        if n > 0:
            std_ = y_test_current[m].std()
            print(f"  {label:20s}: n={n:5d} ({pct:5.1f}%)  current_level std={std_:.3f}")
        else:
            print(f"  {label:20s}: n={n:5d} ({pct:5.1f}%)")

    print("\n" + "=" * 72)
    print("RMSE DECOMPOSITION by regime (CatBoost / best per horizon)")
    print("=" * 72)

    for h in horizons:
        y_train = y_horizons[h].iloc[:split_idx]
        y_test = y_horizons[h].iloc[split_idx:]

        # Pick best algorithm per horizon (h5 → ElasticNet, else CatBoost)
        if h == 5:
            name = "ElasticNet"
            model = clone(elasticnet_template)
            use_scaled = use_scaled_el
        else:
            name = "CatBoost"
            model = clone(catboost_template)
            use_scaled = use_scaled_cat

        Xtr = X_train_s if use_scaled else X_train_raw
        Xte = X_test_s if use_scaled else X_test_raw

        model.fit(Xtr, y_train)
        y_pred = pd.Series(model.predict(Xte), index=y_test.index)

        # Overall
        ov = calc_metrics(y_test.values, y_pred.values)
        print(f"\n--- h{h} ({name}) ---")
        print(f"  OVERALL         : n={len(y_test):5d}  "
              f"RMSE={ov['RMSE']:.4f}  MAE={ov['MAE']:.4f}  NSE={ov['NSE']:.4f}")

        # Group by TARGET y[t+h] regime (the thing we're predicting)
        for label, mask_fn in REGIMES:
            m = mask_fn(y_test)
            n = m.sum()
            if n < 10:
                print(f"  [by target y+h] {label:18s}: n={n:5d}  (too few)")
                continue
            met = calc_metrics(y_test[m].values, y_pred[m].values)
            mean_abs_err = met["MAE"]
            pct_err = mean_abs_err / y_test[m].mean() * 100
            rows.append({
                "horizon": h, "model": name, "group_by": "target",
                "regime": label, "n": n,
                "RMSE": met["RMSE"], "MAE": met["MAE"],
                "bias_mean_err": (y_pred[m] - y_test[m]).mean(),
                "PBIAS": met["PBIAS"],
                "pct_err": pct_err,
            })
            print(f"  [by target y+h] {label:18s}: n={n:5d}  "
                  f"RMSE={met['RMSE']:.4f}  MAE={met['MAE']:.4f}  "
                  f"mean_err={(y_pred[m]-y_test[m]).mean():+.4f}")

        # Also group by CURRENT level y[t] (regime decision at prediction time)
        yt = X_test_raw["Dhompo_t0"]
        print(f"  -- by CURRENT level y[t] (what gate would use) --")
        for label, mask_fn in REGIMES:
            m = mask_fn(yt)
            n = m.sum()
            if n < 10:
                continue
            met = calc_metrics(y_test[m].values, y_pred[m].values)
            rows.append({
                "horizon": h, "model": name, "group_by": "current",
                "regime": label, "n": n,
                "RMSE": met["RMSE"], "MAE": met["MAE"],
                "bias_mean_err": (y_pred[m] - y_test[m]).mean(),
                "PBIAS": met["PBIAS"],
                "pct_err": met["MAE"] / y_test[m].mean() * 100,
            })
            print(f"  [by current y(t)] {label:18s}: n={n:5d}  "
                  f"RMSE={met['RMSE']:.4f}  MAE={met['MAE']:.4f}  "
                  f"mean_err={(y_pred[m]-y_test[m]).mean():+.4f}")

    # Save
    df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "reports" / "tables" / "diagnostic_regime_errors.xlsx"
    df.to_excel(out_path, index=False)

    print("\n" + "=" * 72)
    print("SUMMARY: RMSE target < 0.10 feasibility per regime (by CURRENT level)")
    print("=" * 72)
    print(f"{'H':>2s}  {'Regime':<20s}  {'n':>5s}  {'RMSE':>8s}  {'feasible_<0.10?':>16s}")
    print("-" * 66)
    sub = df[df["group_by"] == "current"].sort_values(["horizon"])
    for _, r in sub.iterrows():
        feas = "YES" if r["RMSE"] < 0.10 else ("CLOSE" if r["RMSE"] < 0.15 else "no")
        print(f"{r['horizon']:>2d}  {r['regime']:<20s}  {r['n']:>5d}  "
              f"{r['RMSE']:>8.4f}  {feas:>16s}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
