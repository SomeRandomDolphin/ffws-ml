"""Δ-target (residual) learning experiment.

Hipotesis
---------
Memprediksi ``Δ = y[t+h] - y[t]`` (perubahan) lebih mudah daripada ``y[t+h]``
langsung karena model tidak perlu belajar ulang baseline level (~9.84). Hasil
akhir dikomposisi: ``y_pred_abs = y[t] + model_delta(X)``.

Perbandingan ABS vs DELTA dilakukan pada pipeline identik:
    - fitur: sama (baseline rainfall, tanpa B1-B4)
    - model: 9 algoritma × 5 horizon
    - split: segment-aware (all 2022 + 80% 2023 train, sisanya test)
    - evaluasi: metrics dihitung pada skala ABSOLUTE (apples-to-apples)

Usage
-----
    python training/run_delta_experiment.py
    python training/run_delta_experiment.py --save-models
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from dhompo.config import load_yaml_config, resolve_path_from_config
from dhompo.data.features import (
    align_features_targets,
    build_features_from_segments,
    build_targets,
)
from dhompo.data.loader import (
    RAINFALL_COLUMN,
    UPSTREAM_STATIONS,
    load_combined_data,
)
from dhompo.models.sklearn_models import HORIZONS, get_model_definitions
from training.evaluate import calc_metrics


TARGET_COL_FEATURE = "Dhompo_t0"  # feature name yang = y[t] (current level)


def _prepare_data(train_cfg: dict, include_rainfall: bool = True):
    """Return (X_train_raw, X_test_raw, y_horizons_abs, split_idx, target_station)."""
    train_split = train_cfg.get("train_split", 0.8)
    horizons = train_cfg.get("horizons", HORIZONS)
    horizon_steps = {int(h): int(h) * 2 for h in horizons}
    target_station = train_cfg.get("target_station", "Dhompo")
    extra_columns = [RAINFALL_COLUMN] if include_rainfall else None

    data_sources = train_cfg["data_sources"]
    source_paths = {}
    for src in data_sources:
        resolved = resolve_path_from_config("configs/training.yaml", src["path"])
        source_paths[src["label"]] = resolved

    segments = load_combined_data(
        clean_path=source_paths["2022_clean"],
        generated_path=source_paths["2023_generated"],
    )

    X_features = build_features_from_segments(
        segments,
        upstream_stations=UPSTREAM_STATIONS,
        target=target_station,
        extra_columns=extra_columns,
    )

    # Build absolute targets per segment (avoid cross-gap leak)
    y_parts = []
    for seg in segments:
        y_seg = build_targets(seg.df, horizons, horizon_steps, target=target_station)
        y_parts.append(pd.concat({h: s for h, s in y_seg.items()}, axis=1))
    y_all = pd.concat(y_parts, axis=0)
    y_horizons_abs = {h: y_all[h] for h in horizons}

    X_full, y_horizons_abs = align_features_targets(X_features, y_horizons_abs)

    # Split: all 2022 + 80% of 2023 → train
    seg_2023_start = segments[1].df.index.min()
    idx_2023 = X_full.index >= seg_2023_start
    n_2023 = idx_2023.sum()
    n_2023_train = int(n_2023 * train_split)
    split_idx = (~idx_2023).sum() + n_2023_train

    X_train_raw = X_full.iloc[:split_idx]
    X_test_raw = X_full.iloc[split_idx:]

    return X_train_raw, X_test_raw, y_horizons_abs, split_idx, horizons, target_station


def _run_one_mode(
    mode: str,  # "ABS" or "DELTA"
    X_train_raw, X_test_raw, y_horizons_abs, split_idx, horizons,
    model_cfg: dict,
    save_models: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Train all models under one target mode. Returns (metrics_df, best_models)."""

    # y[t] (current level) extracted from features — same rows as X
    y_t_train = X_train_raw[TARGET_COL_FEATURE]
    y_t_test = X_test_raw[TARGET_COL_FEATURE]

    # Scale features once (shared across scaled models)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=X_train_raw.columns, index=X_train_raw.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X_test_raw.columns, index=X_test_raw.index,
    )

    model_defs = get_model_definitions(model_cfg)
    results = []
    best_models: dict = {}

    for h in horizons:
        y_abs_train = y_horizons_abs[h].iloc[:split_idx]
        y_abs_test = y_horizons_abs[h].iloc[split_idx:]

        if mode == "ABS":
            y_train_fit = y_abs_train
            y_test_eval = y_abs_test
        elif mode == "DELTA":
            y_train_fit = y_abs_train - y_t_train   # Δ = future - current
            y_test_eval = y_abs_test                # we evaluate on ABS below
        else:
            raise ValueError(mode)

        print(f"\n--- [{mode}] Horizon +{h}h ---")
        for name, (model_template, use_scaled) in model_defs.items():
            Xtr = X_train_s if use_scaled else X_train_raw
            Xte = X_test_s if use_scaled else X_test_raw

            model = clone(model_template)
            model.fit(Xtr, y_train_fit)

            raw_pred = model.predict(Xte)
            if mode == "DELTA":
                # Compose: absolute prediction = current + delta
                y_pred_abs = y_t_test.values + raw_pred
            else:
                y_pred_abs = raw_pred

            # Train metrics on absolute scale (for apples-to-apples)
            raw_train_pred = model.predict(Xtr)
            y_train_pred_abs = (
                y_t_train.values + raw_train_pred if mode == "DELTA" else raw_train_pred
            )
            tr = calc_metrics(y_abs_train.values, y_train_pred_abs)
            te = calc_metrics(y_abs_test.values, y_pred_abs)

            y_mean = y_abs_test.mean()
            y_range = y_abs_test.max() - y_abs_test.min()
            nrmse_mean = te["RMSE"] / y_mean * 100 if y_mean else float("inf")

            results.append({
                "mode": mode,
                "horizon": h,
                "model": name,
                "train_NSE": tr["NSE"],
                "test_NSE": te["NSE"],
                "test_RMSE": te["RMSE"],
                "test_MAE": te["MAE"],
                "test_PBIAS": te["PBIAS"],
                "nRMSE_mean_%": nrmse_mean,
            })

            print(
                f"  {name:30s} NSE={te['NSE']:.4f}  RMSE={te['RMSE']:.4f}  "
                f"nRMSE={nrmse_mean:.2f}%"
            )

            key = (mode, h)
            if key not in best_models or te["NSE"] > best_models[key][1]:
                best_models[key] = (name, te["NSE"], model, scaler if use_scaled else None)

    return pd.DataFrame(results), best_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Δ-target (residual) learning experiment")
    parser.add_argument(
        "--save-models", action="store_true",
        help="Save best DELTA models to models/sklearn_delta/",
    )
    parser.add_argument(
        "--no-rainfall", action="store_true",
        help="Disable rainfall feature (default: include rainfall = True, match exp A)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml_config("configs/training.yaml")
    model_cfg = load_yaml_config("configs/sklearn_model.yaml")
    include_rainfall = not args.no_rainfall

    print("=" * 70)
    print("Δ-TARGET EXPERIMENT: Absolute vs Residual Learning")
    print("=" * 70)
    print(f"Feature set: A_Baseline (rainfall={include_rainfall})")

    t0 = time.time()
    X_train_raw, X_test_raw, y_horizons_abs, split_idx, horizons, target_station = _prepare_data(
        train_cfg, include_rainfall=include_rainfall,
    )
    print(
        f"Features: {X_train_raw.shape[1]} | "
        f"Train: {len(X_train_raw)} | Test: {len(X_test_raw)}"
    )

    # ABS baseline (replicates run_experiments.py Experiment A)
    df_abs, _ = _run_one_mode(
        "ABS", X_train_raw, X_test_raw, y_horizons_abs, split_idx, horizons, model_cfg,
    )

    # DELTA (residual) — same features, same split
    df_delta, best_delta = _run_one_mode(
        "DELTA", X_train_raw, X_test_raw, y_horizons_abs, split_idx, horizons, model_cfg,
        save_models=args.save_models,
    )

    # Save best DELTA models
    if args.save_models:
        out_dir = PROJECT_ROOT / "models" / "sklearn_delta"
        out_dir.mkdir(parents=True, exist_ok=True)
        scaler_saved = False
        for (_, h), (mname, nse, model, sc) in best_delta.items():
            fname = f"{mname.lower().replace(' ', '_')}_h{h}.pkl"
            joblib.dump(model, out_dir / fname)
            print(f"  Saved: {out_dir.name}/{fname} (NSE={nse:.4f})")
            if sc is not None and not scaler_saved:
                joblib.dump(sc, out_dir / "scaler.pkl")
                scaler_saved = True

    # Combine + save
    results_df = pd.concat([df_abs, df_delta], ignore_index=True)
    out_path = PROJECT_ROOT / "reports" / "tables" / "experiment_delta_vs_abs.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_excel(out_path, index=False)

    # Summary: best per mode × horizon
    print("\n" + "=" * 70)
    print("SUMMARY: Best model per horizon (ABS vs DELTA, evaluated on ABS scale)")
    print("=" * 70)
    print(f"{'Mode':<8s} {'H':>2s} {'Model':<22s} {'NSE':>8s} {'RMSE':>8s} {'nRMSE':>8s}")
    print("-" * 62)
    best_abs = df_abs.loc[df_abs.groupby("horizon")["test_NSE"].idxmax()]
    best_del = df_delta.loc[df_delta.groupby("horizon")["test_NSE"].idxmax()]
    for row in best_abs.itertuples():
        print(
            f"{'ABS':<8s} {row.horizon:>2d} {row.model:<22s} "
            f"{row.test_NSE:>8.4f} {row.test_RMSE:>8.4f} {row._asdict()['nRMSE_mean_%']:>7.2f}%"
        )
    print("-" * 62)
    for row in best_del.itertuples():
        print(
            f"{'DELTA':<8s} {row.horizon:>2d} {row.model:<22s} "
            f"{row.test_NSE:>8.4f} {row.test_RMSE:>8.4f} {row._asdict()['nRMSE_mean_%']:>7.2f}%"
        )

    # Head-to-head delta
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD RMSE REDUCTION (best ABS vs best DELTA per horizon)")
    print("=" * 70)
    print(f"{'H':>2s} {'RMSE_abs':>10s} {'RMSE_delta':>12s} {'Δ':>10s} {'%gain':>8s}")
    print("-" * 52)
    merged = best_abs.set_index("horizon")[["test_RMSE"]].rename(
        columns={"test_RMSE": "rmse_abs"}
    ).join(
        best_del.set_index("horizon")[["test_RMSE"]].rename(
            columns={"test_RMSE": "rmse_delta"}
        )
    )
    merged["delta"] = merged["rmse_abs"] - merged["rmse_delta"]
    merged["pct_gain"] = merged["delta"] / merged["rmse_abs"] * 100
    for h, row in merged.iterrows():
        print(
            f"{h:>2d} {row.rmse_abs:>10.4f} {row.rmse_delta:>12.4f} "
            f"{row.delta:>+10.4f} {row.pct_gain:>+7.2f}%"
        )

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
