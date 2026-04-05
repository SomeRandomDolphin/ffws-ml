"""Run progressive feature enhancement experiments (no MLflow dependency).

Usage
-----
    python training/run_experiments.py
    python training/run_experiments.py --experiment B4
    python training/run_experiments.py --all

Runs experiments A through B4 with progressively more features enabled,
outputs comparison table to reports/tables/experiment_results.xlsx.
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
from dhompo.data.loader import (
    RAINFALL_COLUMN,
    UPSTREAM_STATIONS,
    load_combined_data,
)
from dhompo.models.sklearn_models import HORIZONS, get_model_definitions
from training.evaluate import calc_metrics

# Experiment configurations: (name, include_rainfall, B1, B2, B3, B4)
EXPERIMENTS = {
    "A": ("Baseline+Rainfall", True, False, False, False, False),
    "B1": ("+TravelTimeLags", True, True, False, False, False),
    "B2": ("+CumulativeRain", True, True, True, False, False),
    "B3": ("+Interaction", True, True, True, True, False),
    "B4": ("FullFeatures", True, True, True, True, True),
}


def run_experiment(
    exp_name: str,
    include_rainfall: bool,
    b1: bool, b2: bool, b3: bool, b4: bool,
    train_cfg: dict,
    model_cfg: dict,
    save_models: bool = False,
) -> pd.DataFrame:
    """Run one experiment and return metrics DataFrame."""

    print(f"\n{'='*60}")
    print(f"EXPERIMENT {exp_name}: rainfall={include_rainfall}, B1={b1}, B2={b2}, B3={b3}, B4={b4}")
    print(f"{'='*60}")

    train_split = train_cfg.get("train_split", 0.8)
    horizons = train_cfg.get("horizons", HORIZONS)
    horizon_steps = {int(h): int(h) * 2 for h in horizons}
    target_station = train_cfg.get("target_station", "Dhompo")

    extra_columns = [RAINFALL_COLUMN] if include_rainfall else None

    # Load data
    data_sources = train_cfg["data_sources"]
    source_paths = {}
    for src in data_sources:
        resolved = resolve_path_from_config("configs/training.yaml", src["path"])
        source_paths[src["label"]] = resolved

    segments = load_combined_data(
        clean_path=source_paths["2022_clean"],
        generated_path=source_paths["2023_generated"],
    )

    # Build features
    t0 = time.time()
    X_features = build_features_from_segments(
        segments,
        upstream_stations=UPSTREAM_STATIONS,
        target=target_station,
        extra_columns=extra_columns,
        use_travel_time_lags=b1,
        use_cumulative_rainfall=b2,
        use_interaction_features=b3,
        use_seasonal_features=b4,
    )

    # Build targets
    y_parts = []
    for seg in segments:
        y_seg = build_targets(seg.df, horizons, horizon_steps, target=target_station)
        y_combined = pd.concat({h: s for h, s in y_seg.items()}, axis=1)
        y_parts.append(y_combined)
    y_all = pd.concat(y_parts, axis=0)
    y_horizons = {h: y_all[h] for h in horizons}

    X_full, y_horizons = align_features_targets(X_features, y_horizons)

    # Split: all 2022 + 80% of 2023 → train
    seg_2023_start = segments[1].df.index.min()
    idx_2023 = X_full.index >= seg_2023_start
    n_2023 = idx_2023.sum()
    n_2023_train = int(n_2023 * train_split)
    split_idx = (~idx_2023).sum() + n_2023_train

    X_train_raw = X_full.iloc[:split_idx]
    X_test_raw = X_full.iloc[split_idx:]

    print(f"Features: {X_full.shape[1]} | Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")

    # Scale
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index
    )

    model_defs = get_model_definitions(model_cfg)
    results = []
    best_models = {}

    for h in horizons:
        y_train = y_horizons[h].iloc[:split_idx]
        y_test = y_horizons[h].iloc[split_idx:]

        print(f"\n--- Horizon +{h}h ---")
        for name, (model_template, use_scaled) in model_defs.items():
            Xtr = X_train_s if use_scaled else X_train_raw
            Xte = X_test_s if use_scaled else X_test_raw

            model = clone(model_template)
            model.fit(Xtr, y_train)

            train_metrics = calc_metrics(y_train.values, model.predict(Xtr))
            test_metrics = calc_metrics(y_test.values, model.predict(Xte))

            # Compute nRMSE (normalized by mean)
            y_mean = y_test.mean()
            y_range = y_test.max() - y_test.min()
            nrmse_mean = (test_metrics["RMSE"] / y_mean * 100) if y_mean != 0 else float("inf")
            nrmse_range = (test_metrics["RMSE"] / y_range * 100) if y_range != 0 else float("inf")

            results.append({
                "experiment": exp_name,
                "horizon": h,
                "model": name,
                "n_features": Xtr.shape[1],
                "train_NSE": train_metrics["NSE"],
                "test_NSE": test_metrics["NSE"],
                "test_RMSE": test_metrics["RMSE"],
                "test_MAE": test_metrics["MAE"],
                "test_PBIAS": test_metrics["PBIAS"],
                "test_R2": test_metrics["R2"],
                "nRMSE_mean_%": nrmse_mean,
                "nRMSE_range_%": nrmse_range,
            })

            nse = test_metrics["NSE"]
            rmse = test_metrics["RMSE"]
            print(f"  {name:30s} NSE={nse:.4f}  RMSE={rmse:.4f}  nRMSE={nrmse_mean:.1f}%")

            # Track best per horizon
            key = (exp_name, h)
            if key not in best_models or nse > best_models[key][1]:
                best_models[key] = (name, nse, model, scaler if use_scaled else None)

    elapsed = time.time() - t0
    print(f"\nExperiment {exp_name} completed in {elapsed:.1f}s")

    # Save best models if requested
    if save_models:
        models_dir = PROJECT_ROOT / "models" / "sklearn"
        models_dir.mkdir(parents=True, exist_ok=True)
        for (ename, h), (mname, nse, model, sc) in best_models.items():
            fname = f"{mname.lower().replace(' ', '_')}_h{h}.pkl"
            joblib.dump(model, models_dir / fname)
            print(f"  Saved: {fname} (NSE={nse:.4f})")
        if any(sc is not None for (_, sc) in [(v[2], v[3]) for v in best_models.values()]):
            joblib.dump(scaler, models_dir / "scaler.pkl")

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature enhancement experiments")
    parser.add_argument(
        "--experiment", default=None,
        help="Run specific experiment (A, B1, B2, B3, B4). Default: all.",
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument(
        "--save-models", action="store_true",
        help="Save best models from the last experiment to models/sklearn/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml_config("configs/training.yaml")
    model_cfg = load_yaml_config("configs/sklearn_model.yaml")

    if args.experiment:
        exps_to_run = [args.experiment.upper()]
    else:
        exps_to_run = list(EXPERIMENTS.keys())

    all_results = []

    for exp_key in exps_to_run:
        if exp_key not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_key}")
            continue
        label, rain, b1, b2, b3, b4 = EXPERIMENTS[exp_key]
        is_last = (exp_key == exps_to_run[-1])
        df = run_experiment(
            exp_name=f"{exp_key}_{label}",
            include_rainfall=rain,
            b1=b1, b2=b2, b3=b3, b4=b4,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            save_models=args.save_models and is_last,
        )
        all_results.append(df)

    # Combine and save results
    results_df = pd.concat(all_results, ignore_index=True)

    output_dir = PROJECT_ROOT / "reports" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment_progressive_features.xlsx"
    results_df.to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary: best per horizon per experiment
    print(f"\n{'='*80}")
    print("SUMMARY: Best model per horizon per experiment")
    print(f"{'='*80}")
    print(f"{'Experiment':<25s} {'H':>2s} {'Model':<25s} {'NSE':>7s} {'RMSE':>7s} {'nRMSE%':>7s}")
    print("-" * 80)

    for exp_key in exps_to_run:
        label = EXPERIMENTS[exp_key][0]
        full_name = f"{exp_key}_{label}"
        exp_df = results_df[results_df["experiment"] == full_name]
        for h in sorted(exp_df["horizon"].unique()):
            h_df = exp_df[exp_df["horizon"] == h]
            best_row = h_df.loc[h_df["test_NSE"].idxmax()]
            print(
                f"{full_name:<25s} {h:>2d} {best_row['model']:<25s} "
                f"{best_row['test_NSE']:>7.4f} {best_row['test_RMSE']:>7.4f} "
                f"{best_row['nRMSE_mean_%']:>6.1f}%"
            )


if __name__ == "__main__":
    main()
