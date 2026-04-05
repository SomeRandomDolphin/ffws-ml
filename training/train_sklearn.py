"""Train 30 sklearn/XGBoost models and log them to MLflow.

Usage
-----
    python training/train_sklearn.py
    python training/train_sklearn.py --config configs/sklearn_model.yaml
    python training/train_sklearn.py --experiment dhompo_combined
    python training/train_sklearn.py --experiment dhompo_2022only --single-source 2022_clean

Workflow
--------
1. Load data (single or multi-source with segment-aware features)
2. Build feature matrix (~160 base + ~13 per extra column)
3. Temporal split: all 2022 + 80% 2023 for train, 20% latest 2023 for test
4. For each horizon × algorithm:
   - Train model
   - Log params + metrics (NSE, RMSE, MAE, PBIAS, R²) to MLflow
   - Register best model per horizon in MLflow Model Registry
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow running as `python training/...py` without package install.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import skops.io
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from dhompo.config import load_serving_config, load_yaml_config, resolve_path_from_config
from dhompo.data.features import (
    align_features_targets,
    build_features_from_segments,
    build_forecast_features,
    build_targets,
)
from dhompo.data.loader import (
    RAINFALL_COLUMN,
    UPSTREAM_STATIONS,
    load_combined_data,
    load_data,
)
from dhompo.models.sklearn_models import HORIZON_STEPS, HORIZONS, get_model_definitions
from training.evaluate import calc_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dhompo sklearn models")
    parser.add_argument(
        "--config",
        default="configs/sklearn_model.yaml",
        help="Path to sklearn hyperparameter config YAML",
    )
    parser.add_argument(
        "--train-config",
        default="configs/training.yaml",
        help="Path to training split config YAML",
    )
    parser.add_argument(
        "--experiment",
        default="dhompo_sklearn",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to data-clean.csv (defaults to data/data-clean.csv)",
    )
    parser.add_argument(
        "--single-source",
        default=None,
        dest="single_source",
        help="Use only one data source by label (e.g. '2022_clean'). "
             "If not set, uses all sources from config.",
    )
    return parser.parse_args()


def _skops_trusted_types(model: object) -> list[str] | None:
    trusted = skops.io.get_untrusted_types(data=skops.io.dumps(model))
    return trusted or None


def _pip_requirements_for_model(model: object) -> list[str]:
    requirements = list(
        mlflow.sklearn.get_default_pip_requirements(include_skops=True)
    )
    if type(model).__module__.startswith("xgboost"):
        try:
            requirements.append(f"xgboost=={version('xgboost')}")
        except PackageNotFoundError:
            requirements.append("xgboost")
    if type(model).__module__.startswith("lightgbm"):
        try:
            requirements.append(f"lightgbm=={version('lightgbm')}")
        except PackageNotFoundError:
            requirements.append("lightgbm")
    if type(model).__module__.startswith("catboost"):
        try:
            requirements.append(f"catboost=={version('catboost')}")
        except PackageNotFoundError:
            requirements.append("catboost")
    return requirements


def main() -> None:
    args = parse_args()
    model_cfg = load_yaml_config(args.config)
    train_cfg = load_yaml_config(args.train_config)

    serving_cfg = load_serving_config()
    mlflow_uri = serving_cfg.get("mlflow_uri", "http://localhost:5000")

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.experiment)

    train_split = train_cfg.get("train_split", 0.8)
    horizons = train_cfg.get("horizons", HORIZONS)
    horizon_steps = {int(h): int(h) * 2 for h in horizons}
    target_station = train_cfg.get("target_station", "Dhompo")
    include_rainfall = train_cfg.get("include_rainfall", False)
    data_sources = train_cfg.get("data_sources", None)

    # Enhanced feature flags
    feature_cfg = train_cfg.get("features", {})
    use_travel_time_lags = feature_cfg.get("travel_time_lags", False)
    use_cumulative_rainfall = feature_cfg.get("cumulative_rainfall", False)
    use_interaction_features = feature_cfg.get("interaction_features", False)
    use_seasonal_features = feature_cfg.get("seasonal_features", False)

    extra_columns = [RAINFALL_COLUMN] if include_rainfall else None

    # --- Data loading ---
    use_multi_source = data_sources is not None and args.single_source is None and args.data is None
    data_source_labels: list[str] = []

    if use_multi_source:
        print("Loading multi-source data (segment-aware)...")
        source_paths = {}
        for src in data_sources:
            resolved = resolve_path_from_config(args.train_config, src["path"])
            source_paths[src["label"]] = resolved
            data_source_labels.append(src["label"])

        segments = load_combined_data(
            clean_path=source_paths["2022_clean"],
            generated_path=source_paths["2023_generated"],
        )

        # Build features per segment (prevents cross-gap contamination)
        X_features = build_features_from_segments(
            segments,
            upstream_stations=UPSTREAM_STATIONS,
            target=target_station,
            extra_columns=extra_columns,
            use_travel_time_lags=use_travel_time_lags,
            use_cumulative_rainfall=use_cumulative_rainfall,
            use_interaction_features=use_interaction_features,
            use_seasonal_features=use_seasonal_features,
        )

        # Build targets: need a combined df for shifting
        # Targets must also be built per segment to avoid cross-gap issues
        y_parts = []
        for seg in segments:
            df_seg = seg.df
            y_seg = build_targets(df_seg, horizons, horizon_steps, target=target_station)
            # Concat targets across horizons into a temp DataFrame for alignment
            y_combined = pd.concat(
                {h: s for h, s in y_seg.items()}, axis=1
            )
            y_parts.append(y_combined)
        y_all = pd.concat(y_parts, axis=0)
        y_horizons = {h: y_all[h] for h in horizons}

        X_full, y_horizons = align_features_targets(X_features, y_horizons)

        # Split strategy: all 2022 + 80% of 2023 → train; 20% latest 2023 → test
        seg_2023_start = segments[1].df.index.min()
        idx_2023 = X_full.index >= seg_2023_start
        n_2023 = idx_2023.sum()
        n_2023_train = int(n_2023 * train_split)
        # All 2022 rows + first 80% of 2023
        split_idx = (~idx_2023).sum() + n_2023_train
        split_strategy = f"all_2022 + {train_split*100:.0f}% of 2023 train, rest test"
    else:
        # Single source mode
        if args.single_source:
            # Find the matching source from config
            src_match = None
            if data_sources:
                for src in data_sources:
                    if src["label"] == args.single_source:
                        src_match = src
                        break
            if src_match and src_match["format"] == "csv":
                data_path = resolve_path_from_config(args.train_config, src_match["path"])
                data_source_labels = [args.single_source]
            else:
                print(f"ERROR: single-source '{args.single_source}' not found or not CSV")
                sys.exit(1)
        else:
            data_path = args.data or resolve_path_from_config(
                args.train_config, train_cfg.get("data_path")
            )
            data_source_labels = ["single"]

        print("Loading data...")
        df = load_data(data_path)
        X_features = build_forecast_features(
            df, UPSTREAM_STATIONS, target=target_station,
            extra_columns=extra_columns,
            use_travel_time_lags=use_travel_time_lags,
            use_cumulative_rainfall=use_cumulative_rainfall,
            use_interaction_features=use_interaction_features,
            use_seasonal_features=use_seasonal_features,
        )
        y_horizons = build_targets(df, horizons, horizon_steps, target=target_station)
        X_full, y_horizons = align_features_targets(X_features, y_horizons)

        split_idx = int(len(X_full) * train_split)
        split_strategy = f"{train_split*100:.0f}/{(1-train_split)*100:.0f} temporal"

    X_train_raw = X_full.iloc[:split_idx]
    X_test_raw = X_full.iloc[split_idx:]

    print(
        f"Data: {X_full.shape} | Train: {len(X_train_raw)} | Test: {len(X_test_raw)}"
    )
    print(f"Split: {split_strategy}")
    print(f"Sources: {data_source_labels}")
    if extra_columns:
        print(f"Extra features: {extra_columns}")

    # Scale once per split (shared across models that need it)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=X_train_raw.columns,
        index=X_train_raw.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X_test_raw.columns,
        index=X_test_raw.index,
    )

    model_defs = get_model_definitions(model_cfg)

    best_per_horizon: dict[int, tuple[str, float, object]] = {}

    # Training Loop
    for h in horizons:
        y = y_horizons[h]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        print(f"\n=== Horizon +{h}h ===")
        for name, (model_template, use_scaled) in model_defs.items():
            Xtr = X_train_s if use_scaled else X_train_raw
            Xte = X_test_s if use_scaled else X_test_raw

            with mlflow.start_run(run_name=f"h{h}_{name.lower().replace(' ', '_')}"):
                model = clone(model_template)
                model.fit(Xtr, y_train)

                train_metrics = calc_metrics(y_train.values, model.predict(Xtr))
                test_metrics = calc_metrics(y_test.values, model.predict(Xte))

                mlflow.log_params(
                    {
                        "algorithm": name,
                        "horizon": h,
                        "train_size": len(Xtr),
                        "test_size": len(Xte),
                        "n_features": Xtr.shape[1],
                        "use_scaled": use_scaled,
                        "data_sources": ",".join(data_source_labels),
                        "split_strategy": split_strategy,
                        "include_rainfall": include_rainfall,
                        "travel_time_lags": use_travel_time_lags,
                        "cumulative_rainfall": use_cumulative_rainfall,
                        "interaction_features": use_interaction_features,
                        "seasonal_features": use_seasonal_features,
                    }
                )
                mlflow.log_metrics(
                    {f"train_{k}": v for k, v in train_metrics.items()}
                )
                mlflow.log_metrics(
                    {f"test_{k}": v for k, v in test_metrics.items()}
                )

                registered_name = f"dhompo_h{h}"
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
                    skops_trusted_types=_skops_trusted_types(model),
                    pip_requirements=_pip_requirements_for_model(model),
                    registered_model_name=registered_name,
                )

                nse = test_metrics["NSE"]
                rmse = test_metrics["RMSE"]
                print(f"  {name:30s}  NSE={nse:.4f}  RMSE={rmse:.4f}")

                # Track best per horizon
                if h not in best_per_horizon or nse > best_per_horizon[h][1]:
                    best_per_horizon[h] = (name, nse, model)

    # Summary
    print("\n=== Best Models ===")
    for h, (name, nse, _) in best_per_horizon.items():
        grade = "✓" if nse >= 0.95 else "!"
        print(f"  h{h}: {name} (NSE={nse:.4f}) {grade}")

    print("\nDone. View results: make mlflow-ui")


if __name__ == "__main__":
    main()
