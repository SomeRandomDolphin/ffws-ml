"""Train 30 sklearn/XGBoost models and log them to MLflow.

Usage
-----
    python training/train_sklearn.py
    python training/train_sklearn.py --config configs/sklearn_model.yaml
    python training/train_sklearn.py --config configs/sklearn_model.yaml --experiment my_exp

Workflow
--------
1. Load data from data/data-clean.csv
2. Build 160-feature matrix
3. Temporal split 80/20
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
    build_forecast_features,
    build_targets,
)
from dhompo.data.loader import UPSTREAM_STATIONS, load_data
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
    data_path = args.data or resolve_path_from_config(
        args.train_config, train_cfg.get("data_path")
    )

    # Data
    print("Loading data...")
    df = load_data(data_path)
    X_features = build_forecast_features(df, UPSTREAM_STATIONS, target=target_station)
    y_horizons = build_targets(df, horizons, horizon_steps, target=target_station)
    X_full, y_horizons = align_features_targets(X_features, y_horizons)

    split_idx = int(len(X_full) * train_split)
    X_train_raw = X_full.iloc[:split_idx]
    X_test_raw = X_full.iloc[split_idx:]

    print(
        f"Data: {X_full.shape} | Train: {len(X_train_raw)} | Test: {len(X_test_raw)}"
    )

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
