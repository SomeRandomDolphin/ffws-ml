"""Stacking ensemble training for Dhompo flood forecasting.

Usage
-----
    python training/train_stacking.py
    python training/train_stacking.py --horizon 4 5

Implements a two-level stacking ensemble:
  Level-0: XGBoost, LightGBM, Ridge, Gradient Boosting (base learners)
  Level-1: Ridge regression as meta-learner

Cross-validation on training set generates level-0 out-of-fold predictions
to train the meta-learner, preventing data leakage.

Reference: Fan et al. (2020) — stacking LSTM + tree-based >> single model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
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
from dhompo.models.sklearn_models import HORIZONS, get_model_definitions
from training.evaluate import calc_metrics


def _get_base_learners(model_cfg: dict) -> dict[str, tuple]:
    """Select base learners for stacking from the model registry.

    Uses tree-based models (no scaling needed) + Ridge (scaled) as base learners.
    """
    all_models = get_model_definitions(model_cfg)
    # Select specific models for stacking
    base_names = ["XGBoost", "Gradient Boosting", "Ridge", "Random Forest"]

    # Add LightGBM and CatBoost if available
    if "LightGBM" in all_models:
        base_names.append("LightGBM")
    if "CatBoost" in all_models:
        base_names.append("CatBoost")

    return {name: all_models[name] for name in base_names if name in all_models}


def generate_oof_predictions(
    base_learners: dict[str, tuple],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_splits: int = 5,
) -> tuple[np.ndarray, list[object]]:
    """Generate out-of-fold predictions for meta-learner training.

    Parameters
    ----------
    base_learners:
        {name: (model_template, use_scaled)} from model registry.
    X_train:
        Training feature matrix (numpy array).
    y_train:
        Training target values.
    feature_names:
        Column names for the features.
    n_splits:
        Number of TimeSeriesSplit folds.

    Returns
    -------
    oof_predictions:
        Array of shape (n_train, n_base_learners) with OOF predictions.
    trained_models:
        List of fitted base learners (trained on full training set).
    """
    n_samples = len(X_train)
    n_models = len(base_learners)
    oof_preds = np.full((n_samples, n_models), np.nan)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Generate OOF predictions
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]

        for model_idx, (name, (model_template, use_scaled)) in enumerate(base_learners.items()):
            model = clone(model_template)

            if use_scaled:
                scaler = StandardScaler()
                X_tr_fit = scaler.fit_transform(X_tr)
                X_val_fit = scaler.transform(X_val)
            else:
                X_tr_fit, X_val_fit = X_tr, X_val

            model.fit(X_tr_fit, y_tr)
            oof_preds[val_idx, model_idx] = model.predict(X_val_fit)

    # Train base learners on full training set for test-time inference
    trained_models = []
    scalers = []
    for name, (model_template, use_scaled) in base_learners.items():
        model = clone(model_template)
        if use_scaled:
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(X_train)
            scalers.append(scaler)
        else:
            X_fit = X_train
            scalers.append(None)
        model.fit(X_fit, y_train)
        trained_models.append(model)

    return oof_preds, trained_models, scalers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stacking ensemble")
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
        default="dhompo_stacking",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        nargs="+",
        default=None,
        help="Horizons to train (e.g. --horizon 4 5). Defaults to all.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of TimeSeriesSplit folds for OOF generation",
    )
    parser.add_argument(
        "--meta-alpha",
        type=float,
        default=1.0,
        help="Ridge alpha for meta-learner",
    )
    parser.add_argument(
        "--best-params-dir",
        default=None,
        help="Directory with tuned params from tune_optuna.py. "
             "If provided, uses tuned hyperparameters for base learners.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = load_yaml_config(args.config)
    train_cfg = load_yaml_config(args.train_config)

    serving_cfg = load_serving_config()
    mlflow_uri = serving_cfg.get("mlflow_uri", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.experiment)

    train_split = train_cfg.get("train_split", 0.8)
    horizons = args.horizon or train_cfg.get("horizons", HORIZONS)
    horizon_steps = {int(h): int(h) * 2 for h in horizons}
    target_station = train_cfg.get("target_station", "Dhompo")
    include_rainfall = train_cfg.get("include_rainfall", False)
    data_sources = train_cfg.get("data_sources", None)

    feature_cfg = train_cfg.get("features", {})
    use_travel_time_lags = feature_cfg.get("travel_time_lags", False)
    use_cumulative_rainfall = feature_cfg.get("cumulative_rainfall", False)
    use_interaction_features = feature_cfg.get("interaction_features", False)
    use_seasonal_features = feature_cfg.get("seasonal_features", False)

    extra_columns = [RAINFALL_COLUMN] if include_rainfall else None

    # --- Data loading (same as train_sklearn.py) ---
    if data_sources is not None:
        print("Loading multi-source data...")
        source_paths = {}
        for src in data_sources:
            resolved = resolve_path_from_config(args.train_config, src["path"])
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
            use_travel_time_lags=use_travel_time_lags,
            use_cumulative_rainfall=use_cumulative_rainfall,
            use_interaction_features=use_interaction_features,
            use_seasonal_features=use_seasonal_features,
        )

        y_parts = []
        for seg in segments:
            y_seg = build_targets(seg.df, horizons, horizon_steps, target=target_station)
            y_combined = pd.concat({h: s for h, s in y_seg.items()}, axis=1)
            y_parts.append(y_combined)
        y_all = pd.concat(y_parts, axis=0)
        y_horizons = {h: y_all[h] for h in horizons}

        X_full, y_horizons = align_features_targets(X_features, y_horizons)

        seg_2023_start = segments[1].df.index.min()
        idx_2023 = X_full.index >= seg_2023_start
        n_2023 = idx_2023.sum()
        n_2023_train = int(n_2023 * train_split)
        split_idx = (~idx_2023).sum() + n_2023_train
    else:
        data_path = resolve_path_from_config(
            args.train_config, train_cfg.get("data_path")
        )
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

    X_train_raw = X_full.iloc[:split_idx]
    X_test_raw = X_full.iloc[split_idx:]

    print(f"Data: {X_full.shape} | Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")

    base_learners = _get_base_learners(model_cfg)
    base_names = list(base_learners.keys())
    print(f"Base learners: {base_names}")

    best_per_horizon: dict[int, tuple[str, float]] = {}

    for h in horizons:
        y_train = y_horizons[h].iloc[:split_idx].values
        y_test = y_horizons[h].iloc[split_idx:].values

        print(f"\n=== Horizon +{h}h — Stacking Ensemble ===")

        # Step 1: Generate OOF predictions for meta-learner training
        print("  Generating out-of-fold predictions...")
        oof_preds, trained_models, trained_scalers = generate_oof_predictions(
            base_learners,
            X_train_raw.values,
            y_train,
            feature_names=list(X_train_raw.columns),
            n_splits=args.cv_folds,
        )

        # Remove rows where OOF has NaN (first fold's training rows)
        valid_mask = ~np.isnan(oof_preds).any(axis=1)
        oof_valid = oof_preds[valid_mask]
        y_train_valid = y_train[valid_mask]

        # Step 2: Train meta-learner on OOF predictions
        meta_learner = Ridge(alpha=args.meta_alpha)
        meta_scaler = StandardScaler()
        oof_scaled = meta_scaler.fit_transform(oof_valid)
        meta_learner.fit(oof_scaled, y_train_valid)

        # Step 3: Generate test predictions from base learners
        test_base_preds = np.zeros((len(X_test_raw), len(base_names)))
        for i, (name, (_, use_scaled)) in enumerate(base_learners.items()):
            if trained_scalers[i] is not None:
                X_test_fit = trained_scalers[i].transform(X_test_raw.values)
            else:
                X_test_fit = X_test_raw.values
            test_base_preds[:, i] = trained_models[i].predict(X_test_fit)

        # Step 4: Meta-learner final prediction
        test_meta_input = meta_scaler.transform(test_base_preds)
        y_pred_stack = meta_learner.predict(test_meta_input)

        # Evaluate
        stack_metrics = calc_metrics(y_test, y_pred_stack)
        nse = stack_metrics["NSE"]
        rmse = stack_metrics["RMSE"]
        print(f"  Stacking NSE={nse:.4f}  RMSE={rmse:.4f}")

        # Also evaluate individual base learners on test set for comparison
        print("  Base learner comparison:")
        for i, name in enumerate(base_names):
            base_metrics = calc_metrics(y_test, test_base_preds[:, i])
            print(f"    {name:25s} NSE={base_metrics['NSE']:.4f}")

        # Meta-learner weights
        print(f"  Meta-learner weights: {dict(zip(base_names, meta_learner.coef_.round(3)))}")

        # Log to MLflow
        with mlflow.start_run(run_name=f"h{h}_stacking"):
            mlflow.log_params({
                "algorithm": "Stacking",
                "horizon": h,
                "base_learners": ",".join(base_names),
                "meta_learner": "Ridge",
                "meta_alpha": args.meta_alpha,
                "cv_folds": args.cv_folds,
                "n_features": X_train_raw.shape[1],
                "train_size": len(X_train_raw),
                "test_size": len(X_test_raw),
            })

            # Train metrics (OOF-based)
            oof_meta_pred = meta_learner.predict(oof_scaled)
            train_metrics = calc_metrics(y_train_valid, oof_meta_pred)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in stack_metrics.items()})

            # Log meta-learner weights
            for name, w in zip(base_names, meta_learner.coef_):
                mlflow.log_metric(f"weight_{name.lower().replace(' ', '_')}", w)

        best_per_horizon[h] = ("Stacking", nse)

    # Summary
    print(f"\n{'='*60}")
    print("STACKING RESULTS")
    print(f"{'='*60}")
    for h, (name, nse) in best_per_horizon.items():
        grade = "Very Good" if nse >= 0.75 else "Good" if nse >= 0.65 else "Satisfactory"
        print(f"  h{h}: NSE={nse:.4f} ({grade})")


if __name__ == "__main__":
    main()
