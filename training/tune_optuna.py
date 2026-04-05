"""Optuna-based hyperparameter tuning for Dhompo flood forecasting models.

Usage
-----
    python training/tune_optuna.py
    python training/tune_optuna.py --model xgboost --horizon 4
    python training/tune_optuna.py --all-models --all-horizons

Performs Bayesian optimization using Optuna with TimeSeriesSplit cross-validation.
Best hyperparameters are saved to configs/best_params/ and logged to MLflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from dhompo.config import load_yaml_config, resolve_path_from_config
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
from dhompo.models.sklearn_models import HORIZONS
from training.evaluate import calc_metrics

# Optional imports
try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


# ---------------------------------------------------------------------------
# Model-specific search spaces
# ---------------------------------------------------------------------------

def _suggest_xgboost(trial: optuna.Trial) -> dict:
    from xgboost import XGBRegressor
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    return params, XGBRegressor(**params), False


def _suggest_lightgbm(trial: optuna.Trial) -> dict:
    from lightgbm import LGBMRegressor
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    return params, LGBMRegressor(**params), False


def _suggest_catboost(trial: optuna.Trial) -> dict:
    from catboost import CatBoostRegressor
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000, step=50),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_seed": 42,
        "verbose": 0,
    }
    return params, CatBoostRegressor(**params), False


def _suggest_gradient_boosting(trial: optuna.Trial) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": 42,
    }
    return params, GradientBoostingRegressor(**params), False


def _suggest_random_forest(trial: optuna.Trial) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "random_state": 42,
        "n_jobs": -1,
    }
    return params, RandomForestRegressor(**params), False


def _suggest_ridge(trial: optuna.Trial) -> dict:
    from sklearn.linear_model import Ridge
    params = {
        "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
    }
    return params, Ridge(**params), True


def _suggest_lasso(trial: optuna.Trial) -> dict:
    from sklearn.linear_model import Lasso
    params = {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "max_iter": 10000,
    }
    return params, Lasso(**params), True


def _suggest_elasticnet(trial: optuna.Trial) -> dict:
    from sklearn.linear_model import ElasticNet
    params = {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
        "max_iter": 10000,
    }
    return params, ElasticNet(**params), True


MODEL_SUGGEST = {
    "xgboost": _suggest_xgboost,
    "lightgbm": _suggest_lightgbm,
    "catboost": _suggest_catboost,
    "gradient_boosting": _suggest_gradient_boosting,
    "random_forest": _suggest_random_forest,
    "ridge": _suggest_ridge,
    "lasso": _suggest_lasso,
    "elasticnet": _suggest_elasticnet,
}


# ---------------------------------------------------------------------------
# Core tuning logic
# ---------------------------------------------------------------------------

def create_objective(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> callable:
    """Create an Optuna objective function for a given model and data."""

    suggest_fn = MODEL_SUGGEST[model_name]

    def objective(trial: optuna.Trial) -> float:
        params, model, use_scaled = suggest_fn(trial)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        nse_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            if use_scaled:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_val = scaler.transform(X_val)

            model_clone = model.__class__(**{
                k: v for k, v in model.get_params().items()
                if k != "verbose"  # avoid issues with CatBoost
            })
            # Re-set verbose for CatBoost
            if hasattr(model_clone, "verbose"):
                model_clone.set_params(verbose=0)

            model_clone.fit(X_tr, y_tr)
            y_pred = model_clone.predict(X_val)

            metrics = calc_metrics(y_val, y_pred)
            nse_scores.append(metrics["NSE"])

            # Optuna pruning: report intermediate result
            trial.report(np.mean(nse_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(nse_scores)

    return objective


def load_data_for_tuning(
    train_cfg: dict,
    args: argparse.Namespace,
) -> tuple:
    """Load and prepare data, returning (X_full, y_horizons, feature_names)."""
    train_split = train_cfg.get("train_split", 0.8)
    horizons = train_cfg.get("horizons", HORIZONS)
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

    if data_sources is not None:
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
    else:
        data_path = resolve_path_from_config(
            args.train_config, train_cfg.get("data_path")
        )
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
    return X_full, y_horizons


# Need pandas for concat in load_data_for_tuning
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument(
        "--train-config",
        default="configs/training.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=list(MODEL_SUGGEST.keys()),
        help="Model to tune",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Single horizon to tune (1-5). If not set, tunes all.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Tune all available models",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override number of trials from config",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/best_params",
        help="Directory to save best parameters",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml_config(args.train_config)
    tuning_cfg = train_cfg.get("tuning", {})

    n_trials = args.n_trials or tuning_cfg.get("n_trials", 100)
    n_splits = tuning_cfg.get("cv_folds", 5)
    timeout = tuning_cfg.get("timeout", 3600)

    print("Loading data...")
    X_full, y_horizons = load_data_for_tuning(train_cfg, args)
    print(f"Data shape: {X_full.shape}")

    models_to_tune = list(MODEL_SUGGEST.keys()) if args.all_models else [args.model]
    horizons_to_tune = [args.horizon] if args.horizon else train_cfg.get("horizons", HORIZONS)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for model_name in models_to_tune:
        for h in horizons_to_tune:
            study_name = f"{model_name}_h{h}"
            print(f"\n{'='*60}")
            print(f"Tuning {model_name} for horizon +{h}h")
            print(f"{'='*60}")

            y = y_horizons[h].values
            X = X_full.values

            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
            )

            objective = create_objective(model_name, X, y, n_splits=n_splits)

            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
            )

            best = study.best_trial
            print(f"\nBest NSE: {best.value:.4f}")
            print(f"Best params: {best.params}")

            # Save best params
            result_entry = {
                "model": model_name,
                "horizon": h,
                "best_nse_cv": best.value,
                "best_params": best.params,
                "n_trials": len(study.trials),
            }
            results[study_name] = result_entry

            param_file = output_dir / f"{study_name}.json"
            with open(param_file, "w") as f:
                json.dump(result_entry, f, indent=2)
            print(f"Saved: {param_file}")

    # Summary
    print(f"\n{'='*60}")
    print("TUNING SUMMARY")
    print(f"{'='*60}")
    for name, res in results.items():
        print(f"  {name}: NSE={res['best_nse_cv']:.4f} ({res['n_trials']} trials)")

    # Save combined results
    combined_file = output_dir / "all_results.json"
    with open(combined_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to: {combined_file}")


if __name__ == "__main__":
    main()
