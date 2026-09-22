"""Train baseline sklearn models for the urban water-level dataset.

Usage
-----
    python training/surabaya/train_urban_sklearn.py --models ridge
    python training/surabaya/train_urban_sklearn.py --models gradient_boosting,xgboost

The pipeline is intentionally independent from ``training/dhompo/train_sklearn.py``
because the existing script targets the Dhompo station schema.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from dhompo.config import load_yaml_config, resolve_path_from_config
from dhompo.data.urban_features import (
    align_urban_features_targets,
    build_urban_delta_targets,
    build_urban_forecast_features,
    build_urban_targets,
)
from dhompo.data.urban_loader import preprocess_urban_wide_data
from dhompo.models.sklearn_models import get_model_definitions
from training.evaluate import calc_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train urban water-level baselines")
    parser.add_argument(
        "--config",
        default="configs/surabaya/urban_water_level.yaml",
        help="Urban preprocessing/training config",
    )
    parser.add_argument(
        "--model-config",
        default="configs/shared/sklearn_model.yaml",
        help="Shared sklearn hyperparameter config",
    )
    parser.add_argument(
        "--models",
        default="ridge",
        help=(
            "Comma-separated model keys, e.g. ridge,lasso,gradient_boosting,xgboost. "
            "Use 'all' to train every available sklearn model definition."
        ),
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional CSV path override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build features/targets and print split info without fitting models.",
    )
    return parser.parse_args()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _selected_model_defs(
    model_defs: dict[str, tuple],
    selected: str,
) -> dict[str, tuple]:
    by_slug = {_slug(name): (name, definition) for name, definition in model_defs.items()}
    if selected.strip().lower() == "all":
        return model_defs

    out: dict[str, tuple] = {}
    for raw_key in selected.split(","):
        key = raw_key.strip().lower()
        if not key:
            continue
        if key not in by_slug:
            available = ", ".join(sorted(by_slug))
            raise ValueError(f"Unknown model key '{key}'. Available: {available}")
        name, definition = by_slug[key]
        out[name] = definition
    if not out:
        raise ValueError("No models selected.")
    return out


def _rolling_windows(config: dict[str, Any]) -> list[tuple[int, str]]:
    windows = config.get("features", {}).get("rolling_windows", [])
    return [(int(w["steps"]), str(w["label"])) for w in windows]


def _json_safe_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}


def _walk_forward_scores(
    template,
    use_scaled: bool,
    X: pd.DataFrame,
    y: pd.Series,
    current: pd.Series,
    future_level: pd.Series,
    target_mode: str,
    n_splits: int,
    gap: int,
) -> tuple[list[dict[str, float]], np.ndarray]:
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores: list[dict[str, float]] = []
    absolute_errors: list[float] = []
    for train_idx, validation_idx in splitter.split(X):
        X_train = X.iloc[train_idx]
        X_validation = X.iloc[validation_idx]
        if use_scaled:
            scaler = StandardScaler().fit(X_train)
            X_train = pd.DataFrame(
                scaler.transform(X_train), index=X_train.index, columns=X_train.columns
            )
            X_validation = pd.DataFrame(
                scaler.transform(X_validation),
                index=X_validation.index,
                columns=X_validation.columns,
            )
        model = clone(template).fit(X_train, y.iloc[train_idx])
        prediction = model.predict(X_validation)
        if target_mode in {"delta", "persistence_residual"}:
            prediction = current.iloc[validation_idx].to_numpy() + prediction
        truth = future_level.iloc[validation_idx].to_numpy()
        scores.append(_json_safe_metrics(calc_metrics(truth, prediction)))
        absolute_errors.extend(np.abs(truth - prediction).tolist())
    return scores, np.asarray(absolute_errors, dtype=float)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    model_cfg = load_yaml_config(args.model_config)

    data = preprocess_urban_wide_data(data_path=args.data, config_path=args.config)

    feature_cfg = config.get("features", {})
    lag_steps = [int(v) for v in feature_cfg.get("lag_steps", [1, 2, 3])]
    rolling_windows = _rolling_windows(config)
    include_quality_flags = bool(feature_cfg.get("include_quality_flags", True))

    X_features = build_urban_forecast_features(
        data.values,
        feature_columns=data.feature_columns,
        quality_flags=data.quality_flags,
        include_quality_flags=include_quality_flags,
        lag_steps=lag_steps,
        rolling_windows=rolling_windows,
        pump_control=feature_cfg.get("pump_control"),
    )
    horizons = [int(h) for h in config.get("horizons", [1, 2, 3, 4, 5])]
    target_mode = str(config.get("target_mode", "level")).lower()
    future_targets = build_urban_targets(
        data.modeling_canonical,
        target_column=data.target_column,
        horizons=horizons,
    )
    if target_mode in {"delta", "persistence_residual"}:
        y_horizons = build_urban_delta_targets(
            data.modeling_canonical,
            current_values=data.values,
            target_column=data.target_column,
            horizons=horizons,
        )
    elif target_mode == "level":
        y_horizons = future_targets
    else:
        raise ValueError(
            "target_mode must be 'level', 'delta', or 'persistence_residual'."
        )
    X_full, y_horizons = align_urban_features_targets(X_features, y_horizons)
    future_targets = {h: y.loc[X_full.index] for h, y in future_targets.items()}
    current_target = data.values[data.target_column].loc[X_full.index]

    train_split = float(config.get("train_split", 0.8))
    split_idx = int(len(X_full) * train_split)
    if split_idx <= 0 or split_idx >= len(X_full):
        raise ValueError(
            f"Invalid train/test split after alignment: rows={len(X_full)}, split_idx={split_idx}"
        )

    X_train_raw = X_full.iloc[:split_idx]
    X_test_raw = X_full.iloc[split_idx:]

    print(f"Target: {data.target_column}")
    print(f"Target mode: {target_mode}")
    print(f"Features: {X_full.shape[1]} columns from {len(data.feature_columns)} signals")
    print(f"Rows after alignment: {len(X_full)} | train={len(X_train_raw)} test={len(X_test_raw)}")
    print(f"Range: {X_full.index.min()} -> {X_full.index.max()}")

    required_signals = [str(v) for v in feature_cfg.get("required_signals", [])]
    missing_required = [signal for signal in required_signals if signal not in data.feature_columns]
    if missing_required:
        raise ValueError(
            "Required pump-aware signals are unavailable after coverage filtering: "
            + ", ".join(missing_required)
        )

    if args.dry_run:
        return

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

    all_model_defs = get_model_definitions(model_cfg)
    model_defs = _selected_model_defs(all_model_defs, args.models)

    output_dir_cfg = config.get("models_output_dir", "../../models/surabaya/urban")
    output_dir = resolve_path_from_config(args.config, output_dir_cfg)
    if output_dir is None:
        raise ValueError("models_output_dir is required.")
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, output_dir / "standard_scaler_global.pkl")

    metadata: dict[str, Any] = {
        "target_column": data.target_column,
        "target_mode": target_mode,
        "model_version": "urban_file_v2",
        "feature_columns": list(X_full.columns),
        "source_signals": data.feature_columns,
        "horizons": horizons,
        "train_split": train_split,
        "train_rows": len(X_train_raw),
        "test_rows": len(X_test_raw),
        "start": X_full.index.min().isoformat(),
        "end": X_full.index.max().isoformat(),
        "models": {},
        "coverage": data.coverage.to_dict(orient="records"),
        "outlier_summary": data.outlier_summary.to_dict(orient="records"),
        "uses_pump_telemetry": bool(feature_cfg.get("pump_control", {}).get("activity_columns")),
        "validation": {
            "strategy": "expanding_window",
            "n_splits": int(config.get("validation", {}).get("n_splits", 3)),
            "gap_steps": int(config.get("validation", {}).get("gap_steps", max(horizons) * 2)),
        },
        "prediction_intervals": {},
        "baselines": {},
    }

    best_per_horizon: dict[int, tuple[str, float, Path]] = {}
    interval_errors: dict[tuple[int, str], np.ndarray] = {}
    validation_cfg = config.get("validation", {})
    validation_splits = int(validation_cfg.get("n_splits", 3))
    validation_gap = int(validation_cfg.get("gap_steps", max(horizons) * 2))
    for h in horizons:
        y = y_horizons[h]
        y_train = y.iloc[:split_idx]
        level_train = future_targets[h].iloc[:split_idx]
        level_test = future_targets[h].iloc[split_idx:]
        current_train = current_target.iloc[:split_idx]
        current_test = current_target.iloc[split_idx:]
        persistence_metrics = calc_metrics(
            level_test.to_numpy(), current_test.to_numpy()
        )
        metadata["baselines"][f"h{h}"] = {
            "persistence": _json_safe_metrics(persistence_metrics),
        }
        print(f"\n=== Horizon +{h}h ===")

        for model_name, (template, use_scaled) in model_defs.items():
            fold_scores, absolute_errors = _walk_forward_scores(
                template,
                use_scaled,
                X_train_raw,
                y_train,
                current_train,
                level_train,
                target_mode,
                validation_splits,
                validation_gap,
            )
            model = clone(template)
            Xtr = X_train_s if use_scaled else X_train_raw
            Xte = X_test_s if use_scaled else X_test_raw
            model.fit(Xtr, y_train)

            train_pred_raw = model.predict(Xtr)
            test_pred_raw = model.predict(Xte)
            if target_mode in {"delta", "persistence_residual"}:
                train_pred_eval = current_train.values + train_pred_raw
                test_pred_eval = current_test.values + test_pred_raw
            else:
                train_pred_eval = train_pred_raw
                test_pred_eval = test_pred_raw

            train_metrics = calc_metrics(level_train.values, train_pred_eval)
            test_metrics = calc_metrics(level_test.values, test_pred_eval)
            model_key = _slug(model_name)
            model_path = output_dir / f"urban_{model_key}_h{h}.pkl"
            joblib.dump(model, model_path)

            metadata["models"][f"h{h}:{model_key}"] = {
                "path": model_path.name,
                "use_scaled": bool(use_scaled),
                "train_metrics": _json_safe_metrics(train_metrics),
                "test_metrics": _json_safe_metrics(test_metrics),
                "walk_forward_metrics": fold_scores,
                "beats_persistence_on_test": (
                    float(test_metrics["RMSE"]) < float(persistence_metrics["RMSE"])
                ),
            }
            interval_errors[(h, model_key)] = absolute_errors

            cv_rmse = float(np.mean([score["RMSE"] for score in fold_scores]))
            rmse = float(test_metrics["RMSE"])
            print(
                f"  {model_name:24s} CV_RMSE={cv_rmse:.4f} "
                f"TEST_RMSE={rmse:.4f} PERSISTENCE_RMSE={persistence_metrics['RMSE']:.4f}"
            )
            if h not in best_per_horizon or cv_rmse < best_per_horizon[h][1]:
                best_per_horizon[h] = (model_key, cv_rmse, model_path)

    metadata["best_per_horizon"] = {
        f"h{h}": {"model_key": key, "cv_rmse": score, "path": path.name}
        for h, (key, score, path) in best_per_horizon.items()
    }
    for h, (key, _, _) in best_per_horizon.items():
        errors = interval_errors[(h, key)]
        metadata["prediction_intervals"][f"h{h}"] = {
            "method": "walk_forward_absolute_error",
            "coverage": 0.90,
            "absolute_error_p90": float(np.quantile(errors, 0.90)),
        }
    promotion_by_horizon = {
        f"h{h}": bool(metadata["models"][f"h{h}:{key}"]["beats_persistence_on_test"])
        for h, (key, _, _) in best_per_horizon.items()
    }
    metadata["promotion"] = {
        "required_to_beat_persistence": bool(
            config.get("deployment", {}).get("require_persistence_improvement", True)
        ),
        "eligible_by_horizon": promotion_by_horizon,
        "all_horizons_eligible": all(promotion_by_horizon.values()),
    }
    metadata_path = output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Best Models ===")
    for h, (key, score, path) in best_per_horizon.items():
        print(f"  h{h}: {key} CV_RMSE={score:.4f} -> {path.name}")
    print(f"\nSaved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
