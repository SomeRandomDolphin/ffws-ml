"""Train baseline sklearn models for the urban water-level dataset.

Usage
-----
    python training/surabaya/train_urban_sklearn.py --models ridge
    python training/surabaya/train_urban_sklearn.py --models gradient_boosting,xgboost

The pipeline is intentionally independent from ``training/dhompo/train_sklearn.py``
because the existing script targets the Dhompo station schema.
"""

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
import pandas as pd
from sklearn.base import clone
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
    }

    best_per_horizon: dict[int, tuple[str, float, Path]] = {}
    for h in horizons:
        y = y_horizons[h]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        level_train = future_targets[h].iloc[:split_idx]
        level_test = future_targets[h].iloc[split_idx:]
        current_train = current_target.iloc[:split_idx]
        current_test = current_target.iloc[split_idx:]
        print(f"\n=== Horizon +{h}h ===")

        for model_name, (template, use_scaled) in model_defs.items():
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
            }

            nse = float(test_metrics["NSE"])
            rmse = float(test_metrics["RMSE"])
            print(f"  {model_name:24s} NSE={nse:.4f} RMSE={rmse:.4f}")
            if h not in best_per_horizon or nse > best_per_horizon[h][1]:
                best_per_horizon[h] = (model_key, nse, model_path)

    metadata["best_per_horizon"] = {
        f"h{h}": {"model_key": key, "nse": nse, "path": path.name}
        for h, (key, nse, path) in best_per_horizon.items()
    }
    metadata_path = output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Best Models ===")
    for h, (key, nse, path) in best_per_horizon.items():
        print(f"  h{h}: {key} NSE={nse:.4f} -> {path.name}")
    print(f"\nSaved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
