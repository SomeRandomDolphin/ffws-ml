"""Latih baseline multi-output untuk 15 stasiun dan horizon +1..+6 jam.

Data dibangun per segmen agar lag dan target tidak menyeberangi gap. Setiap
segmen dibagi temporal 80/20; persistence, Ridge, dan HistGradientBoosting
dibandingkan dengan metrik per stasiun dan agregat makro.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dhompo.config import load_yaml_config, resolve_path_from_config
from dhompo.data.features import build_multistation_dataset_from_segments
from dhompo.data.loader import RAINFALL_COLUMN, load_combined_data
from dhompo.data.network import load_network
from training.evaluate import calc_metrics

DEFAULT_CONFIG = "configs/dhompo/multistation_training.yaml"
METRIC_COLUMNS = ("RMSE", "MAE", "R2", "NSE", "KGE", "PBIAS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline multi-stasiun Dhompo")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--models",
        default=None,
        help="Daftar model dipisah koma; persistence,ridge,hist_gradient_boosting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validasi dan tampilkan bentuk dataset tanpa melatih model.",
    )
    return parser.parse_args()


def segment_split_masks(
    sources: pd.Series,
    train_split: float,
    purge_rows: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bagi tiap segmen temporal dengan purge gap sebelum test."""
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split harus berada di antara 0 dan 1.")
    if purge_rows < 0:
        raise ValueError("purge_rows tidak boleh negatif.")

    train = np.zeros(len(sources), dtype=bool)
    test = np.zeros(len(sources), dtype=bool)
    values = sources.to_numpy()
    for label in sources.drop_duplicates().tolist():
        positions = np.flatnonzero(values == label)
        split = int(len(positions) * train_split)
        train_end = split - purge_rows
        if train_end <= 0 or split == len(positions):
            raise ValueError(f"Segmen {label!r} terlalu pendek untuk dibagi.")
        train[positions[:train_end]] = True
        test[positions[split:]] = True
    return train, test


def make_model(name: str, config: dict, random_state: int = 42):
    """Bangun estimator multi-output baseline berdasarkan nama."""
    model_cfg = config.get("models", {}).get(name, {})
    if name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=float(model_cfg.get("alpha", 1.0)))),
        ])
    if name == "hist_gradient_boosting":
        base = HistGradientBoostingRegressor(
            learning_rate=float(model_cfg.get("learning_rate", 0.05)),
            max_iter=int(model_cfg.get("max_iter", 150)),
            max_depth=int(model_cfg.get("max_depth", 6)),
            l2_regularization=float(model_cfg.get("l2_regularization", 1.0)),
            random_state=random_state,
        )
        return MultiOutputRegressor(base, n_jobs=-1)
    raise ValueError(f"Model tidak dikenal: {name!r}.")


def persistence_predictions(
    X: pd.DataFrame,
    stations: list[str],
) -> np.ndarray:
    """Prediksi persistence: nilai setiap stasiun pada t dipakai untuk semua horizon."""
    columns = [f"{station}_t0" for station in stations]
    missing = [column for column in columns if column not in X.columns]
    if missing:
        raise ValueError(f"Fitur persistence tidak tersedia: {missing}.")
    return X[columns].to_numpy(dtype=float)


def evaluate_predictions(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str,
    horizon: int,
    split: str = "test",
) -> pd.DataFrame:
    """Hitung metrik per stasiun dan satu baris agregat makro."""
    predicted = np.asarray(y_pred, dtype=float)
    if predicted.shape != y_true.shape:
        raise ValueError(
            f"Shape prediksi {predicted.shape} tidak sama dengan target {y_true.shape}."
        )

    rows: list[dict[str, float | int | str]] = []
    for idx, station in enumerate(y_true.columns):
        metrics = calc_metrics(y_true.iloc[:, idx].to_numpy(), predicted[:, idx])
        rows.append({
            "model": model_name,
            "horizon": horizon,
            "split": split,
            "station": station,
            **metrics,
        })

    metric_frame = pd.DataFrame(rows)
    macro = {
        "model": model_name,
        "horizon": horizon,
        "split": split,
        "station": "__macro__",
    }
    macro.update({
        metric: float(metric_frame[metric].replace([np.inf, -np.inf], np.nan).mean())
        for metric in METRIC_COLUMNS
    })
    return pd.concat([metric_frame, pd.DataFrame([macro])], ignore_index=True)


def prepare_dataset(config_path: str | Path):
    """Muat konfigurasi dan bangun dataset segment-aware Fase 1."""
    config = load_yaml_config(config_path)
    data_sources = {item["label"]: item for item in config.get("data_sources", [])}
    required = {"2022_clean", "2023_generated"}
    if not required.issubset(data_sources):
        raise ValueError(f"data_sources harus memuat {sorted(required)}.")

    clean_path = resolve_path_from_config(
        config_path, data_sources["2022_clean"]["path"],
    )
    generated_path = resolve_path_from_config(
        config_path, data_sources["2023_generated"]["path"],
    )
    segments = load_combined_data(clean_path, generated_path)
    stations = load_network().station_names
    horizons = [int(h) for h in config.get("horizons", range(1, 7))]
    horizon_steps = {h: h * 2 for h in horizons}
    extra_columns = [RAINFALL_COLUMN] if config.get("include_rainfall", False) else None

    X, targets, sources = build_multistation_dataset_from_segments(
        segments,
        horizons=horizons,
        horizon_steps=horizon_steps,
        stations=stations,
        extra_columns=extra_columns,
    )
    train_mask, test_mask = segment_split_masks(
        sources,
        float(config.get("train_split", 0.8)),
        purge_rows=max(horizon_steps.values()),
    )
    return config, stations, horizons, X, targets, sources, train_mask, test_mask


def main() -> None:
    args = parse_args()
    (
        config,
        stations,
        horizons,
        X,
        targets,
        sources,
        train_mask,
        test_mask,
    ) = prepare_dataset(args.config)

    print(f"Data: {X.shape[0]} baris x {X.shape[1]} fitur")
    print(f"Stasiun: {len(stations)} | Horizon: {horizons}")
    for label in sources.drop_duplicates():
        label_mask = sources.to_numpy() == label
        print(
            f"  {label}: train={int((train_mask & label_mask).sum())}, "
            f"test={int((test_mask & label_mask).sum())}"
        )
    if args.dry_run:
        return

    configured_models = ["persistence", *config.get("models", {}).keys()]
    model_names = (
        [item.strip() for item in args.models.split(",") if item.strip()]
        if args.models
        else configured_models
    )
    unknown = set(model_names) - {"persistence", "ridge", "hist_gradient_boosting"}
    if unknown:
        raise ValueError(f"Model tidak dikenal: {sorted(unknown)}.")

    report_dir = resolve_path_from_config(
        args.config, config.get("output", {}).get("report_dir"),
    )
    model_dir = resolve_path_from_config(
        args.config, config.get("output", {}).get("model_dir"),
    )
    if report_dir is None or model_dir is None:
        raise ValueError("output.report_dir dan output.model_dir wajib diisi.")
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    metric_parts: list[pd.DataFrame] = []
    random_state = int(config.get("random_state", 42))

    for horizon in horizons:
        y_train = targets[horizon].loc[train_mask]
        y_test = targets[horizon].loc[test_mask]
        for model_name in model_names:
            if model_name == "persistence":
                train_pred = persistence_predictions(X_train, stations)
                test_pred = persistence_predictions(X_test, stations)
            else:
                model = make_model(model_name, config, random_state=random_state)
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                joblib.dump(model, model_dir / f"{model_name}_h{horizon}.joblib")

            metric_parts.append(
                evaluate_predictions(y_train, train_pred, model_name, horizon, "train")
            )
            metric_parts.append(
                evaluate_predictions(y_test, test_pred, model_name, horizon, "test")
            )
            macro = metric_parts[-1].query("station == '__macro__'").iloc[0]
            print(
                f"h{horizon} {model_name:24s} "
                f"NSE={macro['NSE']:.4f} RMSE={macro['RMSE']:.4f}"
            )

    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics_path = report_dir / "fase1_multistation_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    summary = metrics.query("split == 'test' and station == '__macro__'")
    summary_path = report_dir / "fase1_multistation_summary.csv"
    summary.to_csv(summary_path, index=False)

    metadata = {
        "config": str(Path(args.config)),
        "stations": stations,
        "horizons": horizons,
        "features": list(X.columns),
        "models": model_names,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "source_counts": sources.value_counts().to_dict(),
        "metrics": str(metrics_path),
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    print(f"Metrik: {metrics_path}")
    print(f"Model: {model_dir}")


if __name__ == "__main__":
    main()
