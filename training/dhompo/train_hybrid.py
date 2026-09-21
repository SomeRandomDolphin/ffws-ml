"""Latih korektor residual di atas backbone simulator jaringan Fase 2."""

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
from sklearn.multioutput import MultiOutputRegressor

from dhompo.config import load_yaml_config, resolve_path_from_config
from dhompo.data.loader import DataSegment, load_combined_data
from dhompo.data.network import load_network
from dhompo.data.routing_sim import fit_routing_parameters, forecast_levels
from training.dhompo.train_multistation import (
    evaluate_predictions,
    make_model,
    persistence_predictions,
    prepare_dataset,
)

DEFAULT_CONFIG = "configs/dhompo/hybrid_training.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual hybrid Dhompo")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_reference_segments(baseline_config_path: Path) -> list[DataSegment]:
    """Muat dua segmen acuan mengikuti konfigurasi baseline."""
    config = load_yaml_config(baseline_config_path)
    sources = {item["label"]: item for item in config.get("data_sources", [])}
    clean_path = resolve_path_from_config(
        baseline_config_path, sources["2022_clean"]["path"],
    )
    generated_path = resolve_path_from_config(
        baseline_config_path, sources["2023_generated"]["path"],
    )
    return load_combined_data(clean_path, generated_path)


def calibration_segments(
    segments: list[DataSegment],
    sources: pd.Series,
    train_mask: np.ndarray,
) -> list[DataSegment]:
    """Potong data kalibrasi pada timestamp terakhir train tiap segmen."""
    output = []
    for segment in segments:
        timestamps = sources.index[(sources == segment.label) & train_mask]
        if len(timestamps) == 0:
            raise ValueError(f"Tidak ada sampel train untuk segmen {segment.label!r}.")
        cutoff = timestamps.max()
        frame = segment.df.loc[:cutoff]
        rainfall = segment.rainfall.loc[:cutoff] if segment.rainfall is not None else None
        output.append(DataSegment(frame, segment.label, rainfall))
    return output


def build_backbone_forecasts(
    X: pd.DataFrame,
    sources: pd.Series,
    segments: list[DataSegment],
    horizons: list[int],
    stations: list[str],
    routing_parameters,
    history_rows: int,
) -> dict[int, pd.DataFrame]:
    """Forecast simulator zero-rain untuk setiap waktu origin pada dataset."""
    if history_rows < 2:
        raise ValueError("history_rows minimal 2.")
    network = load_network()
    maximum_step = max(horizons) * 2
    outputs = {
        horizon: pd.DataFrame(index=X.index, columns=stations, dtype=float)
        for horizon in horizons
    }
    segment_by_label = {segment.label: segment for segment in segments}

    for label in sources.drop_duplicates():
        segment = segment_by_label[label]
        label_index = X.index[sources == label]
        for timestamp in label_index:
            history = segment.df.loc[:timestamp, stations].tail(history_rows)
            future_index = pd.date_range(
                timestamp + pd.Timedelta("30min"),
                periods=maximum_step,
                freq="30min",
            )
            future_rainfall = pd.Series(0.0, index=future_index, name="rain_mm")
            historical_rainfall = segment.rainfall if segment.rainfall is not None else None
            forecast = forecast_levels(
                history,
                future_rainfall,
                routing_parameters,
                network,
                historical_rainfall=historical_rainfall,
            )
            for horizon in horizons:
                outputs[horizon].loc[timestamp] = forecast.iloc[horizon * 2 - 1].to_numpy()
    return outputs


def augment_with_backbone(
    X: pd.DataFrame,
    backbone: pd.DataFrame,
) -> pd.DataFrame:
    """Tambahkan prediksi simulator sebagai fitur horizon-spesifik."""
    simulator_features = backbone.rename(
        columns={station: f"simulator_{station}" for station in backbone.columns}
    )
    return pd.concat([X, simulator_features], axis=1)


def make_residual_model(config: dict):
    """Bangun korektor residual multi-output HistGradientBoosting."""
    model_cfg = config.get("residual_model", {})
    base = HistGradientBoostingRegressor(
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        max_iter=int(model_cfg.get("max_iter", 150)),
        max_depth=int(model_cfg.get("max_depth", 6)),
        l2_regularization=float(model_cfg.get("l2_regularization", 1.0)),
        random_state=int(model_cfg.get("random_state", 42)),
    )
    return MultiOutputRegressor(base, n_jobs=-1)


def main() -> None:
    args = parse_args()
    hybrid_config = load_yaml_config(args.config)
    baseline_config_path = resolve_path_from_config(
        args.config, hybrid_config.get("baseline_config"),
    )
    simulator_config_path = resolve_path_from_config(
        args.config, hybrid_config.get("simulator_config"),
    )
    if baseline_config_path is None or simulator_config_path is None:
        raise ValueError("baseline_config dan simulator_config wajib diisi.")

    (
        baseline_config,
        stations,
        horizons,
        X,
        targets,
        sources,
        train_mask,
        test_mask,
    ) = prepare_dataset(baseline_config_path)
    segments = load_reference_segments(baseline_config_path)
    network = load_network()
    simulator_config = load_yaml_config(simulator_config_path)
    routing_parameters = fit_routing_parameters(
        calibration_segments(segments, sources, train_mask),
        network,
        simulator_config,
    )
    history_rows = int(hybrid_config.get("history_rows", 48))
    print(
        f"Dataset: {len(X)} baris, train={int(train_mask.sum())}, "
        f"test={int(test_mask.sum())}, fitur={X.shape[1]}"
    )
    print("Membangun backbone simulator zero-rain...")
    backbone = build_backbone_forecasts(
        X,
        sources,
        segments,
        horizons,
        stations,
        routing_parameters,
        history_rows,
    )
    if args.dry_run:
        print(f"Backbone h{max(horizons)}: {backbone[max(horizons)].shape}")
        return

    model_dir = resolve_path_from_config(
        args.config, hybrid_config.get("output", {}).get("model_dir"),
    )
    report_dir = resolve_path_from_config(
        args.config, hybrid_config.get("output", {}).get("report_dir"),
    )
    if model_dir is None or report_dir is None:
        raise ValueError("output.model_dir dan report_dir wajib diisi.")
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(routing_parameters, model_dir / "routing_parameters.joblib")

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    metric_parts = []
    for horizon in horizons:
        y_train = targets[horizon].loc[train_mask]
        y_test = targets[horizon].loc[test_mask]
        simulator_train = backbone[horizon].loc[train_mask].to_numpy(dtype=float)
        simulator_test = backbone[horizon].loc[test_mask].to_numpy(dtype=float)
        persistence_train = persistence_predictions(X_train, stations)
        persistence_test = persistence_predictions(X_test, stations)

        direct_model = make_model(
            "hist_gradient_boosting",
            baseline_config,
            random_state=int(hybrid_config.get("residual_model", {}).get("random_state", 42)),
        )
        direct_model.fit(X_train, y_train)
        direct_train = direct_model.predict(X_train)
        direct_test = direct_model.predict(X_test)
        joblib.dump(direct_model, model_dir / f"direct_h{horizon}.joblib")

        X_train_hybrid = augment_with_backbone(
            X_train, backbone[horizon].loc[train_mask],
        )
        X_test_hybrid = augment_with_backbone(
            X_test, backbone[horizon].loc[test_mask],
        )
        residual_model = make_residual_model(hybrid_config)
        residual_model.fit(
            X_train_hybrid,
            y_train.to_numpy(dtype=float) - simulator_train,
        )
        hybrid_train = simulator_train + residual_model.predict(X_train_hybrid)
        hybrid_test = simulator_test + residual_model.predict(X_test_hybrid)
        joblib.dump(residual_model, model_dir / f"residual_h{horizon}.joblib")

        predictions = {
            "persistence": (persistence_train, persistence_test),
            "simulator": (simulator_train, simulator_test),
            "direct_hist_gradient_boosting": (direct_train, direct_test),
            "hybrid_residual_hist_gradient_boosting": (hybrid_train, hybrid_test),
        }
        for name, (train_prediction, test_prediction) in predictions.items():
            metric_parts.append(
                evaluate_predictions(y_train, train_prediction, name, horizon, "train")
            )
            metric_parts.append(
                evaluate_predictions(y_test, test_prediction, name, horizon, "test")
            )
            macro = metric_parts[-1].query("station == '__macro__'").iloc[0]
            print(
                f"h{horizon} {name:40s} "
                f"NSE={macro['NSE']:.4f} RMSE={macro['RMSE']:.4f}"
            )

    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics_path = report_dir / "fase3_hybrid_metrics.csv"
    summary_path = report_dir / "fase3_hybrid_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    metrics.query("split == 'test' and station == '__macro__'").to_csv(
        summary_path, index=False,
    )
    metadata = {
        "stations": stations,
        "horizons": horizons,
        "history_rows": history_rows,
        "future_rainfall": hybrid_config.get("future_rainfall", "zero"),
        "base_features": list(X.columns),
        "hybrid_feature_columns": list(
            augment_with_backbone(X.iloc[:1], backbone[horizons[0]].iloc[:1]).columns
        ),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "calibration_scope": "train_only_per_segment",
        "metrics": str(metrics_path),
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    print(f"Metrik: {metrics_path}")
    print(f"Model: {model_dir}")


if __name__ == "__main__":
    main()
