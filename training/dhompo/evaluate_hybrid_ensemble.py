"""Evaluasi probabilistik ensemble hujan pada test split Fase 3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dhompo.config import load_yaml_config, resolve_path_from_config
from dhompo.serving.hybrid_predictor import HybridMultiStationPredictor
from training.dhompo.train_hybrid import load_reference_segments
from training.dhompo.train_multistation import prepare_dataset
from training.evaluate import calc_probabilistic_metrics

DEFAULT_CONFIG = "configs/dhompo/hybrid_training.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluasi ensemble hybrid Dhompo")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--stride-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    evaluation_cfg = config.get("probabilistic_evaluation", {})
    scenario_count = int(
        args.scenario_count
        if args.scenario_count is not None
        else evaluation_cfg.get("scenario_count", 30)
    )
    stride_rows = int(
        args.stride_rows
        if args.stride_rows is not None
        else evaluation_cfg.get("stride_rows", 12)
    )
    seed = int(args.seed if args.seed is not None else evaluation_cfg.get("seed", 2026))
    if stride_rows < 1:
        raise ValueError("stride_rows harus minimal 1.")

    baseline_config_path = resolve_path_from_config(
        args.config, config.get("baseline_config"),
    )
    model_dir = resolve_path_from_config(
        args.config, config.get("output", {}).get("model_dir"),
    )
    report_dir = resolve_path_from_config(
        args.config, config.get("output", {}).get("report_dir"),
    )
    figure_dir = resolve_path_from_config(
        args.config, config.get("output", {}).get("figure_dir"),
    )
    if None in (baseline_config_path, model_dir, report_dir, figure_dir):
        raise ValueError("Path baseline/model/report/figure wajib diisi.")

    _, stations, horizons, _, targets, sources, _, test_mask = prepare_dataset(
        baseline_config_path,
    )
    segments = load_reference_segments(baseline_config_path)
    segment_by_label = {segment.label: segment for segment in segments}
    predictor = HybridMultiStationPredictor(model_dir)

    test_sources = sources.loc[test_mask]
    selected = []
    for label in test_sources.drop_duplicates():
        selected.extend(test_sources[test_sources == label].index[::stride_rows])
    selected_index = pd.DatetimeIndex(selected)
    observations = {
        horizon: targets[horizon].loc[selected_index]
        for horizon in horizons
    }
    ensembles = {
        horizon: np.zeros((len(selected_index), scenario_count, len(stations)))
        for horizon in horizons
    }

    for origin_idx, timestamp in enumerate(selected_index):
        label = sources.loc[timestamp]
        segment = segment_by_label[label]
        history = segment.df.loc[:timestamp, stations].tail(48)
        historical_rainfall = segment.rainfall if segment.rainfall is not None else None
        members, _ = predictor.ensemble_members_from_history(
            history,
            scenario_count=scenario_count,
            seed=seed + origin_idx * scenario_count,
            historical_rainfall=historical_rainfall,
        )
        for horizon in horizons:
            ensembles[horizon][origin_idx] = members[horizon]

    rows = []
    for horizon in horizons:
        for station_idx, station in enumerate(stations):
            metrics = calc_probabilistic_metrics(
                observations[horizon][station].to_numpy(),
                ensembles[horizon][:, :, station_idx],
            )
            rows.append({
                "horizon": horizon,
                "station": station,
                "n_origins": len(selected_index),
                "scenario_count": scenario_count,
                **metrics,
            })
    metrics = pd.DataFrame(rows)
    macro = metrics.groupby("horizon", as_index=False)[
        ["COVERAGE", "MEAN_WIDTH", "INTERVAL_SCORE", "CRPS", "MEDIAN_MAE"]
    ].mean()
    macro.insert(1, "station", "__macro__")
    macro["n_origins"] = len(selected_index)
    macro["scenario_count"] = scenario_count
    metrics = pd.concat([metrics, macro], ignore_index=True)

    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "fase4_ensemble_metrics.csv"
    metrics.to_csv(output_path, index=False)

    macro = metrics[metrics["station"] == "__macro__"].sort_values("horizon")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(macro["horizon"], macro["COVERAGE"], marker="o", label="Coverage P10–P90")
    axis.axhline(0.8, color="black", linestyle="--", linewidth=1, label="Nominal 80%")
    axis.set(xlabel="Horizon (jam)", ylabel="Coverage", ylim=(0, 1), title="Coverage Ensemble Hybrid")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "fase4_ensemble_coverage.png", dpi=180)
    plt.close(figure)

    dhompo_idx = stations.index("Dhompo")
    example = len(selected_index) - 1
    truth = [observations[h].iloc[example, dhompo_idx] for h in horizons]
    p10 = [np.quantile(ensembles[h][example, :, dhompo_idx], 0.1) for h in horizons]
    p50 = [np.quantile(ensembles[h][example, :, dhompo_idx], 0.5) for h in horizons]
    p90 = [np.quantile(ensembles[h][example, :, dhompo_idx], 0.9) for h in horizons]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.fill_between(horizons, p10, p90, alpha=0.25, label="P10–P90")
    axis.plot(horizons, p50, marker="o", label="Median ensemble")
    axis.plot(horizons, truth, marker="s", linestyle="--", label="Data acuan")
    axis.set(xlabel="Horizon (jam)", ylabel="Level Dhompo (m)", title="Contoh Interval Prediksi Dhompo")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "fase4_dhompo_interval_example.png", dpi=180)
    plt.close(figure)

    print(f"Origin evaluasi: {len(selected_index)} | Ensemble: {scenario_count}")
    print(macro[["horizon", "COVERAGE", "MEAN_WIDTH", "CRPS"]].to_string(index=False))
    print(f"Metrik: {output_path}")
    print(f"Figur: {figure_dir}")


if __name__ == "__main__":
    main()
