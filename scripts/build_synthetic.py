"""Kalibrasi simulator dan bangun dataset sintetis 15 stasiun DAS Welang."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import yaml

from dhompo.config import load_yaml_config, resolve_path_from_config
from dhompo.data.loader import load_combined_data
from dhompo.data.network import load_network
from dhompo.data.rainfall_sim import fit_rainfall_parameters
from dhompo.data.routing_sim import fit_routing_parameters
from dhompo.data.scenarios import simulate_scenario

DEFAULT_CONFIG = "configs/dhompo/simulator.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bangun data sintetis DAS Welang")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--years", type=float, default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--calibrate-only", action="store_true")
    return parser.parse_args()


def calibrate(config_path: str | Path) -> tuple[dict, object, object, object]:
    """Muat data acuan dan fit parameter hujan serta routing."""
    config = load_yaml_config(config_path)
    source_cfg = config.get("data_sources", {})
    clean_path = resolve_path_from_config(config_path, source_cfg.get("clean"))
    generated_path = resolve_path_from_config(config_path, source_cfg.get("generated"))
    if clean_path is None or generated_path is None:
        raise ValueError("data_sources.clean dan generated wajib diisi.")

    segments = load_combined_data(clean_path, generated_path)
    generated_segments = [segment for segment in segments if segment.rainfall is not None]
    if not generated_segments:
        raise ValueError("Tidak ada segmen dengan curah hujan untuk kalibrasi.")
    rainfall = pd.concat([segment.rainfall for segment in generated_segments], axis=0)

    rainfall_cfg = config.get("rainfall", {})
    rainfall_parameters = fit_rainfall_parameters(
        rainfall,
        monthly_multipliers=rainfall_cfg.get("monthly_multipliers", [1.0] * 12),
        wet_threshold_mm=float(rainfall_cfg.get("wet_threshold_mm", 0.0)),
        laplace_smoothing=float(rainfall_cfg.get("laplace_smoothing", 1.0)),
        min_gamma_shape=float(rainfall_cfg.get("min_gamma_shape", 0.2)),
        max_rain_mm=float(rainfall_cfg.get("max_rain_mm", 50.0)),
    )
    network = load_network()
    routing_parameters = fit_routing_parameters(segments, network, config)
    return config, network, rainfall_parameters, routing_parameters


def save_parameters(
    path: Path,
    rainfall_parameters,
    routing_parameters,
) -> None:
    """Simpan seluruh parameter kalibrasi sebagai YAML yang dapat dimuat ulang."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "provenance": {
            "rainfall": "Data generated 2023.xlsx",
            "routing": "data-clean.csv + Data generated 2023.xlsx",
            "ground_truth": False,
        },
        "rainfall": rainfall_parameters.to_dict(),
        "routing": routing_parameters.to_dict(),
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def summarize_scenario(scenario: pd.DataFrame, stations: list[str]) -> pd.DataFrame:
    """Ringkas hujan dan level untuk pemeriksaan plausibilitas hasil."""
    rows = []
    for variable in ["rain_mm", *stations]:
        values = scenario[variable].astype(float)
        rows.append({
            "variable": variable,
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "q95": float(values.quantile(0.95)),
            "q99": float(values.quantile(0.99)),
            "max": float(values.max()),
            "zero_fraction": float((values == 0).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config, network, rainfall_parameters, routing_parameters = calibrate(args.config)
    generation_cfg = config.get("generation", {})
    parameter_path = resolve_path_from_config(
        args.config, generation_cfg.get("parameter_path", "simulator_calibrated.yaml"),
    )
    if parameter_path is None:
        raise ValueError("generation.parameter_path wajib diisi.")
    save_parameters(parameter_path, rainfall_parameters, routing_parameters)

    print("Kalibrasi selesai")
    print(
        "Hujan: "
        f"P(wet|dry)={rainfall_parameters.p_wet_after_dry:.4f}, "
        f"P(wet|wet)={rainfall_parameters.p_wet_after_wet:.4f}, "
        f"Gamma(k={rainfall_parameters.gamma_shape:.4f}, "
        f"theta={rainfall_parameters.gamma_scale:.4f})"
    )
    print(f"Parameter: {parameter_path}")
    if args.calibrate_only:
        return

    years = float(args.years if args.years is not None else generation_cfg.get("years", 10))
    periods = args.periods if args.periods is not None else int(round(years * 365.25 * 48))
    if periods <= 0:
        raise ValueError("Jumlah periods hasil konfigurasi harus lebih besar dari nol.")
    seed = int(args.seed if args.seed is not None else generation_cfg.get("seed", 42))
    start = args.start or generation_cfg.get("start", "2024-01-01 00:00:00")
    output_path = (
        Path(args.output)
        if args.output
        else resolve_path_from_config(args.config, generation_cfg.get("output_path"))
    )
    summary_path = resolve_path_from_config(
        args.config, generation_cfg.get("summary_path"),
    )
    if output_path is None or summary_path is None:
        raise ValueError("generation.output_path dan summary_path wajib diisi.")

    scenario = simulate_scenario(
        start,
        periods,
        rainfall_parameters,
        routing_parameters,
        network,
        seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario.to_csv(output_path, index_label="timestamp")
    summary = summarize_scenario(scenario, network.station_names)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print(f"Dataset: {output_path} ({len(scenario)} baris)")
    print(f"Ringkasan: {summary_path}")


if __name__ == "__main__":
    main()
