"""Tes kalibrasi dan simulasi routing jaringan 15 stasiun."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from dhompo.data.loader import ALL_STATIONS, DataSegment
from dhompo.data.network import load_network
from dhompo.data.rainfall_sim import RainfallParameters
from dhompo.data.routing_sim import (
    ReachRoutingParameters,
    RoutingParameters,
    StationRoutingParameters,
    fit_routing_parameters,
    forecast_levels,
    simulate_levels,
)
from dhompo.data.scenarios import (
    load_simulator_parameters,
    simulate_ensemble,
    simulate_scenario,
)


def _segments(periods: int = 96) -> list[DataSegment]:
    index = pd.date_range("2022-01-01", periods=periods, freq="30min", name="Datetime")
    time = np.arange(periods, dtype=float)
    frame = pd.DataFrame(
        {
            station: station_idx * 10.0 + 2.0 + np.sin(time / (8.0 + station_idx))
            for station_idx, station in enumerate(ALL_STATIONS)
        },
        index=index,
    )
    rain = pd.Series(np.where(time % 20 < 3, 2.0, 0.0), index=index, name="Curah hujan")
    return [DataSegment(frame, "reference", rain)]


def test_fit_routing_parameters_covers_network():
    network = load_network()
    parameters = fit_routing_parameters(_segments(), network, {"routing": {}})
    assert set(parameters.stations) == set(network.station_names)
    assert len(parameters.reaches) == len(network.reaches)
    assert all(0.0 < reach.attenuation <= 0.95 for reach in parameters.reaches)
    assert all(reach.lag_steps >= 1 for reach in parameters.reaches)


def test_simulated_levels_are_reproducible_finite_and_nonnegative():
    network = load_network()
    parameters = fit_routing_parameters(_segments(), network, {"routing": {}})
    rainfall = pd.Series(
        np.where(np.arange(240) % 40 < 4, 5.0, 0.0),
        index=pd.date_range("2024-01-01", periods=240, freq="30min"),
    )
    first = simulate_levels(rainfall, parameters, network, seed=11)
    second = simulate_levels(rainfall, parameters, network, seed=11)
    assert first.equals(second)
    assert list(first.columns) == network.station_names
    assert np.isfinite(first.to_numpy()).all()
    assert first.to_numpy().min() >= 0.0
    assert first.std().min() > 0.0


def test_upstream_pulse_reaches_dhompo_after_configured_lag():
    network = load_network()
    station_parameters = {
        station: StationRoutingParameters(10.0, 0.0, 0.0, 0.0, 1.0)
        for station in network.station_names
    }
    station_parameters["Klosod"] = StationRoutingParameters(10.0, 0.0, 1.0, 0.0, 1.0)
    reaches = tuple(
        ReachRoutingParameters(
            reach.upstream,
            reach.downstream,
            max(1, int(round(reach.travel_hours / 0.5))),
            0.5,
        )
        for reach in network.reaches
    )
    parameters = RoutingParameters(0.5, 0.0, 0.0, 0.0, station_parameters, reaches)
    rainfall = pd.Series(
        [0.0, 4.0, 0.0, 0.0, 0.0],
        index=pd.date_range("2024-01-01", periods=5, freq="30min"),
    )
    levels = simulate_levels(rainfall, parameters, network, seed=1)
    assert levels["Dhompo"].iloc[2] == pytest.approx(10.0)
    assert levels["Dhompo"].iloc[3] > 10.0


def test_routing_parameter_round_trip():
    network = load_network()
    original = fit_routing_parameters(_segments(), network, {"routing": {}})
    restored = RoutingParameters.from_dict(original.to_dict())
    assert restored == original


def test_forecast_levels_uses_history_and_future_index():
    network = load_network()
    segments = _segments()
    parameters = fit_routing_parameters(segments, network, {"routing": {}})
    history = segments[0].df.iloc[:48]
    future_index = pd.date_range(
        history.index[-1] + pd.Timedelta("30min"), periods=12, freq="30min",
    )
    future_rainfall = pd.Series(0.0, index=future_index)
    first = forecast_levels(history, future_rainfall, parameters, network, seed=3)
    second = forecast_levels(history, future_rainfall, parameters, network, seed=99)
    assert first.equals(second)
    assert first.index.equals(future_index)
    assert list(first.columns) == network.station_names
    assert np.isfinite(first.to_numpy()).all()


def test_scenario_and_ensemble_schema():
    network = load_network()
    routing = fit_routing_parameters(_segments(), network, {"routing": {}})
    rainfall = RainfallParameters(0.1, 0.6, 1.5, 2.0, (1.0,) * 12)
    scenario = simulate_scenario("2024-01-01", 48, rainfall, routing, network, seed=4)
    assert list(scenario.columns) == ["scenario_id", "rain_mm", *network.station_names]
    assert scenario["scenario_id"].nunique() == 1
    ensemble = simulate_ensemble(
        "2024-01-01", 48, rainfall, routing, network, seeds=[4, 5],
    )
    assert len(ensemble) == 96
    assert ensemble["scenario_id"].nunique() == 2


def test_empty_ensemble_is_rejected():
    network = load_network()
    routing = fit_routing_parameters(_segments(), network, {"routing": {}})
    rainfall = RainfallParameters(0.1, 0.6, 1.5, 2.0, (1.0,) * 12)
    with pytest.raises(ValueError, match="Minimal satu seed"):
        simulate_ensemble("2024-01-01", 48, rainfall, routing, network, seeds=[])


def test_load_simulator_parameters(tmp_path):
    network = load_network()
    routing = fit_routing_parameters(_segments(), network, {"routing": {}})
    rainfall = RainfallParameters(0.1, 0.6, 1.5, 2.0, (1.0,) * 12)
    path = tmp_path / "calibrated.yaml"
    path.write_text(
        yaml.safe_dump({
            "rainfall": rainfall.to_dict(),
            "routing": routing.to_dict(),
        }),
        encoding="utf-8",
    )
    restored_rainfall, restored_routing = load_simulator_parameters(path)
    assert restored_rainfall == rainfall
    assert restored_routing == routing
