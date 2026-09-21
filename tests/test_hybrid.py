"""Tes pipeline simulator-backbone dan korektor residual Fase 3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dhompo.data.loader import ALL_STATIONS, DataSegment
from dhompo.data.network import load_network
from dhompo.data.routing_sim import fit_routing_parameters
from training.dhompo.train_hybrid import (
    augment_with_backbone,
    build_backbone_forecasts,
    calibration_segments,
    make_residual_model,
)


def _segment(periods: int = 72) -> DataSegment:
    index = pd.date_range("2022-01-01", periods=periods, freq="30min", name="Datetime")
    time = np.arange(periods, dtype=float)
    frame = pd.DataFrame(
        {
            station: 5.0 + station_idx * 10.0 + np.sin(time / 8.0)
            for station_idx, station in enumerate(ALL_STATIONS)
        },
        index=index,
    )
    rain = pd.Series(
        np.where(time % 20 < 3, 2.0, 0.0), index=index, name="Curah hujan",
    )
    return DataSegment(frame, "segment", rain)


def test_calibration_segments_stop_at_last_training_origin():
    segment = _segment()
    origins = segment.df.index[24:60]
    sources = pd.Series("segment", index=origins)
    train_mask = np.zeros(len(origins), dtype=bool)
    train_mask[:20] = True
    calibrated = calibration_segments([segment], sources, train_mask)
    assert calibrated[0].df.index.max() == origins[19]
    assert calibrated[0].rainfall.index.max() == origins[19]


def test_backbone_forecasts_have_aligned_shapes():
    segment = _segment()
    origins = segment.df.index[24:48]
    X = pd.DataFrame({"feature": np.arange(len(origins))}, index=origins)
    sources = pd.Series("segment", index=origins)
    network = load_network()
    routing = fit_routing_parameters([segment], network, {"routing": {}})
    forecasts = build_backbone_forecasts(
        X,
        sources,
        [segment],
        horizons=[1, 2],
        stations=network.station_names,
        routing_parameters=routing,
        history_rows=24,
    )
    assert set(forecasts) == {1, 2}
    assert forecasts[1].shape == (len(origins), len(ALL_STATIONS))
    assert forecasts[2].index.equals(origins)
    assert np.isfinite(forecasts[2].to_numpy()).all()


def test_backbone_features_and_residual_model_shape():
    index = pd.date_range("2022-01-01", periods=40, freq="30min")
    X = pd.DataFrame({"feature": np.linspace(0.0, 1.0, len(index))}, index=index)
    backbone = pd.DataFrame(
        np.tile(np.arange(len(ALL_STATIONS)), (len(index), 1)),
        columns=ALL_STATIONS,
        index=index,
    )
    augmented = augment_with_backbone(X, backbone)
    assert augmented.shape[1] == len(ALL_STATIONS) + 1
    model = make_residual_model({"residual_model": {"max_iter": 5}})
    target = np.zeros((len(index), len(ALL_STATIONS)))
    model.fit(augmented, target)
    assert model.predict(augmented.iloc[:3]).shape == (3, len(ALL_STATIONS))
