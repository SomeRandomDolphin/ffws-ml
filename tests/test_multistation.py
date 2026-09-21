"""Tes fondasi baseline multi-stasiun Fase 1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dhompo.data.features import (
    build_multistation_dataset_from_segments,
    build_multistation_features,
    build_multistation_targets,
)
from dhompo.data.loader import ALL_STATIONS, DataSegment
from training.dhompo.train_multistation import (
    evaluate_predictions,
    persistence_predictions,
    segment_split_masks,
)
from training.evaluate import calc_metrics


def _station_frame(start: str, periods: int, offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="30min", name="Datetime")
    time = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            station: offset + station_idx * 100.0 + time
            for station_idx, station in enumerate(ALL_STATIONS)
        },
        index=index,
    )


def test_multistation_features_include_every_station_t0():
    frame = _station_frame("2022-01-01", 48)
    features = build_multistation_features(frame)
    assert len(features) == 25
    for station in ALL_STATIONS:
        assert f"{station}_t0" in features.columns


def test_multistation_targets_shift_each_horizon():
    frame = _station_frame("2022-01-01", 48)
    targets = build_multistation_targets(
        frame,
        horizons=[1, 6],
        horizon_steps={1: 2, 6: 12},
    )
    timestamp = frame.index[10]
    assert targets[1].loc[timestamp, "Dhompo"] == frame["Dhompo"].iloc[12]
    assert targets[6].loc[timestamp, "Dhompo"] == frame["Dhompo"].iloc[22]
    assert list(targets[1].columns) == ALL_STATIONS


def test_segment_dataset_does_not_cross_gap():
    first = _station_frame("2022-01-01", 60, offset=0.0)
    second = _station_frame("2023-01-01", 60, offset=10_000.0)
    segments = [
        DataSegment(first, "first", None),
        DataSegment(second, "second", None),
    ]
    X, targets, sources = build_multistation_dataset_from_segments(
        segments,
        horizons=[1, 6],
        horizon_steps={1: 2, 6: 12},
    )
    assert set(sources.unique()) == {"first", "second"}
    first_mask = sources == "first"
    second_mask = sources == "second"
    assert targets[6].loc[first_mask].to_numpy().max() < 10_000.0
    assert targets[1].loc[second_mask].to_numpy().min() >= 10_000.0
    assert X.index.equals(sources.index)


def test_segment_split_masks_split_each_source_temporally():
    sources = pd.Series(["first"] * 10 + ["second"] * 20)
    train, test = segment_split_masks(sources, 0.8)
    assert train.sum() == 24
    assert test.sum() == 6
    assert train[:8].all() and test[8:10].all()
    assert train[10:26].all() and test[26:].all()
    assert not np.any(train & test)


def test_segment_split_rejects_invalid_ratio():
    with pytest.raises(ValueError, match="train_split"):
        segment_split_masks(pd.Series(["x", "x"]), 1.0)


def test_segment_split_can_purge_rows_before_test():
    sources = pd.Series(["first"] * 20 + ["second"] * 20)
    train, test = segment_split_masks(sources, 0.8, purge_rows=3)
    assert train.sum() == 26
    assert test.sum() == 8
    assert not train[13:16].any() and not test[13:16].any()
    assert not train[33:36].any() and not test[33:36].any()


def test_persistence_uses_each_station_t0():
    frame = _station_frame("2022-01-01", 48)
    features = build_multistation_features(frame)
    predicted = persistence_predictions(features, ALL_STATIONS)
    expected = frame.loc[features.index, ALL_STATIONS].to_numpy()
    assert np.array_equal(predicted, expected)


def test_evaluate_predictions_adds_macro_row():
    frame = _station_frame("2022-01-01", 12)[ALL_STATIONS]
    metrics = evaluate_predictions(frame, frame.to_numpy(), "perfect", 1)
    assert len(metrics) == len(ALL_STATIONS) + 1
    macro = metrics.query("station == '__macro__'").iloc[0]
    assert macro["RMSE"] == pytest.approx(0.0)
    assert macro["NSE"] == pytest.approx(1.0)
    assert macro["KGE"] == pytest.approx(1.0)


def test_calc_metrics_reports_kge():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = calc_metrics(values, values.copy())
    assert metrics["KGE"] == pytest.approx(1.0)
