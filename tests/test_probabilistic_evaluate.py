"""Tes metrik evaluasi probabilistik ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from training.evaluate import calc_probabilistic_metrics


def test_probabilistic_metrics_for_centered_ensemble():
    truth = np.array([1.0, 2.0, 3.0])
    ensemble = np.array([
        [0.8, 1.0, 1.2],
        [1.8, 2.0, 2.2],
        [2.8, 3.0, 3.2],
    ])
    metrics = calc_probabilistic_metrics(truth, ensemble)
    assert metrics["COVERAGE"] == pytest.approx(1.0)
    assert metrics["MEAN_WIDTH"] > 0.0
    assert metrics["CRPS"] >= 0.0
    assert metrics["MEDIAN_MAE"] == pytest.approx(0.0)


def test_probabilistic_metrics_penalize_missed_interval():
    truth = np.array([10.0, 10.0])
    good = np.array([[9.0, 10.0, 11.0], [9.0, 10.0, 11.0]])
    bad = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    good_metrics = calc_probabilistic_metrics(truth, good)
    bad_metrics = calc_probabilistic_metrics(truth, bad)
    assert bad_metrics["COVERAGE"] < good_metrics["COVERAGE"]
    assert bad_metrics["INTERVAL_SCORE"] > good_metrics["INTERVAL_SCORE"]
    assert bad_metrics["CRPS"] > good_metrics["CRPS"]


def test_probabilistic_metrics_reject_one_member():
    with pytest.raises(ValueError, match="Minimal dua"):
        calc_probabilistic_metrics(np.array([1.0]), np.array([[1.0]]))
