"""Tes generator hujan stokastik Markov-Gamma."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dhompo.data.rainfall_sim import (
    RainfallParameters,
    fit_rainfall_parameters,
    simulate_rainfall,
)


def test_fit_rainfall_parameters_recovers_valid_distribution():
    index = pd.date_range("2023-01-01", periods=12, freq="30min")
    rainfall = pd.Series([0, 0, 1, 2, 0, 0, 0, 3, 4, 2, 0, 0], index=index)
    parameters = fit_rainfall_parameters(rainfall, [1.0] * 12)
    assert 0.0 < parameters.p_wet_after_dry < 1.0
    assert 0.0 < parameters.p_wet_after_wet < 1.0
    assert parameters.gamma_shape > 0.0
    assert parameters.gamma_scale > 0.0


def test_simulated_rainfall_is_reproducible_and_nonnegative():
    parameters = RainfallParameters(
        p_wet_after_dry=0.10,
        p_wet_after_wet=0.70,
        gamma_shape=1.5,
        gamma_scale=2.0,
        monthly_multipliers=(1.0,) * 12,
        max_rain_mm=10.0,
    )
    first = simulate_rainfall("2024-01-01", 2_000, parameters, seed=7)
    second = simulate_rainfall("2024-01-01", 2_000, parameters, seed=7)
    assert first.equals(second)
    assert first.index.freqstr == "30min"
    assert np.isfinite(first).all()
    assert first.min() >= 0.0
    assert first.max() <= 10.0
    assert 0.0 < float((first > 0).mean()) < 1.0


def test_parameter_round_trip():
    original = RainfallParameters(
        p_wet_after_dry=0.1,
        p_wet_after_wet=0.5,
        gamma_shape=2.0,
        gamma_scale=3.0,
        monthly_multipliers=tuple(range(1, 13)),
        max_rain_mm=20.0,
    )
    assert RainfallParameters.from_dict(original.to_dict()) == original


def test_fit_rejects_invalid_monthly_multiplier_count():
    rainfall = pd.Series([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="12 nilai"):
        fit_rainfall_parameters(rainfall, [1.0] * 11)
