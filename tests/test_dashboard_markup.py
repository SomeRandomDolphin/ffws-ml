"""Tes markup UI dashboard: badge tier model, strip forecast, kartu detail."""

from __future__ import annotations

import pytest
from dash import dcc, html

from dashboard.callbacks import (
    _detail_markup,
    _forecast_strip_markup,
    _model_tier_markup,
    _rain_markup,
)
from dashboard.simulation import build_frame
from tests.test_dashboard_figures import _synthetic_demo


@pytest.fixture()
def demo():
    return _synthetic_demo()


def test_model_tier_markup_ok(demo):
    markup = _model_tier_markup(demo)
    assert isinstance(markup, html.Div)
    assert "tier A" in markup.children


def test_model_tier_markup_degraded(demo):
    demo.serving_tier = "B"
    demo.degraded = ["h1:plausibility_fallback"]
    markup = _model_tier_markup(demo)
    text = markup.children
    assert "tier B" in text and "patch: 1" in text
    assert markup.title == "h1:plausibility_fallback"


def test_forecast_strip_has_one_chip_per_horizon(demo):
    chips = _forecast_strip_markup(demo)
    assert len(chips) == 5
    for chip in chips:
        assert chip.className == "forecast-chip"


def test_detail_markup_contains_spark_and_rain(demo):
    frame = build_frame(demo, 0)
    parts = _detail_markup(frame, "Dhompo", demo)
    graphs = [p for p in parts if isinstance(p, dcc.Graph)]
    assert graphs, "sparkline harus ada di kartu detail"


def test_rain_markup_none_when_absent(demo):
    assert _rain_markup(demo) is None


def test_rain_markup_present_when_available(demo):
    import pandas as pd

    demo.rainfall = pd.Series(1.0, index=demo.df.index)
    label = _rain_markup(demo)
    assert label is not None
    assert "mm" in label.children
