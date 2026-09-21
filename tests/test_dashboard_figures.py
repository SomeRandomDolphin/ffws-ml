"""Smoke test pembangun figur dashboard: semua figur valid untuk tiap lead."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from dhompo.data.loader import STATION_META, TARGET_STATION

from dashboard.data import DemoData
from dashboard.figures import (
    build_geo_figure,
    build_hydro_figure,
    build_profile_figure,
    build_spark_figure,
)
from dashboard.geo import load_station_geo
from dashboard.simulation import build_frame
from dashboard.status import Thresholds


def _synthetic_demo() -> DemoData:
    stations = list(STATION_META.keys())
    idx = pd.date_range(datetime(2022, 10, 4, 5, 30), periods=25, freq="30min")
    cols = {}
    thresholds = {}
    for st in stations:
        base = STATION_META[st][0]
        cols[st] = [base + 0.6 + 0.02 * i for i in range(25)]
        thresholds[st] = Thresholds(alert=base + 0.3, danger=base + 0.9)
    df = pd.DataFrame(cols, index=idx)
    last = float(df[TARGET_STATION].iloc[-1])
    predictions = {f"h{h}": round(last + 0.2 * h, 4) for h in range(1, 6)}
    config = {
        "thresholds": {"rising_delta_m": 0.25},
        "demo": {},
        "data": {},
        "propagation": {},
    }
    return DemoData(
        df=df,
        window=df,
        last_ts=df.index[-1],
        predictions=predictions,
        thresholds=thresholds,
        config=config,
        serving_tier="A",
        model_versions={},
        degraded=[],
    )


@pytest.fixture(scope="module")
def demo():
    return _synthetic_demo()


def _real_traces(fig):
    out = []
    for t in fig.data:
        x = getattr(t, "x", None)
        if x is None:
            continue
        if hasattr(x, "tolist"):
            x = x.tolist()
        if x and x[0] is not None:
            out.append(t)
    return out


@pytest.mark.parametrize("lead", [0, 1, 5])
def test_all_figures_build_for_every_lead(demo, lead):
    frame = build_frame(demo, lead)
    geo = load_station_geo()

    geo_fig = build_geo_figure(frame, geo)
    profile = build_profile_figure(demo, frame)
    hydro = build_hydro_figure(demo, lead)

    assert len(geo_fig.data) >= 2
    assert all(getattr(trace, "mode", "") == "markers" for trace in geo_fig.data)
    assert len(_real_traces(profile)) >= 5
    assert len(_real_traces(hydro)) >= 4


def test_geo_is_marker_only_and_has_station_identity(demo):
    frame = build_frame(demo, 0)
    fig = build_geo_figure(frame, load_station_geo(), TARGET_STATION)
    assert all(trace.mode == "markers" for trace in fig.data)
    assert TARGET_STATION in list(fig.data[0].customdata)
    assert fig.layout.map.style == "carto-positron-nolabels"


def test_hydro_has_threshold_bands(demo):
    fig = build_hydro_figure(demo, 2)
    assert len(fig.layout.shapes) >= 2  # band waspada + bahaya


def test_spark_builds_for_any_station(demo):
    for station in ("Dhompo", "Bd. Suwoto", "Klosod"):
        fig = build_spark_figure(demo, station)
        assert fig.data is not None


def test_frame_values_populated(demo):
    frame = build_frame(demo, 3)
    for station in demo.window.columns:
        assert station in frame.values
        assert frame.values[station] == frame.values[station]  # bukan NaN
        assert frame.statuses[station].status in ("normal", "meningkat", "waspada", "bahaya")
