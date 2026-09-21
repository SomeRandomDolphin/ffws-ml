"""Tes simulasi frame (time slider) dashboard DAS Dhompo."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from dashboard.data import DemoData
from dashboard.simulation import build_frame
from dashboard.status import Thresholds

_STATIONS = ["Bd. Suwoto", "Klosod", "Dhompo", "Jalan Nasional"]


def _make_demo(rising: bool = True) -> DemoData:
    idx = pd.date_range(datetime(2022, 10, 4, 5, 30), periods=25, freq="30min")
    rng = range(25)
    base = [8.0 + 0.08 * i for i in rng] if rising else [12.0] * 25
    df = pd.DataFrame(
        {
            "Bd. Suwoto": [x + 0.5 for x in base],
            "Klosod": [x + 0.2 for x in base],
            "Dhompo": base,
            "Jalan Nasional": [x - 0.1 for x in base],
        },
        index=idx,
    )
    last = base[-1]
    predictions = {f"h{h}": round(last + 0.4 * h, 4) for h in range(1, 6)}
    thresholds = {
        st: Thresholds(alert=11.0, danger=14.0) for st in _STATIONS
    }
    config = {"thresholds": {"rising_delta_m": 0.25}, "demo": {}, "data": {}, "propagation": {}}
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


def test_lead0_dhompo_uses_last_observed():
    demo = _make_demo()
    frame = build_frame(demo, 0)
    assert frame.values["Dhompo"] == pytest.approx(float(demo.window["Dhompo"].iloc[-1]))


def test_lead_h_dhompo_uses_prediction():
    demo = _make_demo()
    for h in (1, 3, 5):
        frame = build_frame(demo, h)
        assert frame.values["Dhompo"] == pytest.approx(demo.predictions[f"h{h}"])


def test_propagation_holds_last_observed_beyond_data():
    # Klosod travel 1 jam; pada lead 5 nilai target melewati data -> jepit ke observasi terakhir
    demo = _make_demo(rising=True)
    frame = build_frame(demo, 5)
    assert frame.values["Klosod"] == pytest.approx(float(demo.window["Klosod"].iloc[-1]))


def test_propagation_at_lead0_uses_past_observation():
    # Klosod travel 1 jam; lead 0 -> nilai = observasi 1 jam sebelum waktu akhir
    demo = _make_demo(rising=True)
    frame = build_frame(demo, 0)
    expected = float(demo.window["Klosod"].iloc[-3])  # -1 jam = 2 langkah
    assert frame.values["Klosod"] == pytest.approx(expected)


def test_frame_populates_statuses_for_all_stations():
    demo = _make_demo()
    frame = build_frame(demo, 2)
    assert set(frame.statuses) == set(_STATIONS)
    assert frame.values["Dhompo"] == pytest.approx(demo.predictions["h2"])
    assert frame.riskiest in _STATIONS


def test_area_waspada_counts_alert_or_danger():
    demo = _make_demo(rising=True)
    frame = build_frame(demo, 0)
    assert isinstance(frame.area_waspada, int)
    assert frame.area_waspada >= 0
    assert frame.max_predicted == max(demo.predictions.values())