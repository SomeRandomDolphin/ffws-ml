"""Tes klasifikasi status tinggi air pada dashboard."""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.status import (
    Thresholds,
    classify_status,
    compute_station_status,
    compute_thresholds,
    delta_over_window,
    risk_score,
)


def test_classify_status_priority_bahaya():
    assert classify_status(value=12.0, alert=10.0, danger=11.0, delta_3h=-1.0, rising_delta_m=0.25) == "bahaya"


def test_classify_status_waspada_over_normal():
    assert classify_status(value=10.5, alert=10.0, danger=11.0, delta_3h=-1.0, rising_delta_m=0.25) == "waspada"


def test_classify_status_meningkat_with_positive_delta():
    assert classify_status(value=9.0, alert=10.0, danger=11.0, delta_3h=0.5, rising_delta_m=0.25) == "meningkat"


def test_classify_status_normal_otherwise():
    assert classify_status(value=8.0, alert=10.0, danger=11.0, delta_3h=0.1, rising_delta_m=0.25) == "normal"


def test_delta_over_window():
    s = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    assert delta_over_window(s, lookback=3) == pytest.approx(0.3)


def test_delta_over_window_short_series():
    assert delta_over_window(pd.Series([1.0, 1.1]), lookback=6) == 0.0


def test_compute_thresholds_percentiles_and_danger_floor():
    df = pd.DataFrame({"A": list(range(1, 101)), "B": list(range(101, 201))})
    th = compute_thresholds(df, alert_quantile=0.9, danger_quantile=0.99)
    assert th["A"].alert == pytest.approx(90.1)
    assert th["A"].danger == pytest.approx(99.01)
    assert th["A"].danger > th["A"].alert


def test_compute_thresholds_override_and_floor():
    df = pd.DataFrame({"A": list(range(1, 101))})
    th = compute_thresholds(
        df,
        alert_quantile=0.9,
        danger_quantile=0.99,
        overrides={"A": {"alert": 50.0, "danger": 45.0}},
    )
    assert th["A"].alert == 50.0
    assert th["A"].danger == 50.0  # dijepit agar danger >= alert


def test_compute_station_status_bundle():
    st = compute_station_status(
        "X", value=12.0, threshold=Thresholds(10.0, 11.0), delta_3h=0.4, rising_delta_m=0.25
    )
    assert st.status == "bahaya"
    assert st.value == 12.0
    assert st.delta_3h == 0.4


def test_risk_score():
    st = compute_station_status("X", value=11.0, threshold=Thresholds(10.0, 12.0), delta_3h=0.0, rising_delta_m=0.25)
    assert risk_score(st) == pytest.approx(0.5)