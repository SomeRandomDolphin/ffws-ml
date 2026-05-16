"""Tests for ETL quality detectors (ARCHITECTURE.md §4)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from dhompo.etl.quality import (
    BAD_FLAGS,
    DEFAULT_CONFIG,
    QualityConfig,
    QualityFlag,
    compute_quality_flags,
    latest_flags,
)


def _ts_index(n: int, freq: str = "30min") -> pd.DatetimeIndex:
    return pd.date_range(datetime(2022, 11, 21, 0, 0), periods=n, freq=freq)


def _baseline_frame(n: int = 24) -> pd.DataFrame:
    idx = _ts_index(n)
    rng = np.random.default_rng(0)
    data = {
        "Dhompo": rng.uniform(0.5, 1.5, n),
        "Klosod": rng.uniform(0.3, 1.0, n),
        "AWLR Kademungan": rng.uniform(0.3, 1.0, n),
        "Purwodadi": rng.uniform(0.3, 1.0, n),
    }
    return pd.DataFrame(data, index=idx)


class TestMissing:
    def test_nan_flagged_missing(self):
        df = _baseline_frame()
        df.loc[df.index[-1], "Klosod"] = np.nan
        flags = compute_quality_flags(df)
        assert flags.iloc[-1]["Klosod"] == QualityFlag.MISSING.value


class TestFlatline:
    def test_constant_window_flagged(self):
        df = _baseline_frame()
        df.loc[df.index[-6:], "Klosod"] = 0.42
        flags = compute_quality_flags(df)
        assert flags.iloc[-1]["Klosod"] == QualityFlag.FLATLINE.value

    def test_zero_shortcut(self):
        df = _baseline_frame()
        df.loc[df.index[-1], "Klosod"] = 0.0
        flags = compute_quality_flags(df)
        assert flags.iloc[-1]["Klosod"] == QualityFlag.FLATLINE.value

    def test_flood_regime_does_not_flag(self):
        df = _baseline_frame()
        df.loc[df.index[-6:], "Dhompo"] = 9.5  # constant but flood regime
        flags = compute_quality_flags(df)
        assert flags.iloc[-1]["Dhompo"] != QualityFlag.FLATLINE.value


class TestStuck:
    def test_self_static_neighbours_moving(self):
        df = _baseline_frame(n=24)
        df["Klosod"] = 0.5  # completely static
        rng = np.random.default_rng(1)
        df["Purwodadi"] = 0.3 + rng.uniform(0, 0.8, len(df))
        df["AWLR Kademungan"] = 0.3 + rng.uniform(0, 0.8, len(df))
        df["Dhompo"] = 0.5 + rng.uniform(0, 0.8, len(df))
        flags = compute_quality_flags(df)
        # Klosod's static signal in moving basin → STUCK or FLATLINE
        assert flags.iloc[-1]["Klosod"] in {
            QualityFlag.STUCK.value,
            QualityFlag.FLATLINE.value,
        }


class TestOutOfRange:
    def test_negative_value(self):
        df = _baseline_frame()
        df.loc[df.index[-1], "Klosod"] = -0.1
        flags = compute_quality_flags(df)
        assert flags.iloc[-1]["Klosod"] == QualityFlag.OUT_OF_RANGE.value

    def test_step_jump(self):
        df = _baseline_frame()
        df.loc[df.index[-2], "Klosod"] = 0.5
        df.loc[df.index[-1], "Klosod"] = 5.0  # +4.5 m in 30 min
        flags = compute_quality_flags(df)
        assert flags.iloc[-1]["Klosod"] == QualityFlag.OUT_OF_RANGE.value

    def test_above_training_max(self):
        df = _baseline_frame()
        df.loc[df.index[-1], "Klosod"] = 3.0
        cfg = QualityConfig(training_max={"Klosod": 1.5})
        flags = compute_quality_flags(df, cfg)
        assert flags.iloc[-1]["Klosod"] == QualityFlag.OUT_OF_RANGE.value


class TestStale:
    def test_telemetry_older_than_30_min(self):
        df = _baseline_frame()
        reference = df.index[-1] + pd.Timedelta(minutes=45)
        flags = compute_quality_flags(df, reference_time=reference)
        assert flags.iloc[-1]["Klosod"] == QualityFlag.STALE.value


class TestHysteresis:
    def test_three_clean_readings_clear(self):
        df = _baseline_frame(n=24)
        df.loc[df.index[10:16], "Klosod"] = 0.42  # flatline window
        # then return to varying values
        rng = np.random.default_rng(2)
        df.loc[df.index[16:], "Klosod"] = rng.uniform(0.5, 1.0, len(df) - 16)
        flags = compute_quality_flags(df)
        # right after flatline ends, three readings should still be sticky
        assert flags.iloc[16]["Klosod"] == QualityFlag.FLATLINE.value
        # eventually clears
        assert flags.iloc[-1]["Klosod"] == QualityFlag.OK.value


class TestPrecedence:
    def test_missing_beats_flatline(self):
        df = _baseline_frame()
        df.loc[df.index[-6:], "Klosod"] = 0.42
        df.loc[df.index[-1], "Klosod"] = np.nan
        flags = compute_quality_flags(df)
        assert flags.iloc[-1]["Klosod"] == QualityFlag.MISSING.value


class TestLatestFlags:
    def test_dict_output_shape(self):
        df = _baseline_frame()
        flags = compute_quality_flags(df)
        latest = latest_flags(flags)
        assert set(latest.keys()) == set(df.columns)
        assert all(isinstance(v, str) for v in latest.values())


def test_bad_flags_membership():
    """Ensure routing-critical flags are all classified as bad."""
    assert QualityFlag.MISSING in BAD_FLAGS
    assert QualityFlag.FLATLINE in BAD_FLAGS
    assert QualityFlag.STUCK in BAD_FLAGS
    assert QualityFlag.OUT_OF_RANGE in BAD_FLAGS
    # STALE and OK are intentionally not in BAD_FLAGS at the routing layer —
    # STALE on telemetry stations is escalated upstream by the cadence check.
    assert QualityFlag.STALE not in BAD_FLAGS
    assert QualityFlag.OK not in BAD_FLAGS
