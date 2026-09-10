from __future__ import annotations

import numpy as np
import pandas as pd

from dhompo.data.urban_features import (
    align_urban_features_targets,
    build_urban_forecast_features,
    build_urban_targets,
)
from dhompo.data.urban_loader import QUALITY_MISSING, QUALITY_OK, QUALITY_STALE


def test_urban_targets_use_two_30min_steps_per_hour():
    idx = pd.date_range("2026-01-01", periods=8, freq="30min", name="Datetime")
    canonical = pd.DataFrame({"target": np.arange(8, dtype=float)}, index=idx)

    targets = build_urban_targets(canonical, target_column="target", horizons=[1, 2])

    assert targets[1].loc[idx[0]] == 2.0
    assert targets[2].loc[idx[0]] == 4.0


def test_features_include_quality_flag_indicators():
    idx = pd.date_range("2026-01-01", periods=30, freq="30min", name="Datetime")
    values = pd.DataFrame({"target": np.arange(30, dtype=float)}, index=idx)
    flags = pd.DataFrame({ "target": [QUALITY_OK] * 30 }, index=idx)
    flags.loc[idx[-2], "target"] = QUALITY_STALE
    flags.loc[idx[-1], "target"] = QUALITY_MISSING

    X = build_urban_forecast_features(
        values,
        feature_columns=["target"],
        quality_flags=flags,
        lag_steps=[1],
        rolling_windows=[(2, "1h")],
    )

    assert "target_flag_stale" in X.columns
    assert "target_flag_missing" in X.columns
    assert X.loc[idx[-2], "target_flag_stale"] == 1.0
    assert X.loc[idx[-1], "target_flag_missing"] == 1.0


def test_align_drops_rows_without_observed_future_target():
    idx = pd.date_range("2026-01-01", periods=30, freq="30min", name="Datetime")
    canonical = pd.DataFrame({"target": np.arange(30, dtype=float)}, index=idx)
    canonical.loc[idx[20], "target"] = np.nan
    values = canonical.ffill()

    X = build_urban_forecast_features(
        values,
        feature_columns=["target"],
        include_quality_flags=False,
        lag_steps=[1],
        rolling_windows=[(2, "1h")],
    )
    targets = build_urban_targets(canonical, target_column="target", horizons=[1])
    X_aligned, y_aligned = align_urban_features_targets(X, targets)

    assert idx[18] not in X_aligned.index
    assert idx[18] not in y_aligned[1].index


def test_missing_feature_values_use_sentinel_when_flags_are_available():
    idx = pd.date_range("2026-01-01", periods=30, freq="30min", name="Datetime")
    values = pd.DataFrame({"sparse": [1.0] + [np.nan] * 29}, index=idx)
    flags = pd.DataFrame(
        {"sparse": [QUALITY_OK] + [QUALITY_MISSING] * 29},
        index=idx,
    )

    X = build_urban_forecast_features(
        values,
        feature_columns=["sparse"],
        quality_flags=flags,
        lag_steps=[1],
        rolling_windows=[(2, "1h")],
    )

    assert not X.empty
    assert X["sparse_t0"].iloc[-1] == 0.0
    assert X["sparse_flag_missing"].iloc[-1] == 1.0
