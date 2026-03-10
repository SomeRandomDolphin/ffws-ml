"""Feature engineering for Dhompo multi-horizon flood forecasting.

Ported from research/create_02_modeling.py — direct multi-step strategy:
build one feature matrix at time t, then shift target by h steps.

Total features: ~160
  t0 values     : 13
  lag 1-3       : 39
  rolling mean  : 39 (3h/6h/12h per station)
  rolling std   : 39 (3h/6h/12h per station)
  rate of change: 26 (diff1/diff2 per station)
  temporal      :  4 (hour_sin, hour_cos, dayofweek, is_night)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import UPSTREAM_STATIONS, TARGET_STATION


def build_forecast_features(
    df: pd.DataFrame,
    upstream_stations: list[str] | None = None,
    target: str = TARGET_STATION,
) -> pd.DataFrame:
    """Build feature matrix at time t for predicting target(t+h).

    Parameters
    ----------
    df:
        Raw sensor DataFrame with DatetimeIndex.
    upstream_stations:
        List of upstream station column names. Defaults to the canonical list.
    target:
        Name of the target station column.

    Returns
    -------
    pd.DataFrame
        Feature matrix with NaN rows dropped (first ~24 rows removed due to lag/rolling).
    """
    if upstream_stations is None:
        upstream_stations = UPSTREAM_STATIONS

    all_stations = upstream_stations + [target]
    cols: dict[str, pd.Series] = {}

    # 1. Current value + lags t-1, t-2, t-3
    for st in all_stations:
        cols[f"{st}_t0"] = df[st]
        for lag in range(1, 4):
            cols[f"{st}_lag{lag}"] = df[st].shift(lag)

    # 2. Rolling mean & std (window in 30-min steps → 3h=6, 6h=12, 12h=24)
    for st in all_stations:
        for window, label in [(6, "3h"), (12, "6h"), (24, "12h")]:
            roll = df[st].rolling(window)
            cols[f"{st}_rmean_{label}"] = roll.mean()
            cols[f"{st}_rstd_{label}"] = roll.std()

    # 3. Rate of change
    for st in all_stations:
        cols[f"{st}_diff1"] = df[st].diff(1)
        cols[f"{st}_diff2"] = df[st].diff(2)

    # 4. Temporal features (cyclical encoding)
    hour = df.index.hour + df.index.minute / 60.0
    cols["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=df.index)
    cols["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=df.index)
    cols["dayofweek"] = pd.Series(df.index.dayofweek.astype(float), index=df.index)
    cols["is_night"] = pd.Series(
        ((df.index.hour >= 19) | (df.index.hour < 6)).astype(float), index=df.index
    )

    feat = pd.concat(cols, axis=1)
    feat = feat.dropna()
    return feat


def build_targets(
    df: pd.DataFrame,
    horizons: list[int],
    horizon_steps: dict[int, int],
    target: str = TARGET_STATION,
) -> dict[int, pd.Series]:
    """Build target series for each horizon.

    Parameters
    ----------
    df:
        Raw sensor DataFrame.
    horizons:
        List of forecast horizons in hours, e.g. [1, 2, 3, 4, 5].
    horizon_steps:
        Mapping from horizon → number of 30-min steps, e.g. {1: 2, 2: 4, ...}.
    target:
        Name of the target column.

    Returns
    -------
    dict mapping horizon (int) → pd.Series of future target values.
    """
    return {h: df[target].shift(-horizon_steps[h]) for h in horizons}


def align_features_targets(
    X: pd.DataFrame,
    y_horizons: dict[int, pd.Series],
) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """Intersect valid indices across all horizons and the feature matrix.

    Returns aligned (X, y_horizons) with no NaN rows.
    """
    valid_idx = X.index
    for h, y in y_horizons.items():
        valid_idx = valid_idx.intersection(y.dropna().index)

    X_aligned = X.loc[valid_idx]
    y_aligned = {h: y.loc[valid_idx] for h, y in y_horizons.items()}
    return X_aligned, y_aligned
