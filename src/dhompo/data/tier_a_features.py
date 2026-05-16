"""Per-station feature builders for the Tier-A adaptive model.

Shared by the training loop and the serving predictor so the on-disk feature
contract has exactly one definition. The seven features per station match the
training contract documented in ``training/run_tier_a_adaptive.py``:

    [t0, lag1, lag2, lag3, rolling_mean_3h, rolling_std_3h, diff1]

The autoregressive lag tensor takes the last ``AR_LAG_DIM`` readings of the
target station (Dhompo), shift 0..AR_LAG_DIM-1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dhompo.data.loader import ALL_STATIONS, TARGET_STATION

FEATURES_PER_STATION: int = 7
AR_LAG_DIM: int = 6
HORIZON_STEPS_PER_HOUR: int = 2

_ROLLING_WINDOW: int = 6


def build_per_station_features(
    df: pd.DataFrame, stations: list[str] | tuple[str, ...] = tuple(ALL_STATIONS),
) -> np.ndarray:
    """Reshape sensor history into (n_timesteps, n_stations, features_per_station).

    Missing stations in the input DataFrame are zero-filled and downstream
    callers are expected to mask them out via the model's mask tensor.
    """
    n_steps = len(df)
    n_stations = len(stations)
    out = np.zeros((n_steps, n_stations, FEATURES_PER_STATION), dtype=np.float32)
    for s_idx, station in enumerate(stations):
        if station not in df.columns:
            continue
        col = df[station]
        out[:, s_idx, 0] = col.to_numpy()
        out[:, s_idx, 1] = col.shift(1).to_numpy()
        out[:, s_idx, 2] = col.shift(2).to_numpy()
        out[:, s_idx, 3] = col.shift(3).to_numpy()
        out[:, s_idx, 4] = col.rolling(window=_ROLLING_WINDOW,
                                       min_periods=_ROLLING_WINDOW).mean().to_numpy()
        out[:, s_idx, 5] = col.rolling(window=_ROLLING_WINDOW,
                                       min_periods=_ROLLING_WINDOW).std(ddof=0).to_numpy()
        out[:, s_idx, 6] = col.diff().to_numpy()
    return out


def build_ar_lags(df: pd.DataFrame, target: str = TARGET_STATION) -> np.ndarray:
    """Stack shift(0..AR_LAG_DIM-1) of the target column into (n_timesteps, AR_LAG_DIM)."""
    series = df[target]
    lags = np.stack([series.shift(i).to_numpy() for i in range(AR_LAG_DIM)], axis=-1)
    return lags.astype(np.float32)


def build_targets(df: pd.DataFrame, target: str = TARGET_STATION) -> np.ndarray:
    """Stack h+1..h+5 hour targets into (n_timesteps, 5)."""
    series = df[target]
    horizons = [
        series.shift(-h * HORIZON_STEPS_PER_HOUR).to_numpy()
        for h in range(1, 6)
    ]
    return np.stack(horizons, axis=-1).astype(np.float32)
