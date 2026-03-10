"""Load and preprocess Dhompo sensor data."""

from pathlib import Path

import pandas as pd

# Canonical station ordering (elevation descending)
STATION_META: dict[str, tuple[float, int]] = {
    "Bd. Suwoto":      (503,  1),
    "Krajan Timur":    (335,  2),
    "Purwodadi":       (287,  3),
    "Bd. Baong":       (169,  4),
    "Bd. Lecari":      (167,  5),
    "Bd. Bakalan":     (136,  6),
    "AWLR Kademungan": (128,  7),
    "Bd. Domas":       (57,   8),
    "Bd Guyangan":     (32,   9),
    "Bd. Grinting":    (28,  10),
    "Sidogiri":        (24,  11),
    "Klosod":          (22,  12),
    "Dhompo":          (7,   13),
    "Jalan Nasional":  (1.8, 14),
}

UPSTREAM_STATIONS: list[str] = [
    "Bd. Suwoto",
    "Krajan Timur",
    "Purwodadi",
    "Bd. Lecari",
    "Bd. Bakalan",
    "Bd. Baong",
    "AWLR Kademungan",
    "Bd Guyangan",
    "Sidogiri",
    "Bd. Domas",
    "Klosod",
    "Bd. Grinting",
]

ALL_STATIONS: list[str] = list(STATION_META.keys())
TARGET_STATION: str = "Dhompo"

_DEFAULT_DATA_PATH = Path(__file__).parents[4] / "data" / "data-clean.csv"


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load cleaned sensor data, indexed by Datetime at 30-minute frequency.

    Parameters
    ----------
    path:
        CSV file path. Defaults to ``data/data-clean.csv`` relative to project root.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex at 30-min frequency.
    """
    csv_path = Path(path) if path else _DEFAULT_DATA_PATH
    df = pd.read_csv(csv_path, parse_dates=["Datetime"], index_col="Datetime")
    df = df.asfreq("30min")
    return df
