"""Load and preprocess Dhompo sensor data."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

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

# Mapping from generated-data column names → canonical names
GENERATED_COLUMN_MAP: dict[str, str] = {
    "Suwoto": "Bd. Suwoto",
    "Kademungan": "AWLR Kademungan",
    "bd. Domas": "Bd. Domas",
    "Jl. Pantura": "Jalan Nasional",
    "Guyangan": "Bd Guyangan",
    "Bd Lecari": "Bd. Lecari",
}

RAINFALL_COLUMN: str = "Curah hujan"


class GeneratedData(NamedTuple):
    """Result of loading the generated-2023 Excel file."""
    stations: pd.DataFrame  # station columns with canonical names, DatetimeIndex
    rainfall: pd.Series     # rainfall series aligned to same index


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


def load_generated_data(path: str | Path) -> GeneratedData:
    """Load the 'Data generated 2023' Excel file.

    The file has header at row 1 (0-indexed), an unnamed index column,
    a 'Curah hujan' (rainfall) column, a 'Time' column, and station columns
    with slightly different names than the canonical set.

    Parameters
    ----------
    path:
        Path to the Excel file.

    Returns
    -------
    GeneratedData
        Named tuple with `stations` (DataFrame) and `rainfall` (Series).
    """
    df = pd.read_excel(path, header=1)

    # Drop unnamed index column
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)])

    # Extract rainfall before renaming
    rainfall = df[RAINFALL_COLUMN].copy()

    # Parse datetime and set as index
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time")
    df.index.name = "Datetime"
    df = df.asfreq("30min")

    rainfall.index = df.index
    # NaN rainfall = no rain recorded → fill with 0
    rainfall = rainfall.fillna(0.0)

    # Rename station columns to canonical names
    df = df.rename(columns=GENERATED_COLUMN_MAP)

    # Separate station data from rainfall
    station_cols = [c for c in df.columns if c != RAINFALL_COLUMN]
    stations = df[station_cols]

    rainfall = rainfall.asfreq("30min")
    rainfall.name = RAINFALL_COLUMN

    return GeneratedData(stations=stations, rainfall=rainfall)


class DataSegment(NamedTuple):
    """A contiguous time segment of sensor data."""
    df: pd.DataFrame
    label: str
    rainfall: pd.Series | None


def load_combined_data(
    clean_path: str | Path,
    generated_path: str | Path,
) -> list[DataSegment]:
    """Load both datasets as separate segments (not concatenated).

    The two datasets have a ~26-day gap (Dec 5 2022 → Jan 1 2023), so they
    are returned as a list of segments to prevent lag/rolling features from
    being contaminated by the gap.

    Parameters
    ----------
    clean_path:
        Path to data-clean.csv.
    generated_path:
        Path to Data generated 2023.xlsx.

    Returns
    -------
    list[DataSegment]
        Two segments: [clean_2022, generated_2023].
    """
    df_clean = load_data(clean_path)
    gen = load_generated_data(generated_path)

    # Use only the canonical station columns present in both datasets
    common_cols = [c for c in ALL_STATIONS if c in df_clean.columns and c in gen.stations.columns]
    df_clean = df_clean[common_cols]
    gen_stations = gen.stations[common_cols]

    return [
        DataSegment(df=df_clean, label="2022_clean", rainfall=None),
        DataSegment(df=gen_stations, label="2023_generated", rainfall=gen.rainfall),
    ]
