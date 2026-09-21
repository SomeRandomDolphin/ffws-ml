"""Geometri DAS: koordinat stasiun, topologi cabang, dan layout skematik.

Topologi mengikuti `src/dhompo/data/clusters.py`:
- cabang barat (8 stasiun) dan cabang timur (4 stasiun) yang bertemu di Dhompo.
- `Bd. Sentono` punya koordinat + data tetapi tidak dipakai model -> node sekunder.

Waktu tempuh (travel time) ke Dhompo: 3 stasiun telemetri resmi diambil dari
docs/dhompo/ARCHITECTURE.md (Purwodadi ~3,5 j, AWLR Kademungan ~2 j, Klosod ~1 j).
Untuk stasiun lain diestimasi lewat regresi linear terhadap elevasi yang
melalui ketiga anchor tersebut, lalu dibulatkan ke 0,5 jam.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from dhompo.config import PROJECT_ROOT
from dhompo.data.loader import STATION_META, TARGET_STATION

GEO_CSV = PROJECT_ROOT / "configs" / "dhompo" / "station_geo.csv"

# Topologi cabang (urutan hulu -> hilir), konsisten dengan clusters.py
WEST_BRANCH: list[str] = [
    "Bd. Suwoto",
    "Krajan Timur",
    "Purwodadi",
    "Bd. Baong",
    "Bd. Bakalan",
    "AWLR Kademungan",
    "Bd. Domas",
    "Bd. Grinting",
]
EAST_BRANCH: list[str] = ["Bd. Lecari", "Bd Guyangan", "Sidogiri", "Klosod"]
LOCAL_STATIONS: list[str] = [TARGET_STATION, "Jalan Nasional"]
AUX_STATION: str = "Bd. Sentono"

# Waktu tempuh (jam) ke Dhompo yang resmi — fallback bila gold metadata
# belum ada; sumber tunggal = data/gold/stations.csv (dibangun dari
# travel_hours_anchor di configs/dhompo/dashboard.yaml).
TRAVEL_ANCHORS: dict[str, float] = {
    "Purwodadi": 3.5,
    "AWLR Kademungan": 2.0,
    "Klosod": 1.0,
    TARGET_STATION: 0.0,
    "Jalan Nasional": 0.0,
}

# Regresi linear travel_hours = a*elevasi + b yang melewati 3 anchor
_TRAVEL_A = 0.009434
_TRAVEL_B = 0.793

# Cache lazy waktu-tempuh dari gold stations.csv
_TRAVEL_GOLD: dict[str, float] | None = None

logger = logging.getLogger(__name__)


def load_travel_hours() -> dict[str, float]:
    """Waktu tempuh per stasiun: dari gold stations.csv, fallback anchors."""
    global _TRAVEL_GOLD
    if _TRAVEL_GOLD is None:
        try:
            from dhompo.data.loader import load_gold_stations

            df = load_gold_stations()
            _TRAVEL_GOLD = {
                s: float(v)
                for s, v in df["travel_hours_to_target"].items()
                if v == v
            }
        except (OSError, KeyError, ValueError):
            logger.warning(
                "travel_hours: gold stations.csv tidak terbaca; pakai anchor fallback"
            )
            _TRAVEL_GOLD = {}
    return _TRAVEL_GOLD or dict(TRAVEL_ANCHORS)


@dataclass(frozen=True)
class StationGeo:
    latitude: float
    longitude: float
    confidence: str
    source: str


def load_station_geo(path: str | Path | None = None) -> dict[str, StationGeo]:
    p = Path(path) if path else GEO_CSV
    out: dict[str, StationGeo] = {}
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["station"]] = StationGeo(
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                confidence=row["confidence"],
                source=row["source"],
            )
    return out


def travel_hours(station: str) -> float:
    gold = load_travel_hours()
    if station in gold:
        return gold[station]
    if station == AUX_STATION and station not in gold:
        return 3.5
    elev = STATION_META[station][0]
    hours = _TRAVEL_A * elev + _TRAVEL_B
    return round(hours * 2) / 2.0


def branch_edges(branch: list[str]) -> list[tuple[str, str]]:
    return list(zip(branch, branch[1:]))


def all_edges() -> list[tuple[str, str]]:
    edges = (
        branch_edges(WEST_BRANCH)
        + [(WEST_BRANCH[-1], TARGET_STATION)]
        + branch_edges(EAST_BRANCH)
        + [(EAST_BRANCH[-1], TARGET_STATION)]
        + [(TARGET_STATION, "Jalan Nasional")]
    )
    return edges


def auxiliary_edges() -> list[tuple[str, str]]:
    return [(AUX_STATION, "Purwodadi")]


def schematic_positions() -> dict[str, tuple[float, float]]:
    """Posisi node pada peta DAS skematik (bukan koordinat geografis).

    Sumbu y merepresentasikan waktu tempuh ke Dhompo (hulu di atas),
    cabang barat di kiri dan cabang timur di kanan.
    """
    pos: dict[str, tuple[float, float]] = {}
    pos[TARGET_STATION] = (0.0, 0.0)
    pos["Jalan Nasional"] = (0.0, -1.4)

    for i, st in enumerate(WEST_BRANCH):
        x = -1.9 + (0.2 if i % 2 else -0.2)
        pos[st] = (x, travel_hours(st) * 1.2)
    for i, st in enumerate(EAST_BRANCH):
        x = 1.9 + (0.2 if i % 2 else -0.2)
        pos[st] = (x, travel_hours(st) * 1.2)

    pos[AUX_STATION] = (-2.9, travel_hours(AUX_STATION) * 1.2)
    return pos


def short_name(station: str) -> str:
    if station == "AWLR Kademungan":
        return "Kademungan"
    if station == "Bd Guyangan":
        return "Guyangan"
    return station.replace("Bd. ", "").replace("Bd ", "")


def elevation_of(station: str) -> float | None:
    meta = STATION_META.get(station)
    return meta[0] if meta else None


def travel_ring_radius(travel: float) -> float:
    """Radius isochrone pada sumbu skematik (skala y = travel * 1.2)."""
    return travel * 1.2