"""Simulasi frame per langkah lead untuk animasi time slider.

Semantik pada waktu simulasi `now + lead` (lead dalam jam, 0..5):

- Stasiun Dhompo: nilai observasi terakhir (lead=0) atau prediksi model
  `h{lead}` (lead>=1). Ini satu-satunya nilai yang benar-benar keluaran ML.
- Stasiun hulu: nilai observasi yang dipropagasi — yaitu observasi pada waktu
  `now + lead - travel(stasiun)`, di mana `travel` adalah waktu tempuh sinyal
  ke Dhompo. Bila waktu tersebut berada di luar jendela observasi, nilai
  dijepit ke observasi terbaru yang tersedia. Ini BUKAN prediksi ML; diberi
  label "propagasi observasi" pada UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dhompo.data.loader import TARGET_STATION

from dashboard.geo import travel_hours
from dashboard.status import (
    StationStatus,
    Thresholds,
    compute_station_status,
    delta_over_window,
    risk_score,
)

MAX_LEAD = 5


@dataclass
class LeadFrame:
    lead: int
    values: dict[str, float]
    statuses: dict[str, StationStatus]
    max_predicted: float
    riskiest: str
    riskiest_status: str
    area_waspada: int


def _station_position(
    window: pd.DataFrame, station: str, lead: int
) -> tuple[float, int]:
    """Nilai + indeks observasi untuk stasiun pada lead tertentu."""
    series = window[station].dropna()
    if series.empty:
        return float("nan"), -1

    if station == TARGET_STATION:
        if lead == 0:
            return float(series.iloc[-1]), len(window[station]) - 1
        return float("nan"), len(window[station]) - 1  # diisi prediksi di luar

    travel = travel_hours(station)
    target_time = window.index[-1] + pd.Timedelta(hours=lead - travel)

    if target_time < window.index[0]:
        return float(series.iloc[0]), 0
    if target_time >= window.index[-1]:
        return float(series.iloc[-1]), len(window[station]) - 1

    pos = window.index.get_indexer([target_time], method="nearest")[0]
    pos = min(pos, len(window) - 1)
    return float(window[station].iloc[pos]), pos


def build_frame(demo, lead: int) -> LeadFrame:
    """Susun frame lengkap untuk satu langkah slider."""
    window = demo.window
    rising_delta = float(demo.config["thresholds"].get("rising_delta_m", 0.25))

    values: dict[str, float] = {}
    statuses: dict[str, StationStatus] = {}
    for station in window.columns:
        value, pos = _station_position(window, station, lead)
        if station == TARGET_STATION:
            if lead >= 1:
                value = float(demo.predictions.get(f"h{lead}", value))
            elif value != value:
                value = float(window[station].dropna().iloc[-1])

        threshold: Thresholds = demo.thresholds.get(
            station, Thresholds(alert=0.0, danger=0.0)
        )
        if value != value:
            value = float(window[station].dropna().iloc[-1])
        series = window[station].dropna()
        delta = delta_over_window(series.iloc[: pos + 1]) if pos >= 0 else 0.0
        statuses[station] = compute_station_status(
            station, value, threshold, delta, rising_delta
        )
        values[station] = value

    max_predicted = max(demo.predictions.values())
    riskiest = max(
        (st for st in statuses if st != "Jalan Nasional"),
        key=lambda st: risk_score(statuses[st]),
    )
    area_waspada = sum(
        1 for st in statuses if statuses[st].status in ("waspada", "bahaya")
    )
    return LeadFrame(
        lead=lead,
        values=values,
        statuses=statuses,
        max_predicted=max_predicted,
        riskiest=riskiest,
        riskiest_status=statuses[riskiest].status,
        area_waspada=area_waspada,
    )