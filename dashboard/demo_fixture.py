"""Deterministic presentation data. No model or historical dataset is loaded."""

from functools import lru_cache

import numpy as np
import pandas as pd

from dashboard.geo import load_station_geo

COLORS = {
    "Normal": "#32956B",
    "Meningkat": "#D9AE32",
    "Waspada": "#E48732",
    "Bahaya": "#CB4949",
}
ORDER = {name: i for i, name in enumerate(COLORS)}
NOW = pd.Timestamp("2026-02-18 12:00", tz="Asia/Jakarta")


@lru_cache(maxsize=1)
def fixture():
    times = pd.date_range(
        NOW - pd.Timedelta(hours=24), NOW + pd.Timedelta(hours=5), freq="30min"
    )
    hours = np.arange(-24, 5.1, 0.5)
    stations = {}
    for i, (name, geo) in enumerate(load_station_geo().items()):
        # An upstream pulse arrives earlier; amplitudes vary by local catchment.
        progress = (geo.latitude + 7.85) / 0.23
        peak = -3.5 + 7 * progress
        base = 1.8 + (i % 5) * 0.46
        amplitude = [0.24, 0.65, 1.2, 0.34, 1.65][i % 5]
        if name == "Dhompo":
            base, amplitude, peak = 7.62, 1.48, 3.0
        values = base + amplitude * np.exp(-0.5 * ((hours - peak) / 3.5) ** 2)
        values += 0.025 * np.sin(hours * 0.8 + i) * (hours < -6)
        stations[name] = dict(
            geo=geo, values=values, alert=base + 0.70, danger=base + 1.32
        )
    rain = 17 * np.exp(-0.5 * ((hours + 5) / 2) ** 2) + 5 * np.exp(
        -0.5 * ((hours + 10) / 1.5) ** 2
    )
    return dict(times=times, stations=stations, rain=rain)


def snapshot(lead=0):
    index = 48 + int(lead) * 2
    rows = []
    for name, station in fixture()["stations"].items():
        value = float(station["values"][index])
        delta = value - float(station["values"][index - 6])
        status = (
            "Bahaya"
            if value >= station["danger"]
            else "Waspada"
            if value >= station["alert"]
            else "Meningkat"
            if delta >= 0.25
            else "Normal"
        )
        rows.append(
            dict(
                name=name,
                value=value,
                delta=delta,
                status=status,
                color=COLORS[status],
                **station,
            )
        )
    return sorted(rows, key=lambda r: (-ORDER[r["status"]], -r["delta"], r["name"]))
