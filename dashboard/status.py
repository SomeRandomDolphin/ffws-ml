"""Klasifikasi status tinggi air per stasiun.

Status dibedakan menjadi empat level: normal, meningkat, waspada, bahaya.
Threshold waspada/bahaya diturunkan dari persentil data tiap stasiun agar
menyesuaikan skala elevasi yang berbeda-beda (lihat configs/dhompo/dashboard.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

# 6 langkah 30 menit = 3 jam untuk deteksi tren naik
LOOKBACK_STEPS = 6


@dataclass(frozen=True)
class Thresholds:
    alert: float
    danger: float


@dataclass(frozen=True)
class StationStatus:
    station: str
    value: float
    status: str
    alert: float
    danger: float
    delta_3h: float


def compute_thresholds(
    df: pd.DataFrame,
    alert_quantile: float = 0.90,
    danger_quantile: float = 0.99,
    overrides: Mapping[str, Mapping[str, float | None]] | None = None,
) -> dict[str, Thresholds]:
    """Hitung threshold waspada/bahaya per stasiun dari persentil data."""
    overrides = overrides or {}
    thresholds: dict[str, Thresholds] = {}
    for station in df.columns:
        series = df[station].dropna()
        if series.empty:
            thresholds[station] = Thresholds(alert=0.0, danger=0.0)
            continue
        alert = float(series.quantile(alert_quantile))
        danger = float(series.quantile(danger_quantile))
        station_cfg = overrides.get(station, {})
        if station_cfg.get("alert") is not None:
            alert = float(station_cfg["alert"])
        if station_cfg.get("danger") is not None:
            danger = float(station_cfg["danger"])
        if danger < alert:
            danger = alert
        thresholds[station] = Thresholds(alert=alert, danger=danger)
    return thresholds


def classify_status(
    value: float,
    alert: float,
    danger: float,
    delta_3h: float,
    rising_delta_m: float,
) -> str:
    if value >= danger:
        return "bahaya"
    if value >= alert:
        return "waspada"
    if delta_3h >= rising_delta_m:
        return "meningkat"
    return "normal"


def delta_over_window(series: pd.Series, lookback: int = LOOKBACK_STEPS) -> float:
    """Kenaikan tinggi air selama `lookback` langkah (default 3 jam)."""
    if len(series) < lookback + 1:
        return 0.0
    recent = series.dropna()
    if len(recent) < lookback + 1:
        return 0.0
    return float(recent.iloc[-1] - recent.iloc[-lookback - 1])


def compute_station_status(
    station: str,
    value: float,
    threshold: Thresholds,
    delta_3h: float,
    rising_delta_m: float,
) -> StationStatus:
    status = classify_status(
        value, threshold.alert, threshold.danger, delta_3h, rising_delta_m
    )
    return StationStatus(
        station=station,
        value=value,
        status=status,
        alert=threshold.alert,
        danger=threshold.danger,
        delta_3h=delta_3h,
    )


def risk_score(status: StationStatus) -> float:
    """Skor risiko 0..1+ relatif terhadap jendela alert..danger."""
    span = status.danger - status.alert
    if span <= 0:
        return 1.0 if status.value >= status.danger else 0.0
    return (status.value - status.alert) / span