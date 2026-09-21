"""Palet warna dan gaya visual dashboard DAS Dhompo.

Warna air/status mengikuti prinsip: biru (normal) -> cyan (meningkat) ->
amber (waspada) -> merah (bahaya). Hindari rainbow heatmap.
"""

from __future__ import annotations

STATUS_ORDER: list[str] = ["normal", "meningkat", "waspada", "bahaya"]

STATUS_COLORS: dict[str, str] = {
    "normal": "#3b82f6",    # biru
    "meningkat": "#06b6d4",  # cyan
    "waspada": "#f59e0b",   # amber
    "bahaya": "#dc2626",    # merah
}

STATUS_LABELS: dict[str, str] = {
    "normal": "Normal",
    "meningkat": "Meningkat",
    "waspada": "Waspada",
    "bahaya": "Bahaya",
}

# Badge teks besar untuk kartu detail: label kapital + warna status.
STATUS_BADGES: dict[str, tuple[str, str]] = {
    s: (STATUS_LABELS[s].upper(), STATUS_COLORS[s]) for s in STATUS_ORDER
}

COLOR_PREDICTED = "#0ea5e9"   # garis prediksi / air
COLOR_OBSERVED = "#94a3b8"    # garis observasi
COLOR_SEGMENT = "#cbd5e1"     # segmen default
COLOR_AUX = "#9ca3af"         # node sekunder (tidak dipakai model)
COLOR_TEXT = "#0f172a"
COLOR_MUTED = "#64748b"
COLOR_BG = "#f8fafc"
COLOR_GRID = "#e2e8f0"
COLOR_THRESHOLD_ALERT = "#f59e0b"
COLOR_THRESHOLD_DANGER = "#dc2626"

# Ketebalan segmen sungai per status (skala visual hierarki risiko)
SEGMENT_WIDTH: dict[str, int] = {
    "normal": 5,
    "meningkat": 6,
    "waspada": 7,
    "bahaya": 8,
}


def status_color(status: str) -> str:
    return STATUS_COLORS.get(status, COLOR_SEGMENT)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"