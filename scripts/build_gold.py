"""Bangun dataset gold untuk dashboard Dhompo.

Menggabungkan sumber-sumber tersebar menjadi dua artefak kurasi di
``data/gold/``:

1. ``hydro.csv``     — deret waktu gabungan 2022 (clean) + 2023 (generated)
                       dengan schema kanonik, kolom ``source``, dan
                       ``rain_mm`` (hanya tersedia mulai 2023; NaN di 2022).
2. ``stations.csv``  — metadata per stasiun: koordinat, elevasi, cabang,
                       waktu tempuh ke Dhompo, peran (target/aux/hulu), datum.

Murni restrukturisasi: tidak ada smoothing, resampling baru, atau
feature engineering. Gap ~26 hari (2022-12-05 → 2023-01-01) dibiarkan
terlihat; konsumen wajib memperhatikan kolom ``source`` saat menghitung
lag/rolling. Larangan preprocessing dari data/README.md tetap berlaku —
sumber asli tidak pernah diubah.

Jalankan: ``make gold`` atau ``python scripts/build_gold.py``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dhompo.config import load_yaml_config  # noqa: E402
from dhompo.data.loader import (  # noqa: E402
    STATION_META,
    TARGET_STATION,
    load_data,
    load_combined_data,
)
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
DASHBOARD_YAML = PROJECT_ROOT / "configs" / "dhompo" / "dashboard.yaml"


# ---- metadata stasiun ------------------------------------------------------

def _branches() -> tuple[list[str], list[str], list[str], str]:
    """Cabang hulu + stasiun lokal + stasiun aux, dari dari config YAML.

    Sumber kebenaran topologi disalin eksplisit ke sini (konsisten dengan
    docs/dhompo/ARCHITECTURE.md dan clusters.py) agar script build tidak
    bergantung paket dashboard.
    """
    cfg = load_yaml_config(DASHBOARD_YAML) or {}
    west = cfg.get("topology", {}).get("west_branch") or [
        "Bd. Suwoto", "Krajan Timur", "Purwodadi", "Bd. Baong",
        "Bd. Bakalan", "AWLR Kademungan", "Bd. Domas", "Bd. Grinting",
    ]
    east = cfg.get("topology", {}).get("east_branch") or [
        "Bd. Lecari", "Bd Guyangan", "Sidogiri", "Klosod",
    ]
    local = cfg.get("topology", {}).get("local") or [TARGET_STATION, "Jalan Nasional"]
    aux = cfg.get("topology", {}).get("aux", "Bd. Sentono")
    return west, east, local, aux


def _travel_hours_from_cfg(west: list[str], east: list[str], local: list[str], aux: str) -> dict[str, float]:
    """Waktu tempuh: 3 anchor resmi + regresi elevasi (sama seperti geo.py)."""
    cfg = load_yaml_config(DASHBOARD_YAML) or {}
    cfg_anchor = cfg.get("propagation", {}).get("travel_hours_anchor") or {}
    anchors = dict(cfg_anchor)
    anchors[TARGET_STATION] = 0.0
    anchors["Jalan Nasional"] = 0.0

    out = dict(anchors)
    aux_anchor = anchors.get("Purwodadi")
    if aux_anchor is not None and aux not in out and aux in STATION_META:
        out[aux] = aux_anchor

    # Regresi linear waktu-tempuh terhadap elevasi melewati anchor telemetri
    # resmi dari YAML saja (identik dengan dashboard/geo.py).
    anchor_items = [(s, float(v)) for s, v in cfg_anchor.items() if s in STATION_META]
    if len(anchor_items) >= 2:
        xs = [STATION_META[s][0] for s, _ in anchor_items]
        ys = [y for _, y in anchor_items]
        n = len(anchor_items)
        sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx
        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n
        for st in west + east + local + [aux]:
            if st in out or st not in STATION_META:
                continue
            out[st] = round((a * STATION_META[st][0] + b) * 2) / 2.0
    else:
        for st in west + east + local + [aux]:
            out.setdefault(st, 0.0)
    return out


def build_stations_csv(path: Path) -> None:
    west, east, local, aux = _branches()
    travel = _travel_hours_from_cfg(west, east, local, aux)
    geo_src = PROJECT_ROOT / "configs" / "dhompo" / "station_geo.csv"
    coords: dict[str, dict] = {}
    if geo_src.exists():
        with open(geo_src, encoding="utf-8") as f:
            coords = {r["station"]: r for r in csv.DictReader(f)}

    rows = []
    for st, (elev, order) in STATION_META.items():
        if st == TARGET_STATION:
            role, branch = "target", ""
        elif st in west:
            role, branch = "upstream", "west"
        elif st in east:
            role, branch = "upstream", "east"
        elif st in local:
            role, branch = "local", ""
        elif st == aux:
            role, branch = "aux", ""
        else:
            role, branch = "unknown", ""
        c = coords.get(st, {})
        rows.append({
            "station": st,
            "latitude": c.get("latitude", ""),
            "longitude": c.get("longitude", ""),
            "elevation_m": elev,
            "order": order,
            "branch": branch,
            "travel_hours_to_target": travel.get(st, ""),
            "role": role,
            "coord_confidence": c.get("confidence", ""),
            "coord_source": c.get("source", ""),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[gold] {path.relative_to(PROJECT_ROOT)}: {len(rows)} stasiun")


# ---- hydro series ----------------------------------------------------------

def build_hydro_csv(clean_path: Path, generated_path: Path, path: Path) -> None:
    segments = load_combined_data(clean_path, generated_path)
    import numpy as np
    import pandas as pd

    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    frames = []
    for seg in segments:
        df = seg.df.copy()
        df.insert(0, "source", seg.label)
        if seg.rainfall is None:
            df.insert(1, "rain_mm", np.nan)
        else:
            df.insert(1, "rain_mm", seg.rainfall)
        df["rain_mm"] = df["rain_mm"].astype("float64")
        frames.append(df)

    hydro = pd.concat(frames)
    hydro.index.name = "timestamp"
    hydro = hydro.sort_index()
    # Setelah gabungan: satu baris per timestamp; jika ada tumpang tindih,
    # simpan baris terakhir (2023 menang karena diurutkan terakhir).
    hydro = hydro[~hydro.index.duplicated(keep="last")]

    path.parent.mkdir(parents=True, exist_ok=True)
    hydro.to_csv(path, index_label="timestamp")

    print(f"[gold] {path.relative_to(PROJECT_ROOT)}: {len(hydro)} baris")
    for label, g in hydro.groupby("source"):
        gaps = g.index.to_series().diff().dropna()
        big_gaps = gaps[gaps > pd.Timedelta("30min")]
        print(
            f"        {label}: {len(g)} baris "
            f"({g.index[0]} → {g.index[-1]}), "
            f"gap non-30min: {len(big_gaps)}"
        )
    nan_rain_2022 = int(hydro.loc[hydro["source"] == "2022_clean", "rain_mm"].isna().sum())
    print(f"        rain_mm NaN di 2022_clean: {nan_rain_2022} (diharapkan)")


def main() -> int:
    cfg = load_yaml_config(DASHBOARD_YAML) or {}
    data_cfg = cfg.get("data", {})
    clean = data_cfg.get("gold_source_clean") or "data/data-clean.csv"
    generated = data_cfg.get("gold_source_generated") or "data/Data generated 2023.xlsx"
    clean_path = PROJECT_ROOT / clean
    generated_path = PROJECT_ROOT / generated
    if Path(clean).is_absolute():
        clean_path = Path(clean)
    if Path(generated).is_absolute():
        generated_path = Path(generated)

    for p in (clean_path, generated_path):
        if not p.exists():
            print(f"[gold] sumber tidak ditemukan: {p}", file=sys.stderr)
            return 1

    build_hydro_csv(clean_path, generated_path, GOLD_DIR / "hydro.csv")
    build_stations_csv(GOLD_DIR / "stations.csv")
    print("[gold] selesai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
