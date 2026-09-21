"""Bangun layer geospasial untuk ITS Water Dashboard (Jawa Timur -> DAS Welang).

Menghasilkan GeoJSON statis di ``frontend/public/geo/``:

- ``jawa_timur_regencies.geojson`` - batas kabupaten/kota (BIG Bapenas).
- ``east_java_rivers.geojson``     - sungai besar Jawa Timur (OSM Overpass).
- ``welang_rivers.geojson``        - jaringan sungai DAS Welang (OSM, di-clip).
- ``surabaya_rivers.geojson``      - jaringan sungai Kota Surabaya (OSM, di-clip).
- ``welang_subdas.geojson``        - sub-DAS hasil delineasi DEMNAS + pour point.

Sumber:
- Administrasi: BIG ``BAPANAS/Batas_Administrasi`` layanan publik.
- Sungai: OpenStreetMap via Overpass API (ODbL). RBI HIDROGRAFI BIG tidak
  menyediakan lembar Jawa Timur pada layanan publik, sehingga OSM dipakai
  sebagai sumber dan dicatat di metadata.
- Sub-DAS: DEMNAS (BIG) via ImageServer ``exportImage`` + WhiteboxTools.
- Batas DAS: aset BIG yang sudah ada (``frontend/public/geo/basin_boundary.geojson``).

Jalankan: ``python scripts/build_geodata.py`` (butuh ekstra ``geodata``).
Unduhan DEM di-cache di ``data/geospatial/demnas/``; layer lain ditulis ulang.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEO_DIR = PROJECT_ROOT / "frontend" / "public" / "geo"
DEM_DIR = PROJECT_ROOT / "data" / "geospatial" / "demnas"
WORK_DIR = DEM_DIR / "_work"
BASIN_PATH = GEO_DIR / "basin_boundary.geojson"
ADMIN_PATH = GEO_DIR / "jawa_timur_regencies.geojson"
STATION_CSV = PROJECT_ROOT / "configs" / "dhompo" / "station_geo.csv"

# Bounding box (min_lon, min_lat, max_lon, max_lat).
WELANG_BBOX = (112.55, -7.99, 112.97, -7.54)
JAWA_BBOX = (110.8, -8.95, 114.8, -5.5)
UTM_CRS = "EPSG:32749"
DEM_PIXEL_DEG = 0.0003  # ~30 m: cukup untuk delineasi DAS, hemat bandwidth.
RIVER_KINDS = "^(river|stream|canal)$"

ADMIN_URL = (
    "https://geoservices.big.go.id/gis/rest/services/BAPANAS/"
    "Batas_Administrasi/MapServer/0/query"
)
DEM_URL = (
    "https://geoservices.big.go.id/raster/rest/services/DEMNAS/"
    "DEM_Indonesia/ImageServer/exportImage"
)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "ITS-Water-Dashboard/0.1 (geodata build; Laboratorium Geospasial ITS)"}


def log(message: str) -> None:
    print(f"[geodata] {message}", flush=True)


def round_coords(value, ndigits: int = 6):
    """Bulatkan koordinat rekursif agar ukuran berkas kecil."""
    if isinstance(value, (list, tuple)):
        return [round_coords(item, ndigits) for item in value]
    if isinstance(value, float):
        return round(value, ndigits)
    return value


def write_geojson(path: Path, data: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["features"] = [
        {**feature, "geometry": {**feature["geometry"], "coordinates": round_coords(feature["geometry"]["coordinates"])}}
        for feature in data.get("features", [])
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_kb = path.stat().st_size / 1024
    log(f"tulis {path.name} ({len(data['features'])} fitur, {size_kb:.0f} KB)")
    return len(data["features"])


def write_metadata(path: Path, **fields) -> None:
    meta = {"retrievedAt": date.today().isoformat(), "crs": "EPSG:4326", **fields}
    (path.parent / f"{path.stem}.metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def request_json(url: str, *, params: dict | None = None, data: dict | None = None, attempts: int = 3):
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(url, data=data, timeout=180, headers=HEADERS) if data is not None else requests.get(
                url, params=params, timeout=180, headers=HEADERS
            )
            response.raise_for_status()
            if "json" in response.headers.get("Content-Type", ""):
                return response.json()
            return json.loads(response.text)
        except Exception as error:  # noqa: BLE001 - dicatat dan dicoba ulang
            last = error
            log(f"percobaan {attempt}/{attempts} gagal: {error}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"gagal memuat {url}: {last}")


def fetch_admin_jawa_timur() -> Path:
    log("ambil batas kabupaten/kota Jawa Timur dari BIG")
    data = request_json(
        ADMIN_URL,
        params={
            "where": "WADMPR='Jawa Timur'",
            "outFields": "WADMKK,KDPKAB,WADMPR",
            "returnGeometry": "true",
            "outSR": "4326",
            "geometryPrecision": "5",
            "f": "geojson",
        },
    )
    out = GEO_DIR / "jawa_timur_regencies.geojson"
    data = simplify_polygons(data, 0.002)
    count = write_geojson(out, data)
    write_metadata(
        out,
        name="Batas Wilayah Administrasi Kabupaten/Kota Jawa Timur",
        publisher="Badan Informasi Geospasial",
        service="Bapenas/Batas_Administrasi layer 0",
        url=ADMIN_URL,
        query="WADMPR='Jawa Timur'",
        features=count,
        license="Data terbuka BIG",
    )
    return out


def overpass_lines(bbox: tuple[float, float, float, float], waterway: str) -> list[dict]:
    min_lon, min_lat, max_lon, max_lat = bbox
    query = (
        f"[out:json][timeout:180];"
        f'way["waterway"~"{waterway}"]({min_lat},{min_lon},{max_lat},{max_lon});'
        f"out geom;"
    )
    data = request_json(OVERPASS_URL, data={"data": query})
    features: list[dict] = []
    for element in data.get("elements", []):
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry") or []
        coords = [[point["lon"], point["lat"]] for point in geometry if "lon" in point and "lat" in point]
        if len(coords) < 2:
            continue
        tags = element.get("tags", {})
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "osm_id": element.get("id"),
                    "waterway": tags.get("waterway", ""),
                    "name": tags.get("name", ""),
                },
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        )
    log(f"Overpass bbox {bbox} -> {len(features)} garis")
    return features


def simplify_lines(features: list[dict], tolerance: float) -> dict:
    import geopandas as gpd

    if not features:
        return {"type": "FeatureCollection", "features": []}
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    return json.loads(gdf.to_json())


def simplify_polygons(data: dict, tolerance: float) -> dict:
    import geopandas as gpd

    if not data.get("features"):
        return data
    gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
    gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    return json.loads(gdf.to_json())


def clip_lines(lines: dict, clip_path: Path, buffer_deg: float = 0.01) -> dict:
    import geopandas as gpd

    if not lines.get("features"):
        return lines
    lines_gdf = gpd.GeoDataFrame.from_features(lines["features"], crs="EPSG:4326")
    clip_gdf = gpd.read_file(clip_path).to_crs("EPSG:4326")
    clip_geom = clip_gdf.geometry.union_all().buffer(buffer_deg)
    clipped = gpd.clip(lines_gdf, clip_geom)
    clipped = clipped[~clipped.geometry.is_empty]
    return json.loads(clipped.to_json())


def surabaya_boundary():
    import geopandas as gpd

    if not ADMIN_PATH.exists():
        raise RuntimeError(f"batas administrasi belum tersedia: {ADMIN_PATH}")
    admin = gpd.read_file(ADMIN_PATH).to_crs("EPSG:4326")
    city = admin[admin["WADMKK"] == "Kota Surabaya"]
    if city.empty:
        raise RuntimeError("batas Kota Surabaya tidak ditemukan pada aset administrasi BIG")
    return city


def fetch_surabaya_rivers() -> None:
    import geopandas as gpd

    city = surabaya_boundary()
    min_lon, min_lat, max_lon, max_lat = (float(value) for value in city.total_bounds)
    bbox = (min_lon, min_lat, max_lon, max_lat)
    rivers = simplify_lines(overpass_lines(bbox, RIVER_KINDS), 0.0004)
    rivers_gdf = gpd.GeoDataFrame.from_features(rivers["features"], crs="EPSG:4326")
    clipped = gpd.clip(rivers_gdf, city.geometry.union_all())
    clipped = clipped[~clipped.geometry.is_empty]

    out = GEO_DIR / "surabaya_rivers.geojson"
    count = write_geojson(out, json.loads(clipped.to_json()))
    write_metadata(
        out,
        name="Jaringan sungai Kota Surabaya",
        publisher="OpenStreetMap contributors",
        service="Overpass API + clip batas administrasi BIG",
        url=OVERPASS_URL,
        query=f'waterway~"{RIVER_KINDS}" bbox={bbox}',
        features=count,
        license="ODbL 1.0",
        note="River, stream, dan canal dipotong ke batas Kota Surabaya dari aset administrasi BIG.",
    )


def fetch_rivers() -> None:
    log("ambil sungai dari OSM Overpass")
    east = overpass_lines(JAWA_BBOX, "^(river)$")
    east_path = GEO_DIR / "east_java_rivers.geojson"
    collected = simplify_lines(east, 0.0008)
    count = write_geojson(east_path, collected)
    write_metadata(
        east_path,
        name="Jaringan sungai besar Jawa Timur",
        publisher="OpenStreetMap contributors",
        service="Overpass API",
        url=OVERPASS_URL,
        query=f'waterway=river bbox={JAWA_BBOX}',
        features=count,
        license="ODbL 1.0",
        note="RBI HIDROGRAFI BIG tidak menyediakan lembar Jawa Timur pada layanan publik; OSM dipakai sebagai sumber.",
    )

    welang = overpass_lines(WELANG_BBOX, RIVER_KINDS)
    welang_clipped = clip_lines(simplify_lines(welang, 0.0004), BASIN_PATH)
    welang_path = GEO_DIR / "welang_rivers.geojson"
    count = write_geojson(welang_path, welang_clipped)
    write_metadata(
        welang_path,
        name="Jaringan sungai DAS Welang",
        publisher="OpenStreetMap contributors",
        service="Overpass API + clip batas DAS BIG 11622",
        url=OVERPASS_URL,
        query=f'waterway~"{RIVER_KINDS}" bbox={WELANG_BBOX}',
        features=count,
        license="ODbL 1.0",
        note="Di-clip ke batas DAS Welang BIG (OBJECTID_1 11622) dengan buffer 0,01 derajat.",
    )


def fetch_demnas(force: bool = False) -> Path:
    raw = DEM_DIR / "welang_demnas_4326.tif"
    if raw.exists() and not force:
        log(f"pakai cache DEM {raw.name}")
        return raw
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    min_lon, min_lat, max_lon, max_lat = WELANG_BBOX
    width = max(1, int((max_lon - min_lon) / DEM_PIXEL_DEG))
    height = max(1, int((max_lat - min_lat) / DEM_PIXEL_DEG))
    log(f"unduh DEMNAS {WELANG_BBOX} pada {width}x{height} piksel")
    response = requests.get(
        DEM_URL,
        params={
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "bboxSR": "4326",
            "imageSR": "4326",
            "size": f"{width},{height}",
            "format": "tiff",
            "pixelType": "F32",
            "noData": "-9999",
            "f": "image",
        },
        timeout=300,
        headers=HEADERS,
    )
    response.raise_for_status()
    if "image" not in response.headers.get("Content-Type", ""):
        raise RuntimeError(f"respons DEMNAS bukan gambar: {response.headers.get('Content-Type')}")
    raw.write_bytes(response.content)
    log(f"tulis {raw.name} ({raw.stat().st_size / 1024 / 1024:.1f} MB)")
    return raw


def prepare_utm_dem(raw: Path) -> Path:
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    out = WORK_DIR / "welang_dem_utm49s.tif"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    with rasterio.open(raw) as src:
        transform, width, height = calculate_default_transform(
            src.crs, UTM_CRS, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=UTM_CRS, transform=transform, width=width, height=height, nodata=-9999.0)
        with rasterio.open(out, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=UTM_CRS,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=-9999.0,
            )
    log(f"reproject DEM -> {UTM_CRS}")
    return out


def load_stations() -> list[dict]:
    import csv

    with STATION_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "name": row["station"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "confidence": row.get("confidence", ""),
        }
        for row in rows
    ]


def upstream_areas(accum_path: Path, points: "object") -> dict[str, float]:
    """Total area hulu per stasiun dari nilai akumulasi di titik snapped."""
    import rasterio
    from rasterio.transform import rowcol

    with rasterio.open(accum_path) as src:
        accum = src.read(1)
        nodata = src.nodata
        transform = src.transform
        cell_km2 = src.res[0] * src.res[1] / 1e6
    areas: dict[str, float] = {}
    for row in points.itertuples():
        row_idx, col_idx = rowcol(transform, row.geometry.x, row.geometry.y)
        if 0 <= row_idx < accum.shape[0] and 0 <= col_idx < accum.shape[1]:
            value = accum[row_idx, col_idx]
            areas[row.name] = 0.0 if value == nodata else float(value) * cell_km2
        else:
            areas[row.name] = 0.0
    return areas


def snap_points_to_mainstem(
    points_path: Path,
    accum_path: Path,
    output_path: Path,
    radius_m: float = 1500.0,
    min_cells: float = 100.0,
) -> None:
    """Geser tiap stasiun ke sel akumulasi terbesar dalam radius (sungai utama).

    Koordinat stasiun berasal dari pengguna dengan keyakinan bervariasi, sehingga
    snap sederhana bisa mendarat di anak sungai kecil. Memilih sel akumulasi
    terbesar membuat titik jatuh ke alur utama. Sel yang sudah dipakai stasiun
    lain dikecualikan agar pour point tetap unik.
    """
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.transform import rowcol, xy
    from shapely.geometry import Point

    points = gpd.read_file(points_path)
    with rasterio.open(accum_path) as src:
        accum = src.read(1)
        nodata = src.nodata
        transform = src.transform
        resolution = src.res[0]
    used: set[tuple[int, int]] = set()
    half = int(radius_m / resolution)
    rows = []
    for row in points.itertuples():
        row_idx, col_idx = rowcol(transform, row.geometry.x, row.geometry.y)
        r0, r1 = max(0, row_idx - half), min(accum.shape[0], row_idx + half + 1)
        c0, c1 = max(0, col_idx - half), min(accum.shape[1], col_idx + half + 1)
        window = accum[r0:r1, c0:c1].astype("float64").copy()
        window[window == nodata] = -1
        for used_row, used_col in used:
            if r0 <= used_row < r1 and c0 <= used_col < c1:
                window[used_row - r0, used_col - c0] = -1
        best_row, best_col = np.unravel_index(np.argmax(window), window.shape)
        if window[best_row, best_col] >= min_cells:
            grid_row, grid_col = r0 + best_row, c0 + best_col
            used.add((grid_row, grid_col))
            x, y = xy(transform, grid_row, grid_col)
        else:
            x, y = row.geometry.x, row.geometry.y
        rows.append({"name": row.name, "confidence": row.confidence, "geometry": Point(x, y)})
    gpd.GeoDataFrame(rows, crs=points.crs).to_file(output_path)


def delineate_subdas():  # noqa: C901 - alur WhiteboxTools berurutan
    import geopandas as gpd
    from shapely.geometry import Point
    from whitebox import WhiteboxTools

    stations = load_stations()
    log(f"delineasi sub-DAS untuk {len(stations)} stasiun")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    points = gpd.GeoDataFrame(
        {
            "name": [station["name"] for station in stations],
            "confidence": [station["confidence"] for station in stations],
        },
        geometry=[Point(station["longitude"], station["latitude"]) for station in stations],
        crs="EPSG:4326",
    ).to_crs(UTM_CRS)
    points_path = WORK_DIR / "stations_utm.shp"
    points.to_file(points_path)

    wbt = WhiteboxTools()
    wbt.work_dir = str(WORK_DIR)
    wbt.verbose = False
    dem = str(prepare_utm_dem(fetch_demnas()))
    filled = str(WORK_DIR / "dem_filled.tif")
    pointer = str(WORK_DIR / "dem_pointer.tif")
    accum = str(WORK_DIR / "dem_accum.tif")
    snapped = str(WORK_DIR / "stations_snapped.shp")
    watersheds = str(WORK_DIR / "welang_watersheds.tif")
    polygons = str(WORK_DIR / "welang_watersheds.shp")

    steps = [
        (
            "flow_accumulation_full_workflow",
            lambda: wbt.flow_accumulation_full_workflow(
                dem, filled, pointer, accum, out_type="cells", esri_pntr=False
            ),
            accum,
        ),
    ]
    for name, call, output in steps:
        log(f"whitebox: {name}")
        call()
        output_path = Path(output)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"WhiteboxTools tidak menghasilkan {output_path.name} pada langkah {name}")

    log("snap pour point ke sungai utama")
    snap_points_to_mainstem(points_path, Path(accum), Path(snapped))

    steps = [
        ("watershed", lambda: wbt.watershed(pointer, snapped, watersheds), watersheds),
        ("raster_to_vector_polygons", lambda: wbt.raster_to_vector_polygons(watersheds, polygons), polygons),
    ]
    for name, call, output in steps:
        log(f"whitebox: {name}")
        call()
        output_path = Path(output)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"WhiteboxTools tidak menghasilkan {output_path.name} pada langkah {name}")

    subdas = gpd.read_file(polygons).to_crs(UTM_CRS)
    if subdas.empty:
        raise RuntimeError("delineasi menghasilkan poligon kosong")
    snapped_utm = gpd.read_file(snapped).to_crs(UTM_CRS)
    upstream = upstream_areas(Path(accum), snapped_utm)

    # Whitebox Watershed dengan banyak titik mempartisi DAS: tiap cell jatuh ke
    # pour point terdekat di hilir, sehingga poligon saling melengkapi (tidak
    # bertingkat). Total area hulu tiap stasiun tetap dihitung dari akumulasi.
    joined = gpd.sjoin(subdas, snapped_utm, how="left", predicate="contains")
    joined = joined.rename(columns={"name": "station", "confidence": "station_confidence"})
    joined = joined[~joined.index.duplicated(keep="first")]
    joined["upstream_area_km2"] = joined["station"].map(upstream).round(2)
    joined["subbasin_id"] = joined.index.astype(str)
    joined = joined.to_crs("EPSG:4326")

    basin = gpd.read_file(BASIN_PATH).to_crs("EPSG:4326")
    basin_geom = basin.geometry.union_all()
    basin_area = basin.to_crs(UTM_CRS).area.sum() / 1e6
    partition_area = subdas.area.sum() / 1e6
    joined["inside_big_basin"] = joined.geometry.intersects(basin_geom)
    joined["geometry"] = joined.geometry.make_valid().simplify(0.0004, preserve_topology=True)
    joined = joined[
        ["station", "station_confidence", "subbasin_id", "upstream_area_km2", "inside_big_basin", "geometry"]
    ].sort_values("station").reset_index(drop=True)

    out = GEO_DIR / "welang_subdas.geojson"
    joined.to_file(out, driver="GeoJSON")
    outside = sorted({row.station for row in joined.itertuples() if row.station and not row.inside_big_basin})
    write_metadata(
        out,
        name="Sub-DAS stasiun DAS Welang",
        publisher="Delineasi mandiri dari DEMNAS (BIG) + WhiteboxTools",
        service="DEMNAS DEM_Indonesia ImageServer exportImage",
        url=DEM_URL,
        demPixelDeg=DEM_PIXEL_DEG,
        crsProcessing=UTM_CRS,
        features=len(joined),
        partitionAreaKm2=round(float(partition_area), 1),
        basinAreaBigKm2=round(float(basin_area), 1),
        note="Sub-DAS adalah area aliran lokal yang mempartisi DAS (tidak bertingkat): "
        "tiap sel ditugaskan ke stasiun terdekat di hilir. 'upstream_area_km2' memuat "
        "total area tangkapan hulu tiap stasiun. Akurasi bergantung pada kualitas "
        "koordinat (kolom station_confidence) dan DEMNAS ~30 m. "
        f"Stasiun di luar batas DAS BIG: {outside or 'tidak ada'}.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["admin", "rivers", "surabaya-rivers", "subdas"],
        action="append",
        help="jalankan hanya langkah tertentu (boleh berulang)",
    )
    parser.add_argument("--force-dem", action="store_true", help="unduh ulang DEM walau cache ada")
    args = parser.parse_args()
    selected = set(args.only) if args.only else {"admin", "rivers", "surabaya-rivers", "subdas"}
    GEO_DIR.mkdir(parents=True, exist_ok=True)

    if "admin" in selected:
        fetch_admin_jawa_timur()
    if "rivers" in selected:
        fetch_rivers()
    if "surabaya-rivers" in selected:
        fetch_surabaya_rivers()
    if "subdas" in selected:
        if args.force_dem:
            fetch_demnas(force=True)
        delineate_subdas()
    log("selesai")
    return 0


if __name__ == "__main__":
    sys.exit(main())
