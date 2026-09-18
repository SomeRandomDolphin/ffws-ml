import { emptyGeoJSON } from "./basin-focus";
import type { FeatureCollection } from "geojson";

export type GeoAssets = {
  regencies: FeatureCollection;
  eastRivers: FeatureCollection;
  welangRivers: FeatureCollection;
  subdas: FeatureCollection;
  basin: FeatureCollection;
};

export type GeoSources = {
  admin: boolean;
  eastRivers: boolean;
  topology: boolean;
  subdas: boolean;
  basin: boolean;
};

type AssetSpec = { path: string; kinds: string[]; verifyBasin?: boolean };

const specs: Record<keyof GeoAssets, AssetSpec> = {
  regencies: { path: "/geo/jawa_timur_regencies.geojson", kinds: ["Polygon", "MultiPolygon"] },
  eastRivers: { path: "/geo/east_java_rivers.geojson", kinds: ["LineString", "MultiLineString"] },
  welangRivers: { path: "/geo/welang_rivers.geojson", kinds: ["LineString", "MultiLineString"] },
  subdas: { path: "/geo/welang_subdas.geojson", kinds: ["Polygon", "MultiPolygon"] },
  basin: { path: "/geo/basin_boundary.geojson", kinds: ["Polygon", "MultiPolygon"], verifyBasin: true },
};

const coordinatesValid = (value: unknown): boolean => {
  if (!Array.isArray(value) || !value.length) return false;
  if (typeof value[0] === "number") {
    return (
      value.length >= 2 &&
      Number.isFinite(value[0]) &&
      Number.isFinite(value[1]) &&
      Math.abs(value[0] as number) <= 180 &&
      Math.abs(value[1] as number) <= 90
    );
  }
  return value.every(coordinatesValid);
};

async function loadAsset(spec: AssetSpec): Promise<{ data: FeatureCollection; local: boolean }> {
  try {
    const response = await fetch(spec.path);
    if (!response.ok) throw new Error("Missing GeoJSON");
    const data = (await response.json()) as FeatureCollection & { features?: Array<{ geometry?: { type: string; coordinates: unknown } }> };
    if (data.type !== "FeatureCollection" || !Array.isArray(data.features) || !data.features.length) throw new Error("Invalid collection");
    const ok = data.features.every(
      (feature) =>
        feature?.geometry &&
        spec.kinds.includes(feature.geometry.type) &&
        "coordinates" in feature.geometry &&
        coordinatesValid(feature.geometry.coordinates),
    );
    if (!ok) throw new Error("Invalid geometry");
    if (spec.verifyBasin) {
      const verified = data.features.every((feature) => {
        const properties = (feature as { properties?: Record<string, unknown> }).properties ?? {};
        return properties.OBJECTID_1 === 11622 && properties.Nama_DAS === "WELANG";
      });
      if (!verified) throw new Error("Unverified basin");
    }
    return { data, local: true };
  } catch {
    return { data: emptyGeoJSON, local: false };
  }
}

export async function loadGeoAssets(): Promise<{ assets: GeoAssets; sources: GeoSources }> {
  const [regencies, eastRivers, welangRivers, subdas, basin] = await Promise.all([
    loadAsset(specs.regencies),
    loadAsset(specs.eastRivers),
    loadAsset(specs.welangRivers),
    loadAsset(specs.subdas),
    loadAsset(specs.basin),
  ]);
  return {
    assets: {
      regencies: regencies.data,
      eastRivers: eastRivers.data,
      welangRivers: welangRivers.data,
      subdas: subdas.data,
      basin: basin.data,
    },
    sources: {
      admin: regencies.local,
      eastRivers: eastRivers.local,
      topology: welangRivers.local,
      subdas: subdas.local,
      basin: basin.local,
    },
  };
}

export function boundsOf(collection: FeatureCollection): [number, number, number, number] | null {
  let minLon = Infinity;
  let minLat = Infinity;
  let maxLon = -Infinity;
  let maxLat = -Infinity;
  const visit = (coordinates: unknown): void => {
    if (!Array.isArray(coordinates)) return;
    if (typeof coordinates[0] === "number" && typeof coordinates[1] === "number") {
      minLon = Math.min(minLon, coordinates[0]);
      minLat = Math.min(minLat, coordinates[1]);
      maxLon = Math.max(maxLon, coordinates[0]);
      maxLat = Math.max(maxLat, coordinates[1]);
      return;
    }
    coordinates.forEach(visit);
  };
  collection.features.forEach((feature) => {
    const geometry = feature.geometry as { coordinates?: unknown } | null;
    if (geometry && "coordinates" in geometry) visit(geometry.coordinates);
  });
  return Number.isFinite(minLon) ? [minLon, minLat, maxLon, maxLat] : null;
}
