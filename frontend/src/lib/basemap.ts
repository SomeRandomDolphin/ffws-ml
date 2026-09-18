import type { ErrorEvent, Map, MapSourceDataEvent } from "maplibre-gl";

export type BasemapState = "loading" | "fallback" | "ready" | "error";
const providers = [
  { tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], maxzoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' },
  { tiles: ["https://tile.opentopomap.org/{z}/{x}/{y}.png"], maxzoom: 17, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, SRTM | <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)' },
];

/** Install after style.load, not load: unavailable tiles must not block setup. */
export function attachBasemap(map: Map, onState: (state: BasemapState) => void) {
  let disposed = false;
  let provider = 0;
  let generation = 0;
  let sourceId = "";
  let timer: ReturnType<typeof setTimeout> | undefined;
  let switching = false;
  let failed = false;
  let hasLoadedTile = false;

  const fail = () => {
    if (disposed || switching) return;
    clearTimeout(timer);
    failed = true;
    if (provider === providers.length - 1) { onState("error"); return; }
    switching = true;
    // Defer source removal until MapLibre finishes dispatching the tile event.
    queueMicrotask(() => { if (!disposed && switching) install(provider + 1); });
  };
  const install = (index: number) => {
    clearTimeout(timer);
    provider = index;
    switching = false;
    failed = false;
    hasLoadedTile = false;
    const previous = sourceId;
    sourceId = `basemap-${++generation}`;
    if (previous) {
      if (map.getLayer(previous)) map.removeLayer(previous);
      if (map.getSource(previous)) map.removeSource(previous);
    }
    onState(index === 0 ? "loading" : "fallback");
    // A fresh source avoids stale failed tile caches on retry.
    map.addSource(sourceId, { type: "raster", tileSize: 256, ...providers[index] });
    const before = map.getStyle().layers.find((layer) => layer.type !== "background")?.id;
    map.addLayer({ id: sourceId, type: "raster", source: sourceId }, before);
    timer = setTimeout(fail, 10000);
  };
  const onError = (event: ErrorEvent) => {
    if ((event as ErrorEvent & { sourceId?: string }).sourceId === sourceId) fail();
  };
  const onData = (event: MapSourceDataEvent) => {
    // Metadata/idle events also occur after failures; only decoded tiles prove recovery.
    if (event.sourceId !== sourceId || event.tile?.state !== "loaded" || disposed || switching) return;
    hasLoadedTile = true;
    onIdle();
  };
  const onIdle = () => {
    if (disposed || switching || failed || !hasLoadedTile || !map.isSourceLoaded(sourceId)) return;
    clearTimeout(timer);
    onState("ready");
  };
  map.on("error", onError);
  map.on("sourcedata", onData);
  map.on("idle", onIdle);
  install(0);
  return {
    retry: () => { if (!disposed) install(0); },
    dispose: () => {
      disposed = true;
      clearTimeout(timer);
      map.off("error", onError);
      map.off("sourcedata", onData);
      map.off("idle", onIdle);
    },
  };
}
