import type { Map, ErrorEvent, MapSourceDataEvent } from "maplibre-gl";

export const basemaps = [
  { id: "street", name: "Jalan", source: "OpenStreetMap", url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", maxzoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' },
  { id: "rbi", name: "Rupabumi Indonesia", source: "BIG", url: "https://geoservices.big.go.id/rbi/rest/services/BASEMAP/Rupabumi_Indonesia/MapServer/tile/{z}/{y}/{x}", maxzoom: 16, attribution: '© <a href="https://geoservices.big.go.id/">Badan Informasi Geospasial</a>' },
  { id: "imagery", name: "Citra satelit", source: "Esri World Imagery", url: "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", maxzoom: 19, attribution: 'Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community' },
] as const;
export type BasemapId = typeof basemaps[number]["id"];
export type BasemapFeedback = { state: "loading" | "ready" | "error"; active: BasemapId | null; requested: BasemapId };

/** Stage a raster below overlays and only replace the last good map after tiles load. */
export function createBasemapController(map: Map, report: (event: BasemapFeedback) => void) {
  let active: { id: string; choice: BasemapId } | null = null;
  let pending: { id: string; choice: BasemapId; loaded: boolean } | null = null;
  let generation = 0;
  let opacity = 1;
  let disposed = false;
  let requested: BasemapId = "street";
  let timer: ReturnType<typeof setTimeout> | undefined;
  const remove = (id: string) => { if (map.getLayer(id)) map.removeLayer(id); if (map.getSource(id)) map.removeSource(id); };
  const cancel = () => { clearTimeout(timer); if (pending) remove(pending.id); pending = null; };
  const fail = () => {
    const candidate = pending;
    if (!candidate || disposed) return;
    clearTimeout(timer);
    pending = null;
    queueMicrotask(() => {
      if (disposed) return;
      remove(candidate.id);
      report({ state: "error", active: active?.choice ?? null, requested: candidate.choice });
    });
  };
  const commit = () => {
    if (!pending?.loaded || !map.isSourceLoaded(pending.id) || disposed) return;
    clearTimeout(timer);
    const candidate = pending;
    pending = null;
    if (active) remove(active.id);
    active = { id: candidate.id, choice: candidate.choice };
    map.setPaintProperty(active.id, "raster-opacity", opacity);
    report({ state: "ready", active: active.choice, requested: active.choice });
  };
  const onData = (event: MapSourceDataEvent) => {
    if (pending && event.sourceId === pending.id && event.tile?.state === "loaded") { pending.loaded = true; commit(); }
  };
  const onError = (event: ErrorEvent) => { if (pending && (event as ErrorEvent & { sourceId?: string }).sourceId === pending.id) fail(); };
  const select = (choice: BasemapId) => {
    if (disposed) return;
    requested = choice;
    cancel();
    if (active?.choice === choice) { report({state:"ready",active:choice,requested:choice}); return; }
    const config = basemaps.find(item => item.id === choice)!;
    const id = "selected-basemap-" + ++generation;
    pending = { id, choice, loaded:false };
    report({state:"loading",active:active?.choice ?? null,requested:choice});
    map.addSource(id, {type:"raster",tiles:[config.url],tileSize:256,maxzoom:config.maxzoom,attribution:config.attribution});
    const before = map.getStyle().layers.find(layer => layer.type !== "background")?.id;
    map.addLayer({id,type:"raster",source:id,paint:{"raster-opacity":active ? 0 : opacity,"raster-fade-duration":0}},before);
    timer = setTimeout(fail, 10000);
  };
  map.on("sourcedata",onData); map.on("idle",commit); map.on("error",onError);
  return {
    select,
    setOpacity(value: number) { opacity = value; if (active) map.setPaintProperty(active.id,"raster-opacity",value); if (pending && !active) map.setPaintProperty(pending.id,"raster-opacity",value); },
    retry() { select(requested); },
    dispose() { disposed = true; clearTimeout(timer); map.off("sourcedata",onData); map.off("idle",commit); map.off("error",onError); },
  };
}
