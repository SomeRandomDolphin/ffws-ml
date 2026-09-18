"use client";

import { useEffect, useRef, useState } from "react";
import { Map, Marker } from "maplibre-gl";
import { attachBasemap, type BasemapState } from "@/lib/basemap";
import type { StationSnapshot } from "@/lib/types";
import Icon from "./icon";
import { configureMapWorker } from "@/lib/map-worker";

export default function LocationMap({ station }: { station: StationSnapshot }) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const retryRef = useRef<(() => void) | null>(null);
  const [state, setState] = useState<BasemapState>("loading");
  useEffect(() => {
    if (!container.current) return;
    let basemap: ReturnType<typeof attachBasemap> | undefined;
    let map: Map;
    try {
      configureMapWorker();
      map = new Map({ container: container.current, center: [station.longitude, station.latitude], zoom: 12,
        scrollZoom: false, style: { version: 8, sources: {}, layers: [{ id: "background", type: "background", paint: { "background-color": "#edf3f6" } }] } });
    } catch { setState("error"); return; }
    mapRef.current = map;
    new Marker({ color: station.color }).setLngLat([station.longitude, station.latitude]).addTo(map);
    map.once("style.load", () => { basemap = attachBasemap(map, setState); retryRef.current = basemap.retry; });
    const observer = new ResizeObserver(() => map.resize());
    observer.observe(container.current);
    return () => { observer.disconnect(); basemap?.dispose(); retryRef.current = null; map.remove(); mapRef.current = null; };
  }, [station.latitude, station.longitude, station.color]);
  return <div className="monitor-map">
    <div className="monitor-map-canvas" ref={container} role="region" aria-label={"Peta lokasi " + station.name} />
    <div className="monitor-map-controls" role="group" aria-label="Kontrol peta">
      <button type="button" aria-label="Perbesar peta" onClick={() => mapRef.current?.zoomIn({duration:0})}><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14" /></svg></button>
      <button type="button" aria-label="Perkecil peta" onClick={() => mapRef.current?.zoomOut({duration:0})}><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M5 12h14" /></svg></button>
      <button type="button" aria-label="Kembali ke posisi stasiun" onClick={() => mapRef.current?.jumpTo({center:[station.longitude,station.latitude],zoom:12})}><Icon name="home" /></button>
    </div>
    <div className="monitor-map-label"><Icon name="pin" />{station.name}</div>
    {state !== "ready" && <div className="monitor-map-state" role="status">{state === "error" ? <>Peta dasar gagal dimuat. <button onClick={() => retryRef.current ? retryRef.current() : window.location.reload()}>Coba lagi</button></> : state === "fallback" ? "Memuat peta alternatif…" : "Memuat peta…"}</div>}
  </div>;
}
