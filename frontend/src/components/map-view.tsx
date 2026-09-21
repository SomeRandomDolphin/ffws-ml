"use client";

import { useEffect, useRef, useState } from "react";
import { LngLatBounds, Map as LibreMap, Marker, NavigationControl, Popup, ScaleControl } from "maplibre-gl";
import type { LayerKey, RegionKey, StationSnapshot } from "@/lib/types";
import { boundsOf, loadGeoAssets, type GeoAssets } from "@/lib/geodata";
import { regionPresets } from "@/lib/camera";
import type { BasemapState } from "@/lib/basemap";
import { createBasemapController, type BasemapId, type BasemapFeedback } from "@/lib/basemap-catalog";
import type { OverlayOpacity, HydroSources } from "./layers-panel";
import { emptyGeoJSON, insideBasin } from "@/lib/basin-focus";
import type { FeatureCollection } from "geojson";
import type { GeoJSONSource } from "maplibre-gl";
import Icon from "./icon";
import { configureMapWorker } from "@/lib/map-worker";
import { qualityColors, connectionColors } from "@/lib/presentation";
import { liveStateLabel, liveTime, type LiveStation } from "@/lib/surabaya";

type Props = {
  liveStations: LiveStation[]; liveDisconnected: boolean;
  onLiveSelect: (station: LiveStation) => void;
  region: RegionKey;
  basemapRequest: {id: BasemapId; sequence: number}; opacity: OverlayOpacity;
  onBasemapFeedback: (feedback: BasemapFeedback) => void; onHydroSources: (sources: HydroSources) => void;
  stations: StationSnapshot[]; selected: string; activeLayers: LayerKey[];
  onSelect: (station: StationSnapshot) => void; onOpenChart: (trigger?: HTMLElement) => void;
  focusRequest: { name: string; sequence: number } | null; timeLabel: string;
};

export default function MapView(props: Props) {
  const basemapController = useRef<ReturnType<typeof createBasemapController> | null>(null);
  const basinData = useRef<FeatureCollection>(emptyGeoJSON);
  const geoData = useRef<GeoAssets>({ regencies: emptyGeoJSON, eastRivers: emptyGeoJSON, welangRivers: emptyGeoJSON, surabayaRivers: emptyGeoJSON, subdas: emptyGeoJSON, basin: emptyGeoJSON });
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<LibreMap | null>(null);
  const markers = useRef(new Map<string, Marker>());
  const liveMarkers = useRef(new Map<string, Marker>());
  const latest = useRef(props);
  latest.current = props;
  const popup = useRef<Popup | null>(null);
  const popupStation = useRef<{ kind: "welang" | "surabaya"; name: string; menu: boolean } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [failed, setFailed] = useState(false);
  const [basemapState, setBasemapState] = useState<BasemapState>("loading");
  const retryBasemap = useRef<() => void>(() => {});
  const [ready, setReady] = useState(false);
  const fitAll = useRef<(duration?: number) => void>(() => {});
  const showPopup = useRef<(name: string, menu: boolean) => void>(() => {});
  const showLivePopup = useRef<(id: string, menu: boolean) => void>(() => {});
  const arrangeLabels = useRef<() => void>(() => {});
  const frameAll = useRef(true);

  useEffect(() => {
    if (!container.current) return;
    let disposed = false;
    setFailed(false); setBasemapState("loading"); setReady(false);
    let basemap: ReturnType<typeof createBasemapController> | undefined;
    let map: LibreMap;
    try {
      configureMapWorker();
      map = new LibreMap({
        container: container.current, renderWorldCopies:false, center: [112.7, -7.75], zoom: 8,
        style: { version: 8, sources: {}, layers: [{ id: "background", type: "background", paint: { "background-color": "#e4edf0" } }] },
      });
    } catch { setFailed(true); return; }
    mapRef.current = map;
    map.addControl(new NavigationControl({ showCompass: false }), "top-left");
    map.addControl(new ScaleControl({ unit: "metric" }), "bottom-left");
    map.on("movestart", (event) => { if (event.originalEvent) frameAll.current = false; });
    fitAll.current = (duration = 0) => {
      frameAll.current = true;
      if (!container.current) return;
      const preset = regionPresets[latest.current.region];
      const coordinates = latest.current.region === "surabaya" ? preset.fallbackBounds : boundsOf(geoData.current[preset.asset]) ?? preset.fallbackBounds;
      const bounds = new LngLatBounds([coordinates[0], coordinates[1]], [coordinates[2], coordinates[3]]);
      const { width, height } = container.current.getBoundingClientRect();
      map.fitBounds(bounds, { padding: { top: Math.min(85, height * .22), bottom: Math.min(60, height * .16), left: Math.min(75, width * .18), right: Math.min(145, width * .25) }, maxZoom: preset.maxZoom, duration });
    };
    showPopup.current = (name, menu) => {
      const station = latest.current.stations.find((item) => item.name === name);
      if (!station) return;
      popup.current?.remove();
      popupStation.current = { kind: "welang", name, menu };
      const content = document.createElement("div");
      content.className = `station-popover${station.status === "Bahaya" ? " is-danger" : ""}`;
      const addText = (tag: string, text: string, className = "") => {
        const node = document.createElement(tag);
        node.textContent = text; node.className = className; content.append(node);
      };
      if(basinData.current.features.length && !insideBasin([station.longitude,station.latitude],basinData.current)) addText("small","Di luar batas DAS sumber BIG","popup-time");
      addText("strong", station.name, "popup-station-name");
      addText("small", "Stasiun pemantauan · DAS Welang", "popup-kicker");
      addText("span", station.valueM.toFixed(2) + " m", "popup-reading");
      addText("small", `Waktu data · ${latest.current.timeLabel}`, "popup-time");
      addText("small", `${station.delta3hM >= 0 ? "Naik" : "Turun"} ${Math.abs(station.delta3hM).toFixed(2)} m dalam 3 jam`, "popup-trend");
      if (station.status === "Bahaya") {
        const alert = document.createElement("div");
        alert.className = `popup-alert popup-alert-${station.status.toLowerCase()}`;
        alert.textContent = "Notifikasi bahaya: perlu perhatian segera.";
        content.append(alert);
      }
      if (menu) {
        const actions = document.createElement("div"); actions.className = "popup-actions";
        const previewBtn = document.createElement("button");
        previewBtn.type = "button";
        previewBtn.className = "popup-preview-btn";
        previewBtn.innerHTML = '<svg class="popup-action-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect width="18" height="18" x="3" y="3" rx="2" /><path d="M15 3v18" /><path d="m8 9 3 3-3 3" /></svg><span>Preview</span>';
        previewBtn.onclick = () => {
          latest.current.onSelect(station);
          latest.current.onOpenChart(markers.current.get(station.name)?.getElement());
        };
        const detail = document.createElement("a");
        detail.href = "/station/" + encodeURIComponent(name);
        detail.innerHTML = '<svg class="popup-action-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 1 1 16 0Z"/><circle cx="12" cy="10" r="2.5"/></svg><span>Detail stasiun</span>';
        const alertAction = document.createElement("button");
        alertAction.className = "popup-alert-action";
        alertAction.innerHTML = '<svg class="popup-action-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9ZM10 21h4"/></svg><span>Simulasikan notifikasi</span>';
        alertAction.setAttribute("aria-label", `Simulasikan notifikasi untuk ${station.name}`);
        alertAction.onclick = () => { alertAction.textContent = "Simulasi selesai, tidak ada notifikasi dikirim"; alertAction.disabled = true; };
        actions.append(previewBtn, detail, alertAction); content.append(actions);
      }
      const next = new Popup({ closeButton: true, closeOnClick: true, focusAfterOpen: false, offset: 18, maxWidth: "340px", className: "station-popup" })
        .setLngLat([station.longitude, station.latitude]).setDOMContent(content).addTo(map);
      next.on("close", () => { if (popup.current === next) popupStation.current = null; });
      popup.current = next;
    };
    showLivePopup.current = (id, menu) => {
      const station = latest.current.liveStations.find(item => item.id === id);
      if (!station || station.latitude === null || station.longitude === null) return;
      popup.current?.remove();
      popupStation.current = { kind: "surabaya", name: id, menu };
      const content = document.createElement("div");
      content.className = "station-popover";
      const addText = (tag: string, text: string, className = "") => {
        const node = document.createElement(tag); node.textContent = text; node.className = className; content.append(node);
      };
      addText("strong", station.name, "popup-station-name");
      const primary = station.sensors.find(sensor => sensor.valueCm !== null);
      addText("span", primary?.valueCm == null ? "—" : `${primary.valueCm.toFixed(1)} cm`, "popup-reading");
      addText("small", `Waktu data · ${liveTime(station.observedAt)}`, "popup-time");
      if (menu) {
        const actions = document.createElement("div"); actions.className = "popup-actions";
        const stationUrl = "/station/" + encodeURIComponent(station.name);
        const previewBtn = document.createElement("button");
        previewBtn.type = "button";
        previewBtn.className = "popup-preview-btn";
        previewBtn.innerHTML = '<svg class="popup-action-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect width="18" height="18" x="3" y="3" rx="2" /><path d="M15 3v18" /><path d="m8 9 3 3-3 3" /></svg><span>Preview detail</span>';
        previewBtn.onclick = () => {
          latest.current.onLiveSelect(station);
        };
        const detail = document.createElement("a"); detail.href = stationUrl;
        detail.innerHTML = '<svg class="popup-action-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 1 1 16 0Z"/><circle cx="12" cy="10" r="2.5"/></svg><span>Detail stasiun</span>';
        const alertAction = document.createElement("button");
        alertAction.className = "popup-alert-action";
        alertAction.innerHTML = '<svg class="popup-action-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9ZM10 21h4"/></svg><span>Simulasikan notifikasi</span>';
        alertAction.setAttribute("aria-label", `Simulasikan notifikasi untuk ${station.name}`);
        alertAction.onclick = () => { alertAction.textContent = "Simulasi selesai, tidak ada notifikasi dikirim"; alertAction.disabled = true; };
        actions.append(previewBtn, detail, alertAction); content.append(actions);
      }
      const next = new Popup({ closeButton: true, closeOnClick: true, focusAfterOpen: false, offset: 18, maxWidth: "340px", className: "station-popup" })
        .setLngLat([station.longitude, station.latitude]).setDOMContent(content).addTo(map);
      next.on("close", () => { if (popup.current === next) popupStation.current = null; });
      popup.current = next;
    };
    // DOM markers are independent of tile, glyph, and optional GeoJSON loading.
    for (const station of latest.current.stations) {
      const button = document.createElement("button");
      button.type = "button"; button.className = "station-pin"; button.dataset.station = station.name;
      const rain = document.createElement("span"); rain.className = "station-rainfall";
      const quality = document.createElement("span"); quality.className = "station-quality";
      const dot = document.createElement("span"); dot.className = "station-pin-dot";
      const label = document.createElement("span"); label.className = "station-pin-label"; label.textContent = station.name;
      button.append(rain, quality, dot, label);
      button.onmouseenter = button.onfocus = () => { if (!popupStation.current?.menu) showPopup.current(station.name, false); };
      button.onmouseleave = button.onblur = () => { if (!popupStation.current?.menu) popup.current?.remove(); };
      button.onclick = (event) => {
        event.stopPropagation();
        const current = latest.current.stations.find((item) => item.name === station.name);
        if (current) latest.current.onSelect(current);
        showPopup.current(station.name, true);
      };
      button.onkeydown = (event) => { if (event.key === "Escape") popup.current?.remove(); };
      markers.current.set(station.name, new Marker({ element: button }).setLngLat([station.longitude, station.latitude]).addTo(map));
    }
    arrangeLabels.current = () => {
      if (!container.current) return;
      const bounds = container.current.getBoundingClientRect();
      const entries = [...markers.current.entries()].sort(([a], [b]) => Number(b === latest.current.selected) - Number(a === latest.current.selected));
      const dots = entries.map(([, marker]) => marker.getElement().querySelector(".station-pin-dot")!.getBoundingClientRect());
      const occupied: Array<{ left: number; right: number; top: number; bottom: number }> = [...dots];
      const search = container.current.parentElement?.querySelector(".station-search");
      if (search) occupied.push(search.getBoundingClientRect());
      const overlaps = (a: typeof occupied[number], b: typeof occupied[number]) => a.left < b.right + 4 && a.right + 4 > b.left && a.top < b.bottom + 3 && a.bottom + 3 > b.top;
      for (const [, marker] of entries) {
        const button = marker.getElement();
        const label = button.querySelector<HTMLElement>(".station-pin-label")!;
        label.style.visibility = "hidden";
        if (!latest.current.activeLayers.includes("labels")) continue;
        for (const position of ["right", "left", "top", "bottom"]) {
          label.dataset.position = position;
          const rect = label.getBoundingClientRect();
          if (rect.left < bounds.left + 8 || rect.right > bounds.right - 8 || rect.top < bounds.top + 8 || rect.bottom > bounds.bottom - 30 || occupied.some((other) => overlaps(rect, other))) continue;
          label.style.visibility = "visible";
          occupied.push(rect);
          break;
        }
      }
    };
    map.on("move", () => arrangeLabels.current());
    void document.fonts.ready.then(() => { if (!disposed) arrangeLabels.current(); });
    setReady(true); fitAll.current();
    const resize = new ResizeObserver(() => { map.resize(); if (frameAll.current) fitAll.current(); arrangeLabels.current(); });
    resize.observe(container.current);
    map.once("style.load", () => {
      if (disposed) return;
      basemap = createBasemapController(map, feedback => { setBasemapState(feedback.state); latest.current.onBasemapFeedback(feedback); });
      basemapController.current = basemap;
      basemap.setOpacity(latest.current.opacity.basemap);
      basemap.select(latest.current.basemapRequest.id);
      retryBasemap.current = basemap.retry;
      const visible = (key: LayerKey) => latest.current.activeLayers.includes(key) ? "visible" as const : "none" as const;
      map.addSource("regencies", { type: "geojson", data: emptyGeoJSON });
      map.addLayer({ id: "regencies-fill", type: "fill", source: "regencies", layout: { visibility: visible("admin") }, paint: { "fill-color": "#9aabb5", "fill-opacity": .035 } });
      map.addLayer({ id: "regencies-outline", type: "line", source: "regencies", layout: { visibility: visible("admin") }, paint: { "line-color": "#6f7f89", "line-width": ["interpolate", ["linear"], ["zoom"], 6, .45, 10, 1.1], "line-opacity": .72 } });
      map.addSource("east-rivers", { type: "geojson", data: emptyGeoJSON });
      map.addLayer({ id: "east-river-lines", type: "line", source: "east-rivers", maxzoom: 10, layout: { visibility: visible("regionalRivers") }, paint: { "line-color": "#68a6cb", "line-width": ["interpolate", ["linear"], ["zoom"], 6, .55, 10, 1.4], "line-opacity": .68 } });
      map.addSource("basin", { type: "geojson", data: emptyGeoJSON });
      map.addLayer({ id: "basin-fill", type: "fill", source: "basin", layout: { visibility: visible("basin") }, paint: { "fill-color": "#167cb5", "fill-opacity": .12 * latest.current.opacity.basin } });
      map.addLayer({ id: "basin-outline", type: "line", source: "basin", layout: { visibility: visible("basin") }, paint: { "line-color": "#167cb5", "line-width": ["interpolate", ["linear"], ["zoom"], 6, 1.5, 11, 3], "line-opacity": latest.current.opacity.basin } });
      map.addSource("subdas", { type: "geojson", data: emptyGeoJSON });
      map.addLayer({ id: "subdas-fill", type: "fill", source: "subdas", minzoom: 8.8, layout: { visibility: visible("subdas") }, paint: { "fill-color": ["match", ["get", "station_confidence"], "Tinggi", "#4a9c7d", "Sedang-Tinggi", "#73aa82", "Sedang", "#9eb77b", "#c5b46a"], "fill-opacity": .15 * latest.current.opacity.subdas } });
      map.addLayer({ id: "subdas-outline", type: "line", source: "subdas", minzoom: 8.8, layout: { visibility: visible("subdas") }, paint: { "line-color": "#538e76", "line-width": 1.1, "line-opacity": latest.current.opacity.subdas } });
      map.addSource("rivers", { type: "geojson", data: emptyGeoJSON });
      map.addLayer({ id: "river-lines", type: "line", source: "rivers", minzoom: 8.5, layout: { visibility: visible("topology"), "line-cap": "round", "line-join": "round" }, paint: { "line-color": ["match", ["get", "waterway"], "river", "#247eb5", "canal", "#6ea8c5", "#4a98c5"], "line-width": ["match", ["get", "waterway"], "river", 2.8, "canal", 1.6, 1.15], "line-opacity": latest.current.opacity.topology } });
      map.addSource("surabaya-rivers", { type: "geojson", data: emptyGeoJSON });
      map.addLayer({ id: "surabaya-river-lines", type: "line", source: "surabaya-rivers", minzoom: 8.5, layout: { visibility: latest.current.activeLayers.includes("surabayaRivers") && latest.current.region === "surabaya" ? "visible" : "none", "line-cap": "round", "line-join": "round" }, paint: { "line-color": ["match", ["get", "waterway"], "river", "#247eb5", "canal", "#6ea8c5", "#4a98c5"], "line-width": ["match", ["get", "waterway"], "river", 2.8, "canal", 1.6, 1.15], "line-opacity": latest.current.opacity.surabayaRivers } });
    });
    loadGeoAssets().then(data => {
      if(disposed) return;
      const apply = () => {
        if(disposed) return;
        (map.getSource("regencies") as GeoJSONSource | undefined)?.setData(data.assets.regencies);
        (map.getSource("east-rivers") as GeoJSONSource | undefined)?.setData(data.assets.eastRivers);
        (map.getSource("rivers") as GeoJSONSource | undefined)?.setData(data.assets.welangRivers);
        (map.getSource("surabaya-rivers") as GeoJSONSource | undefined)?.setData(data.assets.surabayaRivers);
        (map.getSource("subdas") as GeoJSONSource | undefined)?.setData(data.assets.subdas);
        (map.getSource("basin") as GeoJSONSource | undefined)?.setData(data.assets.basin);
        geoData.current = data.assets;
        basinData.current = data.assets.basin;
        latest.current.onHydroSources(data.sources);
        if(frameAll.current) fitAll.current();
        if(popupStation.current?.kind === "welang") showPopup.current(popupStation.current.name,popupStation.current.menu);
      };
      if(map.getSource("rivers")) apply(); else map.once("style.load",apply);
    });
    const currentMarkers = markers.current;
    return () => {
      disposed = true; basemap?.dispose(); basemapController.current = null; retryBasemap.current = () => {}; resize.disconnect(); popup.current?.remove();
      currentMarkers.forEach((marker) => marker.remove()); currentMarkers.clear();
      liveMarkers.current.forEach(marker => marker.remove()); liveMarkers.current.clear();
      arrangeLabels.current = () => {};
      map.remove(); mapRef.current = null;
    };
  }, [attempt]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const ids = new Set(props.liveStations.map(station => station.id));
    liveMarkers.current.forEach((marker, id) => { if (!ids.has(id)) { marker.remove(); liveMarkers.current.delete(id); } });
    for (const station of props.liveStations) {
      if (station.latitude === null || station.longitude === null) continue;
      let marker = liveMarkers.current.get(station.id);
      if (!marker) {
        const button = document.createElement("button");
        button.type = "button"; button.className = "station-pin live-pin";
        const dot = document.createElement("span"); dot.className = "station-pin-dot";
        const label = document.createElement("span"); label.className = "station-pin-label"; label.textContent = station.name; label.dataset.position = "right";
        button.append(dot, label);
        button.onmouseenter = button.onfocus = () => { if (!popupStation.current?.menu) showLivePopup.current(station.id, false); };
        button.onmouseleave = button.onblur = () => { if (!popupStation.current?.menu) popup.current?.remove(); };
        button.onclick = (event) => {
          event.stopPropagation();
          const current = latest.current.liveStations.find(item => item.id === station.id);
          if (current) showLivePopup.current(current.id, true);
        };
        marker = new Marker({ element: button }).setLngLat([station.longitude, station.latitude]).addTo(map);
        liveMarkers.current.set(station.id, marker);
      }
      marker.setLngLat([station.longitude, station.latitude]);
      const button = marker.getElement();
      button.style.display = props.region === "surabaya" ? "grid" : "none";
      button.style.setProperty("--station-color", station.state === "live" && !props.liveDisconnected ? connectionColors.live : connectionColors.delayed);
      button.classList.toggle("is-selected", station.name === props.selected);
      button.classList.toggle("has-label", props.activeLayers.includes("labels"));
      button.setAttribute("aria-label", `${station.name}, ${props.liveDisconnected ? "koneksi terputus" : liveStateLabel[station.state]}, lokasi perkiraan, pengamatan ${liveTime(station.observedAt)}`);
      button.setAttribute("aria-pressed", String(station.name === props.selected));
      button.title = `${station.name} · posisi sensor belum terverifikasi`;
    }
  }, [props.liveStations, props.liveDisconnected, props.region, props.selected, props.activeLayers, ready]);

  useEffect(() => {
    if (!ready) return;
    props.stations.forEach((station) => {
      const marker = markers.current.get(station.name);
      if (!marker) return;
      marker.setLngLat([station.longitude, station.latitude]);
      const button = marker.getElement();
      button.style.display = props.region === "welang" ? "grid" : "none";
      button.style.setProperty("--station-color", station.color);
      button.style.setProperty("--quality-color", qualityColors[station.quality.status]);
      button.style.setProperty("--rain-size", `${Math.min(52, 18 + station.rainfall30mMm * 2.4)}px`);
      button.classList.toggle("is-selected", station.name === props.selected);
      button.classList.toggle("has-label", props.activeLayers.includes("labels"));
      button.classList.toggle("has-rainfall", props.activeLayers.includes("rainfall"));
      button.classList.toggle("has-quality", props.activeLayers.includes("quality"));
      button.setAttribute("aria-label", `${station.name}, muka air ${station.status} ${station.valueM.toFixed(2)} meter, kualitas air ${station.quality.status}, hujan ${station.rainfall30mMm.toFixed(1)} milimeter`);
      button.setAttribute("aria-pressed", String(station.name === props.selected));
    });
    if (popupStation.current?.kind === "welang") showPopup.current(popupStation.current.name, popupStation.current.menu);
    if (popupStation.current?.kind === "surabaya") showLivePopup.current(popupStation.current.name, popupStation.current.menu);
    arrangeLabels.current();
  }, [props.stations, props.selected, props.activeLayers, props.timeLabel, props.region, ready]);
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    for (const [id, key] of [["regencies-fill", "admin"], ["regencies-outline", "admin"], ["east-river-lines", "regionalRivers"], ["basin-fill", "basin"], ["basin-outline", "basin"], ["subdas-fill", "subdas"], ["subdas-outline", "subdas"], ["river-lines", "topology"], ["surabaya-river-lines", "surabayaRivers"]] as const) {
      const inRegion = key !== "surabayaRivers" || props.region === "surabaya";
      if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", props.activeLayers.includes(key) && inRegion ? "visible" : "none");
    }
  }, [props.activeLayers, props.region, ready]);
  useEffect(() => { basemapController.current?.select(props.basemapRequest.id); }, [props.basemapRequest]);
  useEffect(() => {
    const map = mapRef.current;
    basemapController.current?.setOpacity(props.opacity.basemap);
    if(map?.getLayer("river-lines")) map.setPaintProperty("river-lines","line-opacity",props.opacity.topology);
    if(map?.getLayer("surabaya-river-lines")) map.setPaintProperty("surabaya-river-lines","line-opacity",props.opacity.surabayaRivers);
    if(map?.getLayer("subdas-outline")) map.setPaintProperty("subdas-outline","line-opacity",props.opacity.subdas);
    if(map?.getLayer("subdas-fill")) map.setPaintProperty("subdas-fill","fill-opacity",.15 * props.opacity.subdas);
    if(map?.getLayer("basin-outline")) map.setPaintProperty("basin-outline","line-opacity",props.opacity.basin);
    if(map?.getLayer("basin-fill")) map.setPaintProperty("basin-fill","fill-opacity",.12 * props.opacity.basin);
  }, [props.opacity,ready]);
  useEffect(() => {
    if (!ready) return;
    popup.current?.remove();
    fitAll.current(window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 500);
  }, [props.region, ready]);
  useEffect(() => {
    if (!ready || !props.focusRequest) return;
    const liveStation = latest.current.liveStations.find(item => item.name === props.focusRequest?.name);
    if (liveStation && liveStation.latitude !== null && liveStation.longitude !== null) {
      frameAll.current = false;
      mapRef.current?.easeTo({ center: [liveStation.longitude, liveStation.latitude], zoom: 15, duration: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 350 });
      return;
    }
    const station = latest.current.stations.find((item) => item.name === props.focusRequest?.name);
    if (station) {
      frameAll.current = false;
      mapRef.current?.easeTo({ center: [station.longitude, station.latitude], zoom: 13, duration: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 350 });
      showPopup.current(station.name, true);
    }
  }, [props.focusRequest, ready]);

  return <>
    <div className="map-canvas" ref={container} role="region" aria-label={props.region === "surabaya" ? "Peta stasiun Surabaya" : props.region === "welang" ? "Peta stasiun DAS Welang" : "Peta sumber daya air Jawa Timur"} />
    <button className="map-fit" aria-label={`Kembalikan cakupan ${regionPresets[props.region].label}`} title={`Kembalikan cakupan ${regionPresets[props.region].label}`} onClick={() => { popup.current?.remove(); fitAll.current(350); }}><Icon name="home" /></button>
    {basemapState !== "ready" && !failed && <div className="map-message" role="status">
      {basemapState === "loading" ? "Memuat peta dasar…" : basemapState === "fallback" ? "Sumber utama tidak tersedia. Memuat peta cadangan…" : "Peta dasar belum dapat diakses. Periksa koneksi internet; titik stasiun tetap tersedia."}
      {basemapState === "error" && <button onClick={() => retryBasemap.current()}>Coba muat peta lagi</button>}
    </div>}
    {failed && <div className="map-failure" role="alert"><strong>Peta belum dapat ditampilkan</strong><p>Periksa dukungan grafis browser, lalu coba lagi.</p><button onClick={() => setAttempt((value) => value + 1)}>Coba lagi</button></div>}
  </>;
}
