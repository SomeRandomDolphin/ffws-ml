"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getDemoSnapshot, stationHistory } from "@/lib/demo-data";
import { chartTheme, connectionColors } from "@/lib/presentation";
import type { LayerKey, RegionKey, StationSnapshot } from "@/lib/types";
import { defaultRegion, regionPresets } from "@/lib/camera";
import Icon from "./icon";
import { WelangBrand, SiteFooter } from "./brand";
import { usePresence } from "./use-presence";
import { useClock } from "./use-clock";
import MapView from "./map-view";
import LegendPanel from "./legend-panel";
import OverviewPanel from "./overview-panel";
import LayersPanel, { type OverlayOpacity, type HydroSources } from "./layers-panel";
import { type BasemapId, type BasemapFeedback } from "@/lib/basemap-catalog";
import "./layers-panel.css";
import { useSurabaya } from "./use-surabaya";
import LiveStationPanel from "./live-station-panel";
import { surabayaNames, surabayaStationFallback, liveStateLabel, liveTime, type LiveStation } from "@/lib/surabaya";
import WelangStationPanel from "./welang-station-panel";
import RegionFilter from "./region-filter";


type Panel = "overview" | "layers" | "legend";
const OVERVIEW_DISMISSED_KEY = "welang-overview-dismissed";
export default function Dashboard() {
  const searchParams = useSearchParams();
  const requestedStation = searchParams.get("station");
  const [selected, setSelected] = useState(requestedStation || "Dhompo");
  const { data: live, error: liveError } = useSurabaya();
  const liveStations = live?.stations.length ? live.stations : surabayaStationFallback;
  const [liveOpen, setLiveOpen] = useState(!!requestedStation && surabayaNames.some(name => name === requestedStation));
  const liveStation = liveStations.find(item => item.name === selected);
  const [region, setRegion] = useState<RegionKey>(surabayaNames.some(name => name === requestedStation) ? "surabaya" : requestedStation ? "welang" : defaultRegion);
  const [basemap, setBasemap] = useState<BasemapId>("street");
  const [basemapRequest, setBasemapRequest] = useState({id:"street" as BasemapId, sequence:0});
  const [basemapFeedback, setBasemapFeedback] = useState<BasemapFeedback | null>(null);
  const [opacity, setOpacity] = useState<OverlayOpacity>({topology:.85,surabayaRivers:.85,subdas:.7,basin:.75,basemap:1});
  const [hydroSources, setHydroSources] = useState<HydroSources>({topology:false,surabayaRivers:false,basin:false,subdas:false,admin:false,eastRivers:false});
  const chooseBasemap = (id: BasemapId) => { setBasemap(id); setBasemapRequest(current => ({id,sequence:current.sequence+1})); };
  const reportBasemap = (feedback: BasemapFeedback) => { setBasemapFeedback(feedback); if(feedback.state === "error" && feedback.active) setBasemap(feedback.active); };
  const [activeLayers, setActiveLayers] = useState<LayerKey[]>([
    "admin", "regionalRivers", "labels", "rainfall", "quality",
    ...(region === "welang" ? ["topology" as const] : []),
    ...(region === "surabaya" ? ["surabayaRivers" as const] : []),
  ]);
  const [activePanel, setActivePanel] = useState<Panel | null>(null);
  const [chartOpen, setChartOpen] = useState(false);
  const [welangDrawerOpen, setWelangDrawerOpen] = useState(false);
  const shownPanel = usePresence(activePanel);
  const shownChart = usePresence(chartOpen ? "chart" : null);
  const panelTrigger = useRef<HTMLButtonElement | null>(null);
  const chartTrigger = useRef<HTMLElement | null>(null);
  const chartClose = useRef<HTMLButtonElement | null>(null);
  const liveTrigger = useRef<HTMLElement | null>(null);
  const liveClose = useRef<HTMLButtonElement | null>(null);
  const closeLive = () => { setLiveOpen(false); if (liveTrigger.current?.isConnected) liveTrigger.current.focus(); else document.querySelector<HTMLInputElement>('[aria-label="Cari stasiun"]')?.focus(); };
  const dismissOverview = () => sessionStorage.setItem(OVERVIEW_DISMISSED_KEY, "true");
  const closePanel = () => {
    if (activePanel === "overview") dismissOverview();
    setActivePanel(null);
    panelTrigger.current?.focus();
  };
  const closeChart = () => { setChartOpen(false); chartTrigger.current?.focus(); };
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchPointerFocus, setSearchPointerFocus] = useState(false);
  const [focusRequest, setFocusRequest] = useState<{ name: string; sequence: number } | null>(null);
  const snapshot = useMemo(() => getDemoSnapshot(0), []);
  const station = snapshot.stations.find((item) => item.name === selected) ?? snapshot.stations[13];
  const history = useMemo(() => stationHistory(station), [station]);
  const normalizedQuery = query.trim().toLowerCase();
  const results = snapshot.stations.filter((item) => item.name.toLowerCase().includes(normalizedQuery));
  const liveResults = liveStations.filter(item => item.name.toLowerCase().includes(normalizedQuery));
  const scopedResults = region === "surabaya" ? [] : results;
  const scopedLiveResults = region === "welang" ? [] : liveResults;
  const selectLive = (item: LiveStation) => {
    liveTrigger.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setSelected(item.name); setRegion("surabaya"); setLiveOpen(true); setChartOpen(false); setActivePanel(null);
    setQuery(""); setSearchOpen(false);
    setFocusRequest(current => ({ name: item.name, sequence: (current?.sequence ?? 0) + 1 }));
    const url = new URL(window.location.href); url.searchParams.set("station", item.name); window.history.replaceState(null, "", url);
  };
  const timeLabel = snapshot.timestamp;
  const clock = useClock();

  useEffect(() => {
    if (chartOpen && window.matchMedia("(max-width: 700px)").matches) setActivePanel(null);
    if (chartOpen) chartClose.current?.focus();
  }, [chartOpen]);

  useEffect(() => { if (liveOpen) liveClose.current?.focus(); }, [liveOpen]);
  useEffect(() => { if (activePanel) document.querySelector<HTMLButtonElement>('.map-legend-shell:not([inert]) .overlay-close')?.focus(); }, [activePanel]);
  useEffect(() => {
    setActiveLayers(current => {
      const layers: LayerKey[] = current.filter(key => key !== "topology" && key !== "surabayaRivers");
      if (region === "welang") layers.push("topology");
      if (region === "surabaya") layers.push("surabayaRivers");
      return layers;
    });
  }, [region]);

  useEffect(() => {
    if (requestedStation && snapshot.stations.some((item) => item.name === requestedStation)) setSelected(requestedStation);
  }, [requestedStation, snapshot.stations]);
  useEffect(() => {
    if (requestedStation) {
      setActivePanel(null);
      return;
    }
    if (sessionStorage.getItem(OVERVIEW_DISMISSED_KEY) !== "true") setActivePanel("overview");
  }, [requestedStation]);
  const selectStation = (item: StationSnapshot) => {
    setSelected(item.name); setLiveOpen(false); setWelangDrawerOpen(false);
    const url = new URL(window.location.href);
    url.searchParams.set("station", item.name);
    window.history.replaceState(null, "", url);
  };
  const searchSelect = (item: StationSnapshot) => {
    selectStation(item); setRegion("welang"); setQuery(""); setSearchOpen(false);
    setFocusRequest((current) => ({ name: item.name, sequence: (current?.sequence ?? 0) + 1 }));
  };
  const togglePanel = (panel: Panel) => {
    if (window.matchMedia("(max-width: 700px)").matches) { setChartOpen(false); setWelangDrawerOpen(false); }
    if (activePanel === "overview") dismissOverview();
    setActivePanel((current) => current === panel ? null : panel);
  };
  const changeRegion = (key: RegionKey) => {
    setRegion(key); setLiveOpen(false); setWelangDrawerOpen(false);
    if (window.matchMedia("(max-width: 700px)").matches) setActivePanel(null);
  };
  const toggleLayer = (key: LayerKey) => setActiveLayers((current) => current.includes(key) ? current.filter((item) => item !== key) : [...current, key]);

  return <main className="wm-app wm-dashboard">
    <div className="dashboard-topbar"><header className="wm-header"><WelangBrand /><div className="wm-clock" suppressHydrationWarning>{clock}</div></header>
    <nav className="wm-nav" aria-label="Navigasi peta">
      <span className="nav-context">Pemantauan muka air</span>
      {([{ key: "overview", label: "Ringkasan", icon: "overview" }, { key: "layers", label: "Lapisan peta", icon: "layers" }, { key: "legend", label: "Legenda", icon: "legend" }] as const).map((panel) => <button key={panel.key} ref={(node) => { if (node && activePanel === panel.key) panelTrigger.current = node; }} className={`wm-nav-btn${activePanel === panel.key ? " is-active" : ""}`} aria-expanded={activePanel === panel.key} onClick={(event) => { panelTrigger.current = event.currentTarget; togglePanel(panel.key); }}><Icon name={panel.icon} /> {panel.label}{panel.key === "layers" && <b>{activeLayers.filter(key => ["labels", "rainfall", "quality"].includes(key) || (key === "basin" && hydroSources.basin) || (key === "topology" && hydroSources.topology) || (key === "surabayaRivers" && hydroSources.surabayaRivers) || (key === "subdas" && hydroSources.subdas) || (key === "admin" && hydroSources.admin) || (key === "regionalRivers" && hydroSources.eastRivers)).length}</b>}</button>)}
    </nav></div>
    <section className="wm-workspace">
      <div className="wm-map-area">
        <MapView liveStations={liveStations} liveDisconnected={!!liveError} onLiveSelect={selectLive} region={region} basemapRequest={basemapRequest} opacity={opacity} onBasemapFeedback={reportBasemap} onHydroSources={setHydroSources} stations={snapshot.stations} selected={selected} activeLayers={activeLayers} onSelect={selectStation} onOpenChart={(trigger) => { chartTrigger.current = trigger ?? null; setWelangDrawerOpen(true); }} focusRequest={focusRequest} timeLabel={timeLabel} />
         <div className={`station-search${searchPointerFocus ? " is-pointer-focused" : ""}`} onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) { setSearchOpen(false); setSearchPointerFocus(false); } }}>
             <form role="search" onSubmit={(event) => { event.preventDefault(); if (scopedLiveResults[0] && (region === "surabaya" || !scopedResults[0])) selectLive(scopedLiveResults[0]); else if (scopedResults[0]) searchSelect(scopedResults[0]); }}>
             <Icon name="search" />
             <input aria-label="Cari stasiun" aria-controls="station-results" aria-expanded={searchOpen} placeholder="Cari stasiun…" value={query} onPointerDown={() => setSearchPointerFocus(true)} onFocus={() => setSearchOpen(true)} onBlur={() => setSearchPointerFocus(false)} onChange={(event) => { setQuery(event.target.value); setSearchOpen(true); }} onKeyDown={(event) => { setSearchPointerFocus(false); if (event.key === "Escape") setSearchOpen(false); }} />
             {query && <button type="button" aria-label="Hapus pencarian" onClick={() => setQuery("")}><Icon name="close" /></button>}
             <RegionFilter value={region} onChange={changeRegion} onOpen={() => {
               setSearchOpen(false);
               setActivePanel(null);
               setLiveOpen(false);
               setWelangDrawerOpen(false);
               setChartOpen(false);
             }} />
           </form>
           {searchOpen && <div id="station-results" className="station-results"><small>{scopedResults.length + scopedLiveResults.length} stasiun ditemukan</small>{scopedLiveResults.map(item => <button key={item.id} onClick={() => selectLive(item)}><i style={{ background: item.state === "live" && !liveError ? connectionColors.live : connectionColors.delayed }} /><span>{item.name}</span><small>{liveError ? "Koneksi terputus" : liveStateLabel[item.state]}</small></button>)}{scopedResults.map((item) => <button key={item.name} onClick={() => searchSelect(item)}><i style={{ background: item.color }} /><span>{item.name}</span><small>{item.status}</small></button>)}{!scopedResults.length && !scopedLiveResults.length && <p>Nama tidak ditemukan. Coba nama stasiun lain.</p>}</div>}
        </div>
        {region === "surabaya" && !liveOpen && <div className="surabaya-stations"><details><summary>Stasiun Surabaya <span>{liveStations.length}</span></summary><div className="surabaya-station-list">{liveError && <p role="status">{liveError}</p>}{!live && !liveError && <p>Memuat pembacaan…</p>}{liveStations.map(item => <button key={item.id} onClick={() => selectLive(item)}><span>{item.name}</span><small>{liveError ? "Koneksi terputus" : liveStateLabel[item.state]} · {liveTime(item.observedAt)}</small></button>)}</div></details></div>}
        {region === "welang" && !welangDrawerOpen && <div className="welang-stations"><details><summary>Stasiun DAS Welang <span>{snapshot.stations.length}</span></summary><div className="welang-station-list">{snapshot.stations.map(item => <button key={item.name} onClick={() => selectStation(item)}><span>{item.name}</span><small>{item.status} · {item.valueM.toFixed(2)} m</small></button>)}</div></details></div>}
        {liveOpen && (
          <aside className="live-drawer" aria-label="Detail sensor Surabaya" onKeyDown={(event) => { if (event.key === "Escape") closeLive(); }}>
            <div className="live-drawer-topbar">
              <strong>{liveStation?.name ?? "Detail sensor"}</strong>
              <button ref={liveClose} className="live-close" aria-label="Tutup detail" onClick={closeLive}><span aria-hidden="true">×</span></button>
            </div>
            {liveStation ? <LiveStationPanel station={liveStation} disconnected={!!liveError} /> : <p role="status" className="live-loading">{liveError ?? "Memuat data stasiun…"}</p>}
          </aside>
        )}
        {welangDrawerOpen && station && (
          <aside className="live-drawer" aria-label={`Detail stasiun ${station.name}`} onKeyDown={(event) => { if (event.key === "Escape") setWelangDrawerOpen(false); }}>
            <div className="live-drawer-topbar">
              <strong>{station.name}</strong>
              <button className="live-close" aria-label="Tutup detail" onClick={() => setWelangDrawerOpen(false)}><span aria-hidden="true">×</span></button>
            </div>
            <WelangStationPanel station={station} showHeading={false} />
          </aside>
        )}
        {shownPanel === "legend" && <div className={`map-legend-shell${activePanel ? "" : " is-closing"}`} inert={!activePanel} onKeyDown={(event) => { if (event.key === "Escape") closePanel(); }}><LegendPanel onClose={closePanel} activeLayers={activeLayers} sources={hydroSources} /></div>}
        {shownPanel === "overview" && <div className={`map-legend-shell${activePanel ? "" : " is-closing"}`} inert={!activePanel} onKeyDown={(event) => { if (event.key === "Escape") closePanel(); }}><OverviewPanel onClose={closePanel} /></div>}
        {shownPanel === "layers" && <div className={`map-legend-shell${activePanel ? "" : " is-closing"}`} inert={!activePanel} onKeyDown={(event) => { if (event.key === "Escape") closePanel(); }}><LayersPanel active={activeLayers} opacity={opacity} sources={hydroSources} region={region} selected={basemap} feedback={basemapFeedback} onSelect={chooseBasemap} onToggle={toggleLayer} onOpacity={(key,value) => setOpacity(current => ({...current,[key]:value}))} onClear={() => setActiveLayers(current => current.filter(key => key === "topology" || key === "surabayaRivers"))} onClose={closePanel} /></div>}
      </div>
    </section>
    {shownChart && (<section id="station-chart" className={`wm-chart-panel${chartOpen ? "" : " is-closing"}`} inert={!chartOpen} onKeyDown={(event) => { if (event.key === "Escape") closeChart(); }} aria-label="Grafik stasiun"><div className="chart-heading"><div><h2>Muka air · {station.name}</h2><span>24 jam riwayat simulasi</span></div><button ref={chartClose} className="panel-close" aria-label="Tutup grafik" onClick={closeChart}><Icon name="close" /></button><div className="chart-legend"><span><i className="chart-key water" />Muka air (m)</span><span><i className="chart-key rain" />Hujan (mm)</span></div></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={history} margin={{ top: 12, right: 8, left: 0, bottom: 0 }}><CartesianGrid stroke={chartTheme.grid} vertical={false} /><XAxis dataKey="time" tick={chartTheme.tick} minTickGap={28} /><YAxis width={45} tickFormatter={(value: number) => value.toFixed(1)} yAxisId="water" tick={chartTheme.tick} domain={["dataMin - 0.2", "dataMax + 0.3"]} /><YAxis width={35} yAxisId="rain" orientation="right" tick={chartTheme.tick} /><Tooltip formatter={(value) => Number(value).toFixed(2)} contentStyle={chartTheme.tooltip} /><Area isAnimationActive={false} yAxisId="rain" type="monotone" dataKey="rainMm" fill={chartTheme.rainFill} stroke={chartTheme.rain} name="Hujan" unit=" mm" /><Line isAnimationActive={false} yAxisId="water" type="monotone" dataKey="valueM" stroke={chartTheme.water} strokeWidth={2.5} dot={false} name="Muka air" unit=" m" /></ComposedChart></ResponsiveContainer></div></section>)}
    <SiteFooter />
  </main>;
}
