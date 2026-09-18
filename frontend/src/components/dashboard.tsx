"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getDemoSnapshot, stationHistory } from "@/lib/demo-data";
import type { LayerKey, RegionKey, StationSnapshot } from "@/lib/types";
import { defaultRegion, regionOrder, regionPresets } from "@/lib/camera";
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


type Panel = "overview" | "layers" | "legend";
const OVERVIEW_DISMISSED_KEY = "welang-overview-dismissed";
export default function Dashboard() {
  const searchParams = useSearchParams();
  const requestedStation = searchParams.get("station");
  const [selected, setSelected] = useState(requestedStation && getDemoSnapshot(0).stations.some((item) => item.name === requestedStation) ? requestedStation : "Dhompo");
  const [region, setRegion] = useState<RegionKey>(requestedStation ? "welang" : defaultRegion);
  const [focusBasin,setFocusBasin] = useState(true);
  const [basemap, setBasemap] = useState<BasemapId>("street");
  const [basemapRequest, setBasemapRequest] = useState({id:"street" as BasemapId, sequence:0});
  const [basemapFeedback, setBasemapFeedback] = useState<BasemapFeedback | null>(null);
  const [opacity, setOpacity] = useState<OverlayOpacity>({topology:.85,subdas:.7,basin:.75,basemap:1});
  const [hydroSources, setHydroSources] = useState<HydroSources>({topology:false,basin:false,subdas:false,admin:false,eastRivers:false});
  const chooseBasemap = (id: BasemapId) => { setBasemap(id); setBasemapRequest(current => ({id,sequence:current.sequence+1})); };
  const reportBasemap = (feedback: BasemapFeedback) => { setBasemapFeedback(feedback); if(feedback.state === "error" && feedback.active) setBasemap(feedback.active); };
  const [activeLayers, setActiveLayers] = useState<LayerKey[]>(["admin", "regionalRivers", "basin", "subdas", "topology", "labels", "rainfall", "quality"]);
  const [activePanel, setActivePanel] = useState<Panel | null>(null);
  const [chartOpen, setChartOpen] = useState(false);
  const shownPanel = usePresence(activePanel);
  const shownChart = usePresence(chartOpen ? "chart" : null);
  const panelTrigger = useRef<HTMLButtonElement | null>(null);
  const chartTrigger = useRef<HTMLButtonElement | null>(null);
  const dismissOverview = () => sessionStorage.setItem(OVERVIEW_DISMISSED_KEY, "true");
  const closePanel = () => {
    if (activePanel === "overview") dismissOverview();
    setActivePanel(null);
    panelTrigger.current?.focus();
  };
  const closeChart = () => { setChartOpen(false); chartTrigger.current?.focus(); };
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [focusRequest, setFocusRequest] = useState<{ name: string; sequence: number } | null>(null);
  const snapshot = useMemo(() => getDemoSnapshot(0), []);
  const station = snapshot.stations.find((item) => item.name === selected) ?? snapshot.stations[13];
  const history = useMemo(() => stationHistory(station), [station]);
  const results = snapshot.stations.filter((item) => item.name.toLowerCase().includes(query.trim().toLowerCase()));
  const timeLabel = snapshot.timestamp;
  const clock = useClock();

  useEffect(() => {
    if (chartOpen && window.matchMedia("(max-width: 700px)").matches) setActivePanel(null);
  }, [chartOpen]);

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
    setSelected(item.name);
    const url = new URL(window.location.href);
    url.searchParams.set("station", item.name);
    window.history.replaceState(null, "", url);
  };
  const searchSelect = (item: StationSnapshot) => {
    selectStation(item); setRegion("welang"); setQuery(""); setSearchOpen(false);
    setFocusRequest((current) => ({ name: item.name, sequence: (current?.sequence ?? 0) + 1 }));
  };
  const togglePanel = (panel: Panel) => {
    if (window.matchMedia("(max-width: 700px)").matches) setChartOpen(false);
    if (activePanel === "overview") dismissOverview();
    setActivePanel((current) => current === panel ? null : panel);
  };
  const changeRegion = (key: RegionKey) => {
    setRegion(key);
    if (window.matchMedia("(max-width: 700px)").matches) setActivePanel(null);
  };
  const toggleLayer = (key: LayerKey) => setActiveLayers((current) => current.includes(key) ? current.filter((item) => item !== key) : [...current, key]);

  return <main className="wm-app wm-dashboard">
    <div className="dashboard-topbar"><header className="wm-header"><WelangBrand /><div className="wm-clock" suppressHydrationWarning>Waktu sekarang · {clock}</div></header>
    <nav className="wm-nav" aria-label="Navigasi peta">
      <span className="nav-context">PEMANTAUAN MUKA AIR</span>
      {([{ key: "overview", label: "Ringkasan", icon: "overview" }, { key: "layers", label: "Lapisan peta", icon: "layers" }, { key: "legend", label: "Legenda", icon: "legend" }] as const).map((panel) => <button key={panel.key} ref={(node) => { if (node && activePanel === panel.key) panelTrigger.current = node; }} className={`wm-nav-btn${activePanel === panel.key ? " is-active" : ""}`} aria-expanded={activePanel === panel.key} onClick={(event) => { panelTrigger.current = event.currentTarget; togglePanel(panel.key); }}><Icon name={panel.icon} /> {panel.label}{panel.key === "layers" && <b>{activeLayers.filter(key => ["labels", "rainfall", "quality"].includes(key) || (key === "basin" && hydroSources.basin) || (key === "topology" && hydroSources.topology) || (key === "subdas" && hydroSources.subdas) || (key === "admin" && hydroSources.admin) || (key === "regionalRivers" && hydroSources.eastRivers)).length}</b>}</button>)}
    </nav></div>
    <div className="monitor-context"><span className="context-timestamp">Waktu data · {snapshot.timestamp}</span></div><section className="wm-workspace">
      <div className="wm-map-area">
        <MapView region={region} focusBasin={focusBasin} basemapRequest={basemapRequest} opacity={opacity} onBasemapFeedback={reportBasemap} onHydroSources={setHydroSources} stations={snapshot.stations} selected={station.name} activeLayers={activeLayers} onSelect={selectStation} onOpenChart={() => setChartOpen(true)} focusRequest={focusRequest} timeLabel={timeLabel} />
        <div className="region-switcher" role="group" aria-label="Cakupan peta">{regionOrder.map(key => <button key={key} aria-pressed={region === key} title={regionPresets[key].description} onClick={() => changeRegion(key)}>{regionPresets[key].label}</button>)}</div>
        <div className="station-search" onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setSearchOpen(false); }}>
          <form role="search" onSubmit={(event) => { event.preventDefault(); if (results[0]) searchSelect(results[0]); }}>
            <Icon name="search" />
            <input aria-label="Cari stasiun" aria-controls="station-results" aria-expanded={searchOpen} placeholder="Cari stasiun…" value={query} onFocus={() => setSearchOpen(true)} onChange={(event) => { setQuery(event.target.value); setSearchOpen(true); }} onKeyDown={(event) => { if (event.key === "Escape") setSearchOpen(false); }} />
            {query && <button type="button" aria-label="Hapus pencarian" onClick={() => setQuery("")}><Icon name="close" /></button>}
          </form>
          {searchOpen && <div id="station-results" className="station-results"><small>{results.length} stasiun ditemukan</small>{results.map((item) => <button key={item.name} onClick={() => searchSelect(item)}><i style={{ background: item.color }} /><span>{item.name}</span><small>{item.status}</small></button>)}{!results.length && <p>Nama tidak ditemukan. Coba nama stasiun lain.</p>}</div>}
        </div>
        {shownPanel === "legend" && <div className={`map-legend-shell${activePanel ? "" : " is-closing"}`} inert={!activePanel} onKeyDown={(event) => { if (event.key === "Escape") closePanel(); }}><LegendPanel onClose={closePanel} activeLayers={activeLayers} sources={hydroSources} /></div>}
        {shownPanel === "overview" && <div className={`map-legend-shell${activePanel ? "" : " is-closing"}`} inert={!activePanel} onKeyDown={(event) => { if (event.key === "Escape") closePanel(); }}><OverviewPanel onClose={closePanel} snapshot={snapshot} /></div>}
        {shownPanel === "layers" && <div className={`map-legend-shell${activePanel ? "" : " is-closing"}`} inert={!activePanel} onKeyDown={(event) => { if (event.key === "Escape") closePanel(); }}><LayersPanel active={activeLayers} opacity={opacity} sources={hydroSources} focusBasin={focusBasin} onFocusBasin={setFocusBasin} selected={basemap} feedback={basemapFeedback} onSelect={chooseBasemap} onToggle={toggleLayer} onOpacity={(key,value) => setOpacity(current => ({...current,[key]:value}))} onClear={() => setActiveLayers([])} onClose={closePanel} /></div>}
      </div>
    </section>
    {shownChart && (<section id="station-chart" className={`wm-chart-panel${chartOpen ? "" : " is-closing"}`} inert={!chartOpen} onKeyDown={(event) => { if (event.key === "Escape") closeChart(); }} aria-label="Grafik stasiun"><div className="chart-heading"><div><h2>Muka air · {station.name}</h2><span>24 jam riwayat simulasi</span></div><button className="panel-close" aria-label="Tutup grafik" onClick={closeChart}><Icon name="close" /></button><div className="chart-legend"><span><i className="chart-key water" />Muka air (m)</span><span><i className="chart-key rain" />Hujan (mm)</span></div></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={history} margin={{ top: 12, right: 35, left: 12, bottom: 0 }}><CartesianGrid stroke="#e8eef2" vertical={false} /><XAxis dataKey="time" tick={{ fontSize: 11, fill: "#526779" }} interval={3} /><YAxis tickFormatter={(value: number) => value.toFixed(1)} yAxisId="water" tick={{ fontSize: 11, fill: "#526779" }} domain={["dataMin - 0.2", "dataMax + 0.3"]} /><YAxis yAxisId="rain" orientation="right" tick={{ fontSize: 11, fill: "#526779" }} /><Tooltip formatter={(value) => Number(value).toFixed(2)} contentStyle={{ borderRadius: 12, border: "1px solid #DCE5EC", boxShadow: "0 8px 24px #102d4e14" }} /><Area isAnimationActive={false} yAxisId="rain" type="monotone" dataKey="rainMm" fill="#dceaf2" stroke="none" name="Hujan" unit=" mm" /><Line isAnimationActive={false} yAxisId="water" type="monotone" dataKey="valueM" stroke="#287eb4" strokeWidth={3} dot={false} name="Muka air" unit=" m" /></ComposedChart></ResponsiveContainer></div></section>)}
    <SiteFooter />
  </main>;
}
