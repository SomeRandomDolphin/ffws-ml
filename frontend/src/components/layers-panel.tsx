import Icon from "./icon";
import WaterwayKey from "./waterway-key";
import { basemaps, type BasemapId, type BasemapFeedback } from "@/lib/basemap-catalog";
import type { GeoSources } from "@/lib/geodata";
import type { LayerKey, RegionKey } from "@/lib/types";

export type OverlayOpacity = { topology: number; surabayaRivers: number; subdas: number; basin: number; basemap: number };
export type HydroSources = GeoSources;
type Props = { active: LayerKey[]; opacity: OverlayOpacity; sources: HydroSources; region: RegionKey; selected: BasemapId; feedback: BasemapFeedback | null; onSelect: (id: BasemapId) => void; onToggle: (key: LayerKey) => void; onOpacity: (key: keyof OverlayOpacity, value: number) => void; onClear: () => void; onClose: () => void };


export default function LayersPanel(props: Props) {
  return <aside className="tools-card layer-browser map-legend-overlay" aria-label="Lapisan peta">
    <header className="panel-heading map-legend-heading"><button className="overlay-close" aria-label="Tutup lapisan" onClick={props.onClose}><Icon name="close" /></button><h2>Lapisan peta <Icon name="layers" /></h2></header>

    <section>
      <h3><Icon name="pin" />Stasiun</h3>
      <label className="layer-switch"><span>Nama stasiun</span><input type="checkbox" checked={props.active.includes("labels")} onChange={() => props.onToggle("labels")} /></label>
      <label className="layer-switch"><span>Intensitas hujan</span><input type="checkbox" checked={props.active.includes("rainfall")} onChange={() => props.onToggle("rainfall")} /></label>
      <label className="layer-switch"><span>Kualitas air</span><input type="checkbox" checked={props.active.includes("quality")} onChange={() => props.onToggle("quality")} /></label>
    </section>

    <section>
      <h3><Icon name="wave" />Hidrologi</h3>
      <WaterwayKey />
      <label className="opacity-control"><span>Opasitas <output>{Math.round(props.opacity.topology * 100)}%</output></span><input aria-label="Opasitas sungai" type="range" min="0" max="100" value={Math.round(props.opacity.topology * 100)} onChange={event => {
        const value = Number(event.target.value) / 100;
        props.onOpacity("topology", value);
        props.onOpacity("surabayaRivers", value);
      }} /></label>
    </section>

    <section>
      <h3><Icon name="overview" />Konteks Jawa Timur</h3>
      <label className="layer-switch"><span>Batas kabupaten/kota</span><input type="checkbox" disabled={!props.sources.admin} checked={props.sources.admin && props.active.includes("admin")} onChange={() => props.onToggle("admin")} /></label>
      <label className="layer-switch"><span>Sungai besar Jawa Timur</span><input type="checkbox" disabled={!props.sources.eastRivers} checked={props.sources.eastRivers && props.active.includes("regionalRivers")} onChange={() => props.onToggle("regionalRivers")} /></label>
    </section>

    <section>
      <h3><Icon name="layers" />Peta dasar</h3>
      <div className="basemap-options" role="group" aria-label="Pilih peta dasar">
        {basemaps.map(item => <button key={item.id} aria-pressed={props.selected === item.id} onClick={() => props.onSelect(item.id)}>
          <span className={"basemap-preview preview-" + item.id} aria-hidden="true"><i /><b>{props.selected === item.id ? "✓" : ""}</b></span>
          <strong>{item.name}</strong>
          <small>{item.source}</small>
        </button>)}
      </div>
      <label className="opacity-control"><span>Opasitas <output>{Math.round(props.opacity.basemap * 100)}%</output></span><input aria-label="Opasitas peta dasar" type="range" min="0" max="100" value={Math.round(props.opacity.basemap * 100)} onChange={event => props.onOpacity("basemap", Number(event.target.value) / 100)} /></label>
      {props.feedback?.state === "loading" && <p role="status">Memuat peta pilihan…</p>}
      {props.feedback?.state === "error" && <p role="status">Sumber belum tersedia. {props.feedback.active ? "Peta sebelumnya dipertahankan." : "Titik stasiun tetap tersedia."} <button className="basemap-retry" onClick={() => props.onSelect(props.feedback!.requested)}>Coba lagi</button></p>}
    </section>

    <button className="reset-layer" onClick={props.onClear}>Matikan overlay lainnya</button>
  </aside>;
}
