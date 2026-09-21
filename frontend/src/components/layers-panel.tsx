import Icon from "./icon";
import { basemaps, type BasemapId, type BasemapFeedback } from "@/lib/basemap-catalog";
import type { GeoSources } from "@/lib/geodata";
import type { LayerKey, RegionKey } from "@/lib/types";

export type OverlayOpacity = { topology: number; surabayaRivers: number; subdas: number; basin: number; basemap: number };
export type HydroSources = GeoSources;
type Props = { active: LayerKey[]; opacity: OverlayOpacity; sources: HydroSources; region: RegionKey; selected: BasemapId; feedback: BasemapFeedback | null; onSelect: (id: BasemapId) => void; onToggle: (key: LayerKey) => void; onOpacity: (key: keyof OverlayOpacity, value: number) => void; onClear: () => void; onClose: () => void };

type HydroLayer = { key: "topology" | "surabayaRivers"; source: "topology" | "surabayaRivers"; region?: RegionKey; name: string; swatch: "legend-river"; swatchLabel: string; sourceNote: string; emptyNote: string };
const hydroLayers: HydroLayer[] = [
  { key: "topology", source: "topology", name: "Sungai DAS Welang", swatch: "legend-river", swatchLabel: "Aliran sungai dan anak sungai", sourceNote: "OpenStreetMap · di-clip batas DAS BIG", emptyNote: "Belum ada data sungai Welang" },
  { key: "surabayaRivers", source: "surabayaRivers", region: "surabaya", name: "Sungai Surabaya", swatch: "legend-river", swatchLabel: "Sungai dan saluran Kota Surabaya", sourceNote: "OpenStreetMap · di-clip batas Kota Surabaya BIG", emptyNote: "Belum ada data sungai Surabaya" },
];

function HydroLayerRow(props: { layer: HydroLayer; active: LayerKey[]; opacity: OverlayOpacity; sources: HydroSources; region: RegionKey; onToggle: Props["onToggle"]; onOpacity: Props["onOpacity"] }) {
  const { layer } = props;
  const available = props.sources[layer.source];
  const inRegion = !layer.region || layer.region === props.region;
  const enabled = available && inRegion;
  const active = enabled && props.active.includes(layer.key);
  const chip = !available ? "Belum tersedia" : !inRegion ? "Pilih Surabaya" : active ? "Aktif" : "Nonaktif";
  return <details className="hydro-option">
    <summary>{layer.name}<span className={`hydro-chip${!available ? " is-empty" : active ? " is-active" : ""}`}>{chip}</span></summary>
    <div className="hydro-content">
      <p className="hydro-key"><i className={layer.swatch} />{layer.swatchLabel}</p>
      <label className="layer-switch"><span>Tampilkan</span><input type="checkbox" disabled={!enabled} checked={Boolean(active)} onChange={() => props.onToggle(layer.key)} /></label>
      <label className="opacity-control"><span>Opasitas <output>{Math.round(props.opacity[layer.key] * 100)}%</output></span><input aria-label={"Opasitas " + layer.name} type="range" min="0" max="100" value={Math.round(props.opacity[layer.key] * 100)} disabled={!enabled || !active} onChange={event => props.onOpacity(layer.key, Number(event.target.value) / 100)} /></label>
      <small className="hydro-source">{!available ? layer.emptyNote : !inRegion ? "Tersedia pada cakupan Surabaya" : layer.sourceNote}</small>
    </div>
  </details>;
}

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
      {hydroLayers.map(layer => <HydroLayerRow key={layer.key} layer={layer} active={props.active} opacity={props.opacity} sources={props.sources} region={props.region} onToggle={props.onToggle} onOpacity={props.onOpacity} />)}
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

    <button className="reset-layer" onClick={props.onClear}>Matikan overlay</button>
  </aside>;
}
