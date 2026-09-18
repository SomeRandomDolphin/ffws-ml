import Icon from "./icon";
import { basemaps, type BasemapId, type BasemapFeedback } from "@/lib/basemap-catalog";
import type { GeoSources } from "@/lib/geodata";
import type { LayerKey } from "@/lib/types";

export type OverlayOpacity = { topology: number; subdas: number; basin: number; basemap: number };
export type HydroSources = GeoSources;
type Props = { focusBasin: boolean; onFocusBasin: (value:boolean)=>void; active: LayerKey[]; opacity: OverlayOpacity; sources: HydroSources; selected: BasemapId; feedback: BasemapFeedback | null; onSelect: (id: BasemapId) => void; onToggle: (key: LayerKey) => void; onOpacity: (key: keyof OverlayOpacity, value: number) => void; onClear: () => void; onClose: () => void };

type HydroLayer = { key: "topology" | "subdas" | "basin"; name: string; swatch: "legend-river" | "legend-subbasin" | "legend-basin"; swatchLabel: string; sourceNote: string; emptyNote: string };
const hydroLayers: HydroLayer[] = [
  { key: "topology", name: "Sungai DAS Welang", swatch: "legend-river", swatchLabel: "Aliran sungai dan anak sungai", sourceNote: "OpenStreetMap · di-clip batas DAS BIG", emptyNote: "Belum ada data sungai lokal" },
  { key: "subdas", name: "Sub-DAS stasiun", swatch: "legend-subbasin", swatchLabel: "Area aliran lokal per stasiun", sourceNote: "Delineasi DEMNAS ~30 m · WhiteboxTools", emptyNote: "Sub-DAS belum tersedia" },
  { key: "basin", name: "Batas DAS", swatch: "legend-basin", swatchLabel: "Area dan batas DAS", sourceNote: "BIG · Atlas Wilayah Sungai · WELANG (11622)", emptyNote: "Batas DAS valid belum tersedia" },
];

function HydroLayerRow(props: { layer: HydroLayer; active: LayerKey[]; opacity: OverlayOpacity; sources: HydroSources; onToggle: Props["onToggle"]; onOpacity: Props["onOpacity"] }) {
  const { layer } = props;
  const available = props.sources[layer.key];
  const active = available && props.active.includes(layer.key);
  const chip = !available ? "Belum tersedia" : active ? "Aktif" : "Nonaktif";
  return <details className="hydro-option">
    <summary>{layer.name}<span className={`hydro-chip${!available ? " is-empty" : active ? " is-active" : ""}`}>{chip}</span></summary>
    <div className="hydro-content">
      <p className="hydro-key"><i className={layer.swatch} />{layer.swatchLabel}</p>
      <label className="layer-switch"><span>Tampilkan</span><input type="checkbox" disabled={!available} checked={Boolean(active)} onChange={() => props.onToggle(layer.key)} /></label>
      <label className="opacity-control"><span>Opasitas <output>{Math.round(props.opacity[layer.key] * 100)}%</output></span><input aria-label={"Opasitas " + layer.name} type="range" min="0" max="100" value={Math.round(props.opacity[layer.key] * 100)} disabled={!available || !active} onChange={event => props.onOpacity(layer.key, Number(event.target.value) / 100)} /></label>
      <small className="hydro-source">{available ? layer.sourceNote : layer.emptyNote}</small>
    </div>
  </details>;
}

export default function LayersPanel(props: Props) {
  return <aside className="tools-card layer-browser map-legend-overlay" aria-label="Lapisan peta">
    <header className="panel-heading map-legend-heading"><button className="overlay-close" aria-label="Tutup lapisan" onClick={props.onClose}><Icon name="close" />Tutup</button><h2>Lapisan peta <Icon name="layers" /></h2></header>

    <section>
      <h3><Icon name="pin" />Stasiun</h3>
      <label className="layer-switch"><span>Nama stasiun</span><input type="checkbox" checked={props.active.includes("labels")} onChange={() => props.onToggle("labels")} /></label>
      <label className="layer-switch"><span>Intensitas hujan</span><input type="checkbox" checked={props.active.includes("rainfall")} onChange={() => props.onToggle("rainfall")} /></label>
      <label className="layer-switch"><span>Kualitas air</span><input type="checkbox" checked={props.active.includes("quality")} onChange={() => props.onToggle("quality")} /></label>
      <small>Titik utama menunjukkan status muka air; lingkaran biru menunjukkan hujan dan cincin menunjukkan kualitas air.</small>
    </section>

    <section>
      <h3><Icon name="wave" />Hidrologi</h3>
      <label className="layer-switch"><span>Fokus wilayah Welang</span><input type="checkbox" checked={props.focusBasin && props.sources.basin} disabled={!props.sources.basin} onChange={event => props.onFocusBasin(event.target.checked)} /></label>
      <small>{props.sources.basin ? "Area di luar DAS diredupkan." : "Batas DAS belum tersedia; peta berfokus stasiun."}</small>
      {hydroLayers.map(layer => <HydroLayerRow key={layer.key} layer={layer} active={props.active} opacity={props.opacity} sources={props.sources} onToggle={props.onToggle} onOpacity={props.onOpacity} />)}
    </section>

    <section>
      <h3><Icon name="overview" />Konteks Jawa Timur</h3>
      <label className="layer-switch"><span>Batas kabupaten/kota</span><input type="checkbox" disabled={!props.sources.admin} checked={props.sources.admin && props.active.includes("admin")} onChange={() => props.onToggle("admin")} /></label>
      <label className="layer-switch"><span>Sungai besar Jawa Timur</span><input type="checkbox" disabled={!props.sources.eastRivers} checked={props.sources.eastRivers && props.active.includes("regionalRivers")} onChange={() => props.onToggle("regionalRivers")} /></label>
      <small>BIG untuk batas administrasi; OpenStreetMap untuk jaringan sungai regional.</small>
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
