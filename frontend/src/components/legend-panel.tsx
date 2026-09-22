import Icon from "./icon";
import WaterwayKey from "./waterway-key";
import { statusColors } from "@/lib/presentation";
import type { HydroSources } from "./layers-panel";
import type { LayerKey, StationStatus } from "@/lib/types";

const statusDescriptions: Record<StationStatus, string> = { Normal: "Muka air di bawah ambang waspada", Meningkat: "Tren muka air naik dalam 3 jam", Waspada: "Mendekati / melewati ambang waspada", Bahaya: "Melewati ambang bahaya" };

export default function LegendPanel({ onClose, activeLayers, sources }: { onClose: () => void; activeLayers: LayerKey[]; sources: HydroSources }) {
  return <aside className="legend-card map-legend-overlay" role="region" aria-label="Legenda peta">
    <header className="panel-heading map-legend-heading"><button className="overlay-close" aria-label="Tutup legenda" onClick={onClose}><Icon name="close" /></button><h2>Legenda <Icon name="legend" /></h2></header>
    <section className="legend-group">
      <span className="legend-heading">Muka air</span>
      {(Object.keys(statusDescriptions) as StationStatus[]).map((status) => <div className="legend-row legend-row-compact" key={status}><i className="legend-level-dot" style={{ background: statusColors[status] }} /><div><b>{status}</b><small>{statusDescriptions[status]}</small></div></div>)}
    </section>
    {activeLayers.includes("quality") && <section className="legend-group">
      <span className="legend-heading">Kualitas air</span>
      <div className="legend-row legend-row-compact"><i className="legend-quality quality-good" /><div><b>Baik</b><small>Parameter dalam rentang baik</small></div></div>
      <div className="legend-row legend-row-compact"><i className="legend-quality quality-moderate" /><div><b>Sedang</b><small>Satu atau lebih parameter perlu perhatian</small></div></div>
      <div className="legend-row legend-row-compact"><i className="legend-quality quality-poor" /><div><b>Buruk</b><small>Kondisi kualitas air tertekan</small></div></div>
    </section>}
    {activeLayers.includes("rainfall") && <section className="legend-group">
      <span className="legend-heading">Curah hujan 30 menit</span>
      <div className="legend-rain-scale"><span><i className="rain-small" /><b>Ringan</b></span><span><i className="rain-medium" /><b>Sedang</b></span><span><i className="rain-large" /><b>Lebat</b></span></div>
    </section>}
    <details className="legend-details"><summary>Simbol & label stasiun</summary><section><div className="legend-row"><i className="legend-dot" /><div><b>Stasiun pemantauan</b><small>Klik untuk memilih stasiun</small></div></div><div className="legend-row"><i className="legend-dot is-selected" /><div><b>Stasiun terpilih</b><small>Buka Grafik untuk melihat riwayatnya</small></div></div></section>
    <section><span className="legend-heading">Label stasiun</span><div className="legend-label-sample"><i className="legend-dot" /><span>Dhompo</span></div><small className="legend-note">Muncul saat layer "Nama stasiun" aktif</small></section></details>
    <details className="legend-details"><summary>Lapisan peta</summary><section>
      {activeLayers.includes("topology") && sources.topology && <div className="legend-row"><i className="legend-river" /><div><b>Sungai DAS Welang</b><small>OpenStreetMap contributors · ODbL</small></div></div>}
      {activeLayers.includes("surabayaRivers") && sources.surabayaRivers && <div className="legend-row"><i className="legend-river" /><div><b>Sungai Surabaya</b><small>OpenStreetMap contributors · ODbL</small></div></div>}
      {((activeLayers.includes("topology") && sources.topology) || (activeLayers.includes("surabayaRivers") && sources.surabayaRivers)) && <WaterwayKey />}
      {activeLayers.includes("subdas") && sources.subdas && <div className="legend-row"><i className="legend-subbasin" /><div><b>Sub-DAS stasiun</b><small>Partisi area aliran dari DEMNAS ~30 m</small></div></div>}
      {activeLayers.includes("basin") && sources.basin && <div className="legend-row"><i className="legend-basin" /><div><b>Batas DAS Welang</b><small>BIG · Atlas Wilayah Sungai · WELANG (11622)</small></div></div>}
      {activeLayers.includes("admin") && sources.admin && <div className="legend-row"><i className="legend-admin" /><div><b>Kabupaten/kota Jawa Timur</b><small>BIG · Batas Wilayah Administrasi</small></div></div>}
      {activeLayers.includes("regionalRivers") && sources.eastRivers && <div className="legend-row"><i className="legend-regional-river" /><div><b>Sungai besar Jawa Timur</b><small>OpenStreetMap contributors · ODbL</small></div></div>}
      {!activeLayers.some(key => ["basin", "topology", "surabayaRivers", "subdas", "admin", "regionalRivers"].includes(key)) && <small>Tidak ada overlay geospasial aktif.</small>}
    </section></details>
    <p className="legend-note">Seluruh pembacaan muka air, hujan, dan kualitas air pada tahap ini adalah data simulasi.</p>
  </aside>;
}
