import Icon from "./icon";
import type { DashboardSnapshot } from "@/lib/types";
export default function OverviewPanel({ onClose, snapshot }: { onClose: () => void; snapshot: DashboardSnapshot }) {
  return <aside className="overview-card map-legend-overlay" role="region" aria-label="Ringkasan pemantauan">
    <header className="panel-heading map-legend-heading"><button className="overlay-close" aria-label="Tutup ringkasan" onClick={onClose}><Icon name="close" />Tutup</button><h2>Ringkasan</h2></header>
    <div className="overview-body">
      <div><span className="overview-divider">PEMANTAUAN SUMBER DAYA AIR</span><h3>ITS Water Dashboard</h3><p className="overview-desc">Jelajahi informasi muka air, curah hujan, dan kualitas air melalui peta interaktif untuk memahami kondisi pada lokasi pemantauan yang tersedia.</p></div>
      <section><span className="legend-heading">Jelajahi wilayah</span><p className="overview-desc">Gunakan pilihan cakupan dan kontrol peta untuk melihat konteks wilayah, jaringan sungai, serta daerah aliran sungai.</p></section>
      <section><span className="legend-heading">Amati kondisi stasiun</span><p className="overview-desc">Cari atau pilih stasiun pada peta untuk melihat pembacaan, status, dan perubahan muka air. Buka grafik untuk menelusuri riwayatnya.</p></section>
      <section><span className="legend-heading">Sesuaikan informasi peta</span><p className="overview-desc">Gunakan Lapisan peta untuk memilih informasi yang ditampilkan. Buka Legenda untuk memahami simbol dan warna status.</p></section>
      <div className="data-provenance"><span className="simulation-label">Data simulasi</span><p>Waktu data<br /><strong>{snapshot.timestamp}</strong></p><small>Pembacaan ilustratif untuk demonstrasi, bukan peringatan resmi.</small></div>
      <details className="about-dashboard"><summary>Tentang dashboard</summary><p>Departemen Teknik Sipil · Institut Teknologi Sepuluh Nopember.</p><p>Batas administrasi dan DAS: BIG. Sungai: OpenStreetMap. Sub-DAS: delineasi DEMNAS.</p></details>
    </div>
  </aside>;
}
