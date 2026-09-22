export default function WaterwayKey() {
  return <ul className="waterway-key" aria-label="Jenis aliran">
    <li><i className="waterway-river" aria-hidden="true" /><span>Sungai <small>River · aliran utama</small></span></li>
    <li><i className="waterway-canal" aria-hidden="true" /><span>Kanal <small>Canal · saluran buatan</small></span></li>
    <li><i className="waterway-stream" aria-hidden="true" /><span>Anak sungai <small>Stream · aliran kecil</small></span></li>
  </ul>;
}
