"use client";
import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatSensorLabel, liveStateLabel, liveTime, type LiveStation } from "@/lib/surabaya";
import { chartTheme } from "@/lib/presentation";
import "./surabaya.css";

export default function LiveStationPanel({ station, disconnected = false }: { station: LiveStation; disconnected?: boolean }) {
  const [selected, setSelected] = useState("");
  const sensor = station.sensors.find(item => item.id === selected) ?? station.sensors[0];
  const stale = !station.observedAt || Date.now() - Date.parse(station.observedAt) > 600000;
  const hideForecast = disconnected || stale || station.state !== "live";
  const isPucang = station.name.toLowerCase().includes("pucang");
  const isKalibokor = station.name.toLowerCase().includes("kalibokor");
  const isHangTuah = station.name.toLowerCase().includes("hang tuah");
  const usesDistanceReference = isPucang || isKalibokor || isHangTuah;
  const groundReferenceCm = isPucang ? 530 : isKalibokor || isHangTuah ? 500 : 300;
  const history = sensor?.history.map((point, index, all) => ({
    time: Date.parse(point.time), valueCm: point.valueCm === null ? null : usesDistanceReference ? Math.max(0, groundReferenceCm - point.valueCm) : point.valueCm,
    // Break lines where the next observation skips a 30-minute interval.
    gap: index > 0 && Date.parse(point.time) - Date.parse(all[index - 1].time) > 3600000,
  })).flatMap(point => point.gap ? [{ time: point.time - 1, valueCm: null, gap: true }, point] : [point]) ?? [];
  const sensorDistanceCm = sensor?.valueCm ?? null;
  // Pucang Anom memakai acuan 5,3 m; Kalibokor dan Hang Tuah 5 m; stasiun lain tetap 3 m.

  // Rumus: tinggi air = jarak sensor ke dasar tanah/kolam - jarak sensor ke muka air
  const waterLevelCm = sensorDistanceCm !== null ? Math.max(0, groundReferenceCm - sensorDistanceCm) : null;
  const waterLevelM = waterLevelCm !== null ? waterLevelCm / 100 : null;
  // Persentase kapasitas muka air kolam / saluran
  const percentage = waterLevelCm !== null ? Math.min(100, Math.max(0, (waterLevelCm / groundReferenceCm) * 100)) : 0;

  return <section className="live-station-panel" aria-label={`Data live ${station.name}`}>
    {station.error && <p role="status" className="live-warning">{station.error}</p>}

    <div className="live-gauge-container">
      {station.sensors.length > 1 && (
        <div className="live-sensor-tabs" role="group" aria-label="Pilih kanal sensor">
          {station.sensors.map(item => (
            <button
              key={item.id}
              type="button"
              className="live-sensor-tab"
              aria-pressed={sensor?.id === item.id}
              onClick={() => setSelected(item.id)}
            >
              {formatSensorLabel(item.id, station.name)}
            </button>
          ))}
        </div>
      )}

      <div className="live-level-visual">
        <div className="live-tank-card">
          <div className="live-tank-tube" title={`Tinggi air ${waterLevelCm !== null ? waterLevelCm.toFixed(1) : 0} cm dari acuan ${groundReferenceCm.toFixed(0)} cm`}>
            <div className="live-tank-water" style={{ height: `${percentage}%` }}>
              <div className="live-tank-wave" />
            </div>
          </div>
          <span className="live-tank-label">{percentage.toFixed(0)}% Kapasitas</span>
        </div>

        <div className="live-reading-block">
          <span className="live-tank-label" style={{ color: "var(--blue)" }}>Tinggi Muka Air Sungai</span>
          <div className="live-water-headline">
            <span className="live-water-value">
              {waterLevelCm !== null ? waterLevelCm.toFixed(1) : "—"}
            </span>
            <span className="live-water-unit">cm</span>
          </div>
          <span className="live-meter-sub">
            {waterLevelM !== null ? `${waterLevelM.toFixed(2)} meter di atas dasar` : "Data belum tersedia"}
          </span>

          <dl className="live-calc-breakdown">
            <div className="live-calc-item">
              <dt>Jarak ke Muka Air (Distance)</dt>
              <dd>{sensorDistanceCm !== null ? `${sensorDistanceCm.toFixed(1)} cm` : "—"}</dd>
            </div>
            <div className="live-calc-item">
              <dt>Acuan Tinggi Sensor ke Dasar</dt>
              <dd>{groundReferenceCm.toFixed(1)} cm</dd>
            </div>
          </dl>
        </div>
      </div>
    </div>

    <h3 id="riwayat-sensor">Riwayat sensor {sensor?.id}</h3>
    <p className="live-note">24 jam sebelum pembacaan terakhir · sensor ditampilkan terpisah.</p>
    {history.length ? <div className="live-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={history} margin={{ top: 12, right: 16, bottom: 8, left: 0 }}><CartesianGrid stroke={chartTheme.grid} vertical={false} /><XAxis dataKey="time" type="number" domain={["dataMin", "dataMax"]} tickFormatter={value => liveTime(new Date(value).toISOString())} minTickGap={45} tick={chartTheme.tick} /><YAxis domain={["auto", "auto"]} tick={chartTheme.tick} width={55} /><Tooltip contentStyle={chartTheme.tooltip} labelFormatter={value => liveTime(new Date(Number(value)).toISOString())} /><Line dataKey="valueCm" name="Tinggi air" unit=" cm" stroke={chartTheme.water} strokeWidth={2.5} dot={false} connectNulls={false} isAnimationActive={false} /></LineChart></ResponsiveContainer></div> : <p>Riwayat belum tersedia.</p>}
    <h3>Prakiraan +1 sampai +5 jam</h3>
    {hideForecast ? <p className="live-note">Prediksi terkini tidak tersedia: data terlambat atau koneksi terputus.</p> : <>
      {!!sensor?.forecast.points.length && <div className="live-table-wrap"><table><caption>Acuan prediksi: {liveTime(sensor.forecast.issuedAt)} · bukan peringatan resmi</caption><thead><tr><th>Waktu berlaku</th><th>Horizon</th><th>Tinggi air</th></tr></thead><tbody>{sensor.forecast.points.map(point => <tr key={point.leadHours}><td>{liveTime(point.time)}</td><td>+{point.leadHours} jam</td><td>{(usesDistanceReference ? Math.max(0, groundReferenceCm - point.valueCm) : point.valueCm).toFixed(1)} cm</td></tr>)}</tbody></table></div>}
    </>}
    <details className="live-extra"><summary>Data hujan dan lokasi</summary>{station.rainfall ? <><p>Hujan: pembacaan mentah, satuan/akumulasi belum dikonfirmasi · {liveTime(station.rainfall.observedAt)}</p>{station.rainfall.error && <p>{station.rainfall.error}</p>}<dl>{Object.entries(station.rainfall.values).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value ?? "—"}</dd></div>)}</dl></> : <p>Data hujan belum tersedia.</p>}{station.coordinateSource && <a href={station.coordinateSource} target="_blank" rel="noreferrer">Sumber lokasi</a>}</details>
  </section>;
}
