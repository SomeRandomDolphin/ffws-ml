"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { StationSnapshot } from "@/lib/types";
import { chartTheme } from "@/lib/presentation";
import Icon from "./icon";
import { SiteFooter, WelangBrand } from "./brand";
import LocationMap from "./location-map";
import "./station-monitor.css";

type ForecastPoint = { leadHours: number; valueM: number; status: StationSnapshot["status"] };
type Period = "6h" | "24h" | "7d";
type Scale = "linear" | "log";

const periods: Record<Period, { label: string; hours: number; intervalMinutes: number }> = {
  "6h": { label: "6 jam", hours: 6, intervalMinutes: 5 },
  "24h": { label: "24 jam", hours: 24, intervalMinutes: 15 },
  "7d": { label: "7 hari", hours: 168, intervalMinutes: 30 },
};
const clockFormat = new Intl.DateTimeFormat("id-ID", { timeZone: "Asia/Jakarta", hour: "2-digit", minute: "2-digit" });
const timeFormat = new Intl.DateTimeFormat("id-ID", { timeZone: "Asia/Jakarta", day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
const alertLevelAt = (value: number, station: StationSnapshot): "Normal" | "Waspada" | "Bahaya" => value >= station.dangerM ? "Bahaya" : value >= station.alertM ? "Waspada" : "Normal";
const trendLabel = (delta: number) => Math.abs(delta) < 0.03 ? "Stabil" : delta > 0 ? "Naik" : "Turun";

export default function StationDetail({ station, forecasts = [], timestamp = "2026-02-18T05:00:00Z" }: { station: StationSnapshot; forecasts?: ForecastPoint[]; timestamp?: string }) {
  const [period, setPeriod] = useState<Period>("24h");
  const [scale, setScale] = useState<Scale>("linear");
  const end = Date.parse(timestamp);
  const periodConfig = periods[period];
  const history = useMemo(() => {
    const count = periodConfig.hours * 60 / periodConfig.intervalMinutes;
    return Array.from({ length: count + 1 }, (_, index) => {
      const ageMinutes = (count - index) * periodConfig.intervalMinutes;
      const ageHours = ageMinutes / 60;
      const recentTrend = station.delta3hM * Math.min(ageHours, 3) / 3;
      const longCycle = Math.sin(ageHours / 5) * .08 + Math.sin(ageHours / 17) * .04;
      const rainPulse = Math.max(0, station.rainfall30mMm * Math.exp(-((ageHours - 2.5) ** 2) / 10));
      return { time: end - ageMinutes * 60000, valueM: Math.max(.01, station.valueM - recentTrend - longCycle), rainMm: rainPulse };
    });
  }, [end, periodConfig.hours, periodConfig.intervalMinutes, station.delta3hM, station.rainfall30mMm, station.valueM]);
  const summary = useMemo(() => {
    const values = history.map(point => point.valueM);
    return { min: Math.min(...values), max: Math.max(...values), latest: values.at(-1) ?? station.valueM };
  }, [history, station.valueM]);
  const domain: [number, number] = [Math.max(.01, Math.min(summary.min, station.alertM, station.dangerM) * .94), Math.max(summary.max, station.alertM, station.dangerM) * 1.04];
  const alertLevel = alertLevelAt(station.valueM, station);
  const trend = trendLabel(station.delta3hM);
  const toAlert = station.alertM - station.valueM;
  const toDanger = station.dangerM - station.valueM;
  const back = "/?station=" + encodeURIComponent(station.name);
  const stationCode = `WL-${station.name.replace(/[^a-z0-9]/gi, "").slice(0, 8).toUpperCase()}`;

  return <main className="wm-app wm-detail monitor">
    <header className="wm-header"><WelangBrand href={back} /><span className="detail-header-context">Data lokasi pemantauan</span></header>
    <div className="monitor-body">
      <Link className="monitor-back" href={back}><Icon name="back" />Kembali ke dashboard peta</Link>
      <header className="monitor-heading">
        <span className="monitor-eyebrow">Lokasi pemantauan</span>
        <h1>{station.name}</h1>
        <div className="monitor-identity"><strong>{stationCode}</strong><span>DAS Welang · Pasuruan, Jawa Timur</span><span className={`status-chip status-${alertLevel.toLowerCase()}`}>{alertLevel}</span></div>
        <p className="monitor-description">{station.description}</p>
      </header>

      <section className="monitor-section" aria-labelledby="current-heading">
        <div className="monitor-section-heading"><div><span className="monitor-eyebrow">Kondisi terkini</span><h2 id="current-heading">Muka air</h2></div><span className="monitor-updated">Pengamatan simulasi · {clockFormat.format(end)} WIB</span></div>
        <dl className="observation-grid">
          <div className="observation-primary"><dt>Pembacaan</dt><dd>{station.valueM.toFixed(2)} <small>m</small></dd><span>Status {alertLevel}</span></div>
          <div><dt>Perubahan 3 jam</dt><dd className={station.delta3hM > 0 ? "trend-rising" : "trend-falling"}>{station.delta3hM > 0 ? "+" : station.delta3hM < 0 ? "−" : ""}{Math.abs(station.delta3hM).toFixed(2)} <small>m</small></dd><span>{trend}</span></div>
          <div><dt>Ambang waspada</dt><dd>{station.alertM.toFixed(2)} <small>m</small></dd><span>{toAlert >= 0 ? `${toAlert.toFixed(2)} m di bawah ambang` : `${Math.abs(toAlert).toFixed(2)} m di atas ambang`}</span></div>
          <div><dt>Ambang bahaya</dt><dd>{station.dangerM.toFixed(2)} <small>m</small></dd><span>{toDanger >= 0 ? `${toDanger.toFixed(2)} m di bawah ambang` : `${Math.abs(toDanger).toFixed(2)} m di atas ambang`}</span></div>
        </dl>
      </section>

      <section className="monitor-section" aria-labelledby="history-heading">
        <div className="monitor-section-heading"><div><span className="monitor-eyebrow">Data kontinu simulasi</span><h2 id="history-heading">Riwayat muka air dan curah hujan</h2></div><span className="monitor-updated">{timeFormat.format(history[0].time)} – {timeFormat.format(end)} WIB</span></div>
        <div className="history-controls">
          <div><span>Rentang riwayat</span><div className="monitor-toggle" role="group" aria-label="Rentang riwayat">{(Object.keys(periods) as Period[]).map(value => <button key={value} aria-pressed={period === value} onClick={() => setPeriod(value)}>{periods[value].label}</button>)}</div></div>
          <div><span>Skala sumbu</span><div className="monitor-toggle" role="group" aria-label="Skala sumbu muka air">{(["linear", "log"] as Scale[]).map(value => <button key={value} aria-pressed={scale === value} onClick={() => setScale(value)}>{value === "linear" ? "Linear" : "Log"}</button>)}</div></div>
        </div>
        <details className="scale-explanation"><summary>Apa perbedaan skala Linear dan Log?</summary><div><p><strong>Linear</strong> menampilkan perubahan absolut secara proporsional dan direkomendasikan untuk membaca jarak muka air terhadap ambang.</p><p><strong>Log</strong> menekankan perubahan relatif saat rentang nilai sangat lebar. Skala ini dapat membuat jarak absolut terhadap ambang terlihat lebih kecil.</p></div></details>
        <div className="chart-caption"><div><strong>Muka air</strong><span>meter</span></div><div><strong>Curah hujan</strong><span>mm per interval</span></div><div><strong>Ringkasan</strong><span>minimum {summary.min.toFixed(2)} m · maksimum {summary.max.toFixed(2)} m · tren {trend.toLowerCase()}</span></div></div>
        <figure className="monitor-chart"><figcaption className="sr-only">Grafik riwayat {periodConfig.label}. Nilai terbaru {summary.latest.toFixed(2)} meter, minimum {summary.min.toFixed(2)} meter, maksimum {summary.max.toFixed(2)} meter.</figcaption><ResponsiveContainer width="100%" height="100%"><ComposedChart data={history} margin={{top:25,right:8,left:0,bottom:8}}>
          <CartesianGrid stroke={chartTheme.grid} vertical={false} />
          <XAxis dataKey="time" type="number" domain={["dataMin","dataMax"]} tickCount={6} minTickGap={38} tickFormatter={value => period === "7d" ? timeFormat.format(value).replace(/,? 2026/, "") : clockFormat.format(value)} tick={chartTheme.tick} axisLine={{stroke:chartTheme.grid}} tickLine={false} />
          <YAxis yAxisId="water" scale={scale} domain={domain} allowDataOverflow width={50} tickFormatter={value => Number(value).toFixed(1)} tick={chartTheme.tick} axisLine={false} tickLine={false} />
          <YAxis yAxisId="rain" orientation="right" width={40} tick={chartTheme.tick} axisLine={false} tickLine={false} />
          <Tooltip content={({active,payload}) => { const point = payload?.[0]?.payload as typeof history[number] | undefined; return active && point ? <div className="monitor-tooltip"><strong>{timeFormat.format(point.time)} WIB</strong><span>Muka air: {point.valueM.toFixed(2)} m</span><span>Hujan: {point.rainMm.toFixed(1)} mm</span><span>Status: {alertLevelAt(point.valueM,station)}</span></div> : null; }} />
          <Area yAxisId="rain" dataKey="rainMm" fill={chartTheme.rainFill} stroke={chartTheme.rain} isAnimationActive={false} />
          <Line yAxisId="water" dataKey="valueM" stroke={chartTheme.water} strokeWidth={2.5} dot={false} isAnimationActive={false} />
          <ReferenceLine yAxisId="water" y={station.alertM} stroke={chartTheme.warning} strokeDasharray="6 5" label={{value:"Waspada",position:"insideTopRight",fontSize:12,fill:chartTheme.warning}} />
          <ReferenceLine yAxisId="water" y={station.dangerM} stroke={chartTheme.danger} strokeDasharray="2 4" label={{value:"Bahaya",position:"insideTopRight",fontSize:12,fill:chartTheme.danger}} />
        </ComposedChart></ResponsiveContainer></figure>
        <details className="inline-data"><summary>Tampilkan data tabular</summary><div className="monitor-table-wrap"><table className="monitor-table"><caption>Riwayat muka air dan curah hujan, terbaru terlebih dahulu</caption><thead><tr><th scope="col">Waktu pengamatan</th><th scope="col">Muka air</th><th scope="col">Hujan</th><th scope="col">Status</th></tr></thead><tbody>{[...history].reverse().map(point => <tr key={point.time}><td>{timeFormat.format(point.time)} WIB</td><td>{point.valueM.toFixed(2)} m</td><td>{point.rainMm.toFixed(1)} mm</td><td className={`status-${alertLevelAt(point.valueM,station).toLowerCase()}`}>{alertLevelAt(point.valueM,station)}</td></tr>)}</tbody></table></div></details>
      </section>

      <section className="monitor-section" aria-labelledby="forecast-heading">
        <div className="monitor-section-heading"><div><span className="monitor-eyebrow">Simulasi tren</span><h2 id="forecast-heading">Prakiraan 5 jam</h2></div><span className="monitor-updated">Diterbitkan {timeFormat.format(end)} WIB · bukan peringatan resmi</span></div>
        <div className="monitor-table-wrap"><table className="monitor-table"><caption>Prakiraan muka air untuk lima jam berikutnya</caption><thead><tr><th scope="col">Waktu berlaku</th><th scope="col">Muka air</th><th scope="col">Status</th><th scope="col">Arah perubahan</th></tr></thead><tbody>{forecasts.map((point, index) => { const prior = index ? forecasts[index - 1].valueM : station.valueM; const pointStatus = alertLevelAt(point.valueM, station); return <tr key={point.leadHours}><td>{timeFormat.format(end + point.leadHours * 3600000)} WIB <small>+{point.leadHours} jam</small></td><td>{point.valueM.toFixed(2)} m</td><td className={`status-${pointStatus.toLowerCase()}`}>{pointStatus}</td><td>{trendLabel(point.valueM - prior)}</td></tr>; })}</tbody></table></div>
      </section>

      <section className="monitor-section" aria-labelledby="environment-heading">
        <div className="monitor-section-heading"><div><span className="monitor-eyebrow">Parameter pendukung</span><h2 id="environment-heading">Hujan dan kualitas air</h2></div><span className={`quality-label quality-${station.quality.status.toLowerCase()}`}>Kualitas {station.quality.status} · klasifikasi demo</span></div>
        <div className="monitor-table-wrap"><table className="monitor-table environment-table"><caption>Parameter lingkungan simulasi pada lokasi {station.name}</caption><thead><tr><th scope="col">Parameter</th><th scope="col">Nilai</th><th scope="col">Satuan</th><th scope="col">Keterangan</th></tr></thead><tbody>
          <tr><td>Curah hujan 30 menit</td><td>{station.rainfall30mMm.toFixed(1)}</td><td>mm</td><td>Simulasi</td></tr><tr><td>Akumulasi hujan 24 jam</td><td>{station.rainfall24hMm.toFixed(1)}</td><td>mm</td><td>Simulasi</td></tr><tr><td>pH</td><td>{station.quality.ph.toFixed(2)}</td><td>tanpa satuan</td><td>Derajat keasaman</td></tr><tr><td>Oksigen terlarut (DO)</td><td>{station.quality.dissolvedOxygenMgL.toFixed(1)}</td><td>mg/L</td><td>Dissolved oxygen</td></tr><tr><td>Suhu air</td><td>{station.quality.temperatureC.toFixed(1)}</td><td>°C</td><td>Simulasi</td></tr><tr><td>Kekeruhan</td><td>{station.quality.turbidityNtu.toFixed(1)}</td><td>NTU</td><td>Nephelometric Turbidity Unit</td></tr><tr><td>Total zat terlarut (TDS)</td><td>{station.quality.tdsMgL.toFixed(0)}</td><td>mg/L</td><td>Total dissolved solids</td></tr>
        </tbody></table></div>
      </section>

      <section className="monitor-section" aria-labelledby="threshold-heading">
        <div className="monitor-section-heading"><div><span className="monitor-eyebrow">Interpretasi</span><h2 id="threshold-heading">Definisi ambang</h2></div><span className="monitor-updated">Ambang demonstrasi per stasiun</span></div>
        <div className="monitor-table-wrap"><table className="monitor-table threshold-table"><caption>Aturan status muka air untuk {station.name}</caption><thead><tr><th scope="col">Status</th><th scope="col">Kondisi</th></tr></thead><tbody><tr><td className="status-normal">Normal</td><td>Di bawah {station.alertM.toFixed(2)} m</td></tr><tr><td className="status-waspada">Waspada</td><td>{station.alertM.toFixed(2)} m hingga di bawah {station.dangerM.toFixed(2)} m</td></tr><tr><td className="status-bahaya">Bahaya</td><td>Mulai {station.dangerM.toFixed(2)} m</td></tr></tbody></table></div>
      </section>

      <section className="monitor-section location-section" aria-labelledby="location-heading">
        <div className="monitor-section-heading"><div><span className="monitor-eyebrow">Konteks lokasi</span><h2 id="location-heading">Informasi stasiun</h2></div><Link href={back}>Buka di dashboard peta</Link></div>
        <div className="location-layout"><dl className="station-metadata"><div><dt>Kode stasiun</dt><dd>{stationCode}</dd></div><div><dt>Koordinat WGS84</dt><dd>{station.latitude.toFixed(6)}, {station.longitude.toFixed(6)}</dd></div><div><dt>Keyakinan koordinat</dt><dd>{station.confidence}</dd></div><div><dt>Sumber lokasi</dt><dd>{station.source}</dd></div><div><dt>Wilayah hidrologi</dt><dd>DAS Welang, Pasuruan</dd></div><div><dt>Status data</dt><dd>Simulasi · belum diverifikasi</dd></div></dl><figure className="location-figure"><LocationMap station={station} /><figcaption>{station.name} · peta dasar OpenStreetMap/OpenTopoMap</figcaption></figure></div>
      </section>
    </div>
    <SiteFooter />
  </main>;
}
