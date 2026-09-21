"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatSensorLabel, liveStateLabel, liveTime, type LiveStation } from "@/lib/surabaya";
import { chartTheme } from "@/lib/presentation";
import Icon from "./icon";
import { SiteFooter, WelangBrand } from "./brand";
import LocationMap from "./location-map";
import { useSurabaya } from "./use-surabaya";
import "./station-monitor.css";

type Period = "6h" | "24h" | "7d";

const periods: Record<Period, { label: string; hours: number }> = {
  "6h": { label: "6 jam", hours: 6 },
  "24h": { label: "24 jam", hours: 24 },
  "7d": { label: "7 hari", hours: 168 },
};

const clockFormat = new Intl.DateTimeFormat("id-ID", { timeZone: "Asia/Jakarta", hour: "2-digit", minute: "2-digit" });
const timeFormat = new Intl.DateTimeFormat("id-ID", { timeZone: "Asia/Jakarta", day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });

function windDirection(degrees: number | null | undefined) {
  if (degrees === null || degrees === undefined || !Number.isFinite(degrees)) return "—";
  const points = ["Utara", "Timur Laut", "Timur", "Tenggara", "Selatan", "Barat Daya", "Barat", "Barat Laut"];
  return `${Math.round(degrees)}° · ${points[Math.round(degrees / 45) % points.length]}`;
}

function telemetryNumber(value: number | null | undefined, digits = 2) {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

export default function SurabayaDetail({ name }: { name: string }) {
  const { data, error } = useSurabaya();
  const station = data?.stations.find(item => item.name === name);

  const [selectedSensorId, setSelectedSensorId] = useState("");
  const [period, setPeriod] = useState<Period>("24h");

  const sensor = station?.sensors.find(item => item.id === selectedSensorId) ?? station?.sensors[0];
  const sensorLabel = sensor ? formatSensorLabel(sensor.id, station?.name) : "—";

  const nowMs = useMemo(() => {
    if (station?.observedAt) {
      const parsed = Date.parse(station.observedAt);
      if (!Number.isNaN(parsed)) return parsed;
    }
    return Date.now();
  }, [station?.observedAt]);

  const periodConfig = periods[period];

  const isPucang = name.toLowerCase().includes("pucang");
  const isKalibokor = name.toLowerCase().includes("kalibokor");
  const isHangTuah = name.toLowerCase().includes("hang tuah");
  const usesDistanceReference = isPucang || isKalibokor || isHangTuah;
  const groundReferenceCm = isPucang ? 530 : isKalibokor || isHangTuah ? 500 : 300;
  const history = useMemo(() => {
    if (!sensor || !sensor.history || sensor.history.length === 0) return [];
    const minTime = nowMs - periodConfig.hours * 3600000;
    
    // Urutkan riwayat dari yang paling lampau ke terkini
    const sorted = [...sensor.history]
      .map(pt => ({ time: Date.parse(pt.time), valueCm: pt.valueCm }))
      .filter(pt => !Number.isNaN(pt.time) && pt.time >= minTime)
      .sort((a, b) => a.time - b.time);

    return sorted.map(pt => ({
      time: pt.time,
      valueCm: pt.valueCm,
      valueM: pt.valueCm !== null ? (usesDistanceReference ? Math.max(0, groundReferenceCm - pt.valueCm) : pt.valueCm) / 100 : null,
    }));
  }, [sensor, nowMs, periodConfig.hours, usesDistanceReference, groundReferenceCm]);

  const sensorDistanceCm = sensor?.valueCm ?? null;
  const stationDescription = isPucang
    ? "Rumah Pompa Pucang Anom, Surabaya Timur merupakan infrastruktur pengendalian banjir yang berfungsi untuk membantu mengalirkan dan membuang kelebihan air dari kawasan sekitarnya. Fasilitas ini mendukung sistem drainase perkotaan dalam mengurangi risiko genangan, terutama saat curah hujan tinggi, serta menjaga kelancaran aliran air di wilayah Surabaya Timur."
    : "Stasiun pemantauan tinggi muka air perkotaan Surabaya. Nilai pengamatan dalam satuan cm (asumsi operasional) yang dikonversi ke meter untuk analisis hidrologi terpadu.";
  const telemetry = station?.telemetry;
  const telemetryIsStale = telemetry?.observedAt ? Date.now() - Date.parse(telemetry.observedAt) > 600000 : false;

  // Rumus: tinggi air = acuan tinggi sensor ke dasar - jarak sensor ke muka air
  const waterLevelCm = sensorDistanceCm !== null ? Math.max(0, groundReferenceCm - sensorDistanceCm) : null;
  const waterLevelM = waterLevelCm !== null ? waterLevelCm / 100 : null;
  const percentage = waterLevelCm !== null ? Math.min(100, Math.max(0, (waterLevelCm / groundReferenceCm) * 100)) : 0;

  const summary = useMemo(() => {
    const validValues = history.map(p => p.valueM).filter((v): v is number => v !== null);
    if (!validValues.length) {
      const currentM = waterLevelM;
      return { min: currentM ?? 0, max: currentM ?? 0, latest: currentM ?? 0 };
    }
    return {
      min: Math.min(...validValues),
      max: Math.max(...validValues),
      latest: validValues.at(-1) ?? (waterLevelM ?? 0),
    };
  }, [history, waterLevelM]);

  const domain: [number, number] = useMemo(() => {
    const min = Math.max(0, summary.min * 0.9);
    const max = Math.max(0.5, summary.max * 1.1);
    return [min, max];
  }, [summary]);

  const back = `/?station=${encodeURIComponent(name)}`;
  const stationCode = `SBY-${name.replace(/[^a-z0-9]/gi, "").slice(0, 8).toUpperCase()}`;
  const isDisconnected = Boolean(error);
  const statusLabel = isDisconnected ? "Koneksi terputus" : station ? liveStateLabel[station.state] : "Memuat…";

  return (
    <main className="wm-app wm-detail monitor">
      <header className="wm-header">
        <WelangBrand href={back} />
        <span className="detail-header-context">Pemantauan sungai Surabaya</span>
      </header>

      <div className="monitor-body">
        <Link className="monitor-back" href={back}>
          <Icon name="back" />
          Kembali ke dashboard peta
        </Link>

        {error && <p role="alert" className="live-warning" style={{ marginBottom: "16px" }}>{error}</p>}

        {!station ? (
          <div className="live-loading">
            {error
              ? `Data ${name} belum dapat dimuat. Koneksi akan dicoba kembali secara berkala.`
              : data
              ? `Data stasiun ${name} belum tersedia dari sumber database.`
              : `Memuat data sensor ${name}…`}
          </div>
        ) : (
          <>
            <header className="monitor-heading">
              <h1>{station.name}</h1>
              <div className="monitor-identity">
                <strong>{stationCode}</strong>
                <span>Wilayah Surabaya · Jawa Timur</span>
                {isDisconnected && (
                  <span className="status-chip status-waspada">
                    Koneksi terputus
                  </span>
                )}
              </div>
              <p className="monitor-description">
                {stationDescription}
              </p>
            </header>

            {station.sensors.length > 1 && (
              <section className="monitor-section sensor-select-section" aria-labelledby="sensor-select-heading">
                <div className="monitor-section-heading">
                  <div>
                    <h2 id="sensor-select-heading">Pilih sumber data</h2>
                  </div>
                </div>
                <div className="sensor-picker" role="group" aria-label="Pilih sensor">
                  {station.sensors.map((item) => {
                    const isCurrent = (sensor?.id ?? "") === item.id;
                    const label = formatSensorLabel(item.id, station.name);
                    return (
                      <button
                        key={item.id}
                        type="button"
                        className={`sensor-option${isCurrent ? " is-active" : ""}`}
                        aria-pressed={isCurrent}
                        onClick={() => setSelectedSensorId(item.id)}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
              </section>
            )}

            <section className="monitor-section" aria-labelledby="current-heading">
              <div className="monitor-section-heading">
                <div>
                  <h2 id="current-heading">Muka air</h2>
                </div>
              </div>
              <dl className="observation-grid">
                <div className="observation-primary">
                  <dt>Tinggi muka air</dt>
                  <dd>
                    {waterLevelM !== null ? waterLevelM.toFixed(2) : "—"}{" "}
                    <small>m</small>
                  </dd>
                  <span>
                    {waterLevelCm !== null ? `${waterLevelCm.toFixed(1)} cm di atas dasar` : "Nilai belum tersedia"}
                  </span>
                </div>
                <div>
                  <dt>Jarak sensor (Distance)</dt>
                  <dd style={{ fontSize: "var(--text-panel)" }}>
                    {sensorDistanceCm !== null ? `${sensorDistanceCm.toFixed(1)}` : "—"}{" "}
                    <small>cm</small>
                  </dd>
                  <span>Jarak ke permukaan air</span>
                </div>
                <div>
                  <dt>Acuan tinggi dasar</dt>
                  <dd style={{ fontSize: "var(--text-panel)" }}>
                    {groundReferenceCm.toFixed(1)} <small>cm</small>
                  </dd>
                  <span>{isPucang ? "Kedalaman kolam pompa" : "Acuan tinggi sensor ke tanah"}</span>
                </div>
                <div>
                  <dt>Sensor aktif</dt>
                  <dd style={{ fontSize: "var(--text-panel)" }}>
                    {sensorLabel}
                  </dd>
                  <span>Total {station.sensors.length} channel sensor</span>
                </div>
              </dl>
            </section>

            <section className="monitor-section" aria-labelledby="history-heading">
              <div className="monitor-section-heading">
                <div>
                  <span className="monitor-eyebrow">Data riwayat sensor</span>
                  <h2 id="history-heading">Riwayat muka air {sensorLabel}</h2>
                </div>
              </div>
              <div className="history-controls">
                <div>
                  <span>Rentang riwayat</span>
                  <div className="monitor-toggle" role="group" aria-label="Rentang riwayat">
                    {(Object.keys(periods) as Period[]).map(value => (
                      <button
                        key={value}
                        type="button"
                        aria-pressed={period === value}
                        onClick={() => setPeriod(value)}
                      >
                        {periods[value].label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
              <figure className="monitor-chart">
                <figcaption className="sr-only">
                  Grafik riwayat {periodConfig.label}. Nilai terbaru {summary.latest.toFixed(2)} meter, minimum {summary.min.toFixed(2)} meter, maksimum {summary.max.toFixed(2)} meter.
                </figcaption>
                {history.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={history} margin={{ top: 25, right: 8, left: 0, bottom: 8 }}>
                      <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                      <XAxis
                        dataKey="time"
                        type="number"
                        domain={["dataMin", "dataMax"]}
                        tickCount={6}
                        minTickGap={38}
                        tickFormatter={value => period === "7d" ? timeFormat.format(value).replace(/,? 2026/, "") : clockFormat.format(value)}
                        tick={chartTheme.tick}
                        axisLine={{ stroke: chartTheme.grid }}
                        tickLine={false}
                      />
                      <YAxis
                        yAxisId="water"
                        scale="linear"
                        domain={domain}
                        allowDataOverflow
                        width={64}
                        tickFormatter={value => Number(value).toFixed(2)}
                        tick={chartTheme.tick}
                        axisLine={false}
                        tickLine={false}
                        label={{ value: "Muka air (m)", angle: -90, position: "insideLeft", fill: "var(--muted)", fontSize: 11 }}
                      />
                      <Tooltip
                        content={({ active, payload }) => {
                          const point = payload?.[0]?.payload as (typeof history)[number] | undefined;
                          return active && point ? (
                            <div className="monitor-tooltip">
                              <strong>{timeFormat.format(point.time)} WIB</strong>
                              <span>Muka air: {point.valueM !== null ? `${point.valueM.toFixed(2)} m` : "—"}</span>
                              <span>Nilai sensor: {point.valueCm !== null ? `${point.valueCm.toFixed(1)} cm` : "—"}</span>
                            </div>
                          ) : null;
                        }}
                      />
                      <Area
                        yAxisId="water"
                        dataKey="valueM"
                        fill={chartTheme.rainFill}
                        stroke="none"
                        isAnimationActive={false}
                      />
                      <Line
                        yAxisId="water"
                        dataKey="valueM"
                        stroke={chartTheme.water}
                        strokeWidth={2.5}
                        dot={false}
                        isAnimationActive={false}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ display: "grid", placeItems: "center", height: "100%", color: "var(--muted)" }}>
                    Riwayat data untuk rentang waktu ini belum tersedia.
                  </div>
                )}
              </figure>
              <details className="inline-data">
                <summary>Tampilkan data tabular</summary>
                <div className="monitor-table-wrap">
                  <table className="monitor-table">
                    <caption>Riwayat tinggi muka air sensor {sensorLabel}, terbaru terlebih dahulu</caption>
                    <thead>
                      <tr>
                        <th scope="col">Waktu pengamatan</th>
                        <th scope="col">Muka air (m)</th>
                        <th scope="col">Muka air (cm)</th>
                        <th scope="col">Sensor</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.length > 0 ? (
                        [...history].reverse().map(point => (
                          <tr key={point.time}>
                            <td>{timeFormat.format(point.time)} WIB</td>
                            <td>{point.valueM !== null ? `${point.valueM.toFixed(2)} m` : "—"}</td>
                            <td>{point.valueCm !== null ? `${point.valueCm.toFixed(1)} cm` : "—"}</td>
                            <td>{sensorLabel}</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={4} style={{ textAlign: "center", color: "var(--muted)" }}>Belum ada rekaman data riwayat</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </details>
            </section>

            <section className="monitor-section" aria-labelledby="forecast-heading">
              <div className="monitor-section-heading">
                <div>
                  <span className="monitor-eyebrow">Estimasi / Prediksi</span>
                  <h2 id="forecast-heading">Prakiraan +1 sampai +5 jam</h2>
                </div>
                <span className="monitor-updated">
                  {sensor?.forecast.issuedAt ? `Acuan ${liveTime(sensor.forecast.issuedAt)}` : "Belum diterbitkan"} · bukan peringatan resmi
                </span>
              </div>
              <p className="live-note" style={{ margin: "0 0 16px" }}>
                {sensor?.forecast.method === "urban_file"
                  ? "Model terkalibrasi Hang Tuah · eksperimental"
                  : sensor?.forecast.method === "persistence"
                  ? "Metode persistence baseline · nilai diproyeksikan mengikuti tren stabil"
                  : "Prediksi belum tersedia untuk channel sensor ini."}
                {sensor?.forecast.reason ? ` (${sensor.forecast.reason})` : ""}
              </p>
              <div className="monitor-table-wrap">
                <table className="monitor-table">
                  <caption>Prakiraan tinggi muka air untuk horizon 5 jam ke depan</caption>
                  <thead>
                    <tr>
                      <th scope="col">Waktu berlaku</th>
                      <th scope="col">Horizon</th>
                      <th scope="col">Estimasi muka air (m)</th>
                      <th scope="col">Estimasi tinggi (cm)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sensor?.forecast.points && sensor.forecast.points.length > 0 ? (
                      sensor.forecast.points.map(pt => (
                        <tr key={pt.leadHours}>
                          <td>{liveTime(pt.time)}</td>
                          <td>+{pt.leadHours} jam</td>
                          <td>{((usesDistanceReference ? Math.max(0, groundReferenceCm - pt.valueCm) : pt.valueCm) / 100).toFixed(2)} m</td>
                          <td>{pt.valueCm.toFixed(1)} cm</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={4} style={{ textAlign: "center", color: "var(--muted)" }}>
                          Prediksi tidak tersedia (koneksi terputus atau data terlambat).
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>

            {(station.rainfall || (isHangTuah && telemetry)) && (
              <section className="monitor-section" aria-labelledby="rainfall-heading">
                <div className="monitor-section-heading">
                  <div>
                    <span className="monitor-eyebrow">Parameter pendukung</span>
                    <h2 id="rainfall-heading">Kondisi sekitar</h2>
                  </div>
                  <span className={`monitor-updated${telemetryIsStale ? " is-stale" : ""}`}>
                    {telemetry ? (telemetryIsStale ? "Data terakhir" : "Pembacaan terbaru") : "Pengamatan"}{(telemetry?.observedAt ?? station.rainfall?.observedAt) ? ` · ${liveTime(telemetry?.observedAt ?? station.rainfall?.observedAt ?? "")}` : ""}
                  </span>
                </div>
                <div className="monitor-table-wrap">
                  <table className="monitor-table environment-table">
                    <caption>Parameter pendukung kondisi sekitar stasiun {station.name}</caption>
                    <thead>
                      <tr>
                        <th scope="col">Parameter</th>
                        <th scope="col">Nilai</th>
                        <th scope="col">Keterangan</th>
                      </tr>
                    </thead>
                    <tbody>
                      {station.rainfall && Object.entries(station.rainfall.values).map(([key, val]) => (
                        <tr key={key}>
                          <td>{key}</td>
                          <td>{val !== null ? val : "—"}</td>
                          <td>Pembacaan mentah stasiun</td>
                        </tr>
                      ))}
                      {isHangTuah && telemetry && (
                        <>
                          <tr>
                            <td>Arah angin</td>
                            <td>{windDirection(telemetry.windir)}</td>
                            <td>derajat dan arah</td>
                          </tr>
                          <tr>
                            <td>Angin rata-rata</td>
                            <td>{telemetryNumber(telemetry.windavg)}</td>
                            <td>nilai mentah</td>
                          </tr>
                          <tr>
                            <td>Angin maksimum</td>
                            <td>{telemetryNumber(telemetry.windmax)}</td>
                            <td>nilai mentah</td>
                          </tr>
                          <tr>
                            <td>Intensitas cahaya</td>
                            <td>{telemetryNumber(telemetry.lightlevel, 0)}</td>
                            <td>nilai mentah</td>
                          </tr>
                        </>
                      )}
                    </tbody>
                  </table>
                </div>
                {isHangTuah && telemetry && (
                  <p className="telemetry-note">
                    Satuan angin dan cahaya belum terverifikasi di sumber database. Nilai ini digunakan sebagai konteks tambahan, bukan pembacaan utama.
                  </p>
                )}
              </section>
            )}

            <section className="monitor-section location-section" aria-labelledby="location-heading">
              <div className="monitor-section-heading">
                <div>
                  <span className="monitor-eyebrow">Konteks lokasi</span>
                  <h2 id="location-heading">Informasi fasilitas & lokasi</h2>
                </div>
                <Link href={back}>Buka di dashboard peta</Link>
              </div>
              <div className="location-layout">
                <dl className="station-metadata">
                  <div><dt>Kode stasiun</dt><dd>{stationCode}</dd></div>
                  <div>
                    <dt>Koordinat WGS84</dt>
                    <dd>
                      {station.latitude !== null && station.longitude !== null
                        ? `${station.latitude.toFixed(6)}, ${station.longitude.toFixed(6)}`
                        : "Belum tercatat"}
                    </dd>
                  </div>
                  <div>
                    <dt>Sumber koordinat</dt>
                    <dd>
                      {station.coordinateSource ? (
                        <a href={station.coordinateSource} target="_blank" rel="noreferrer">
                          Tautan sumber
                        </a>
                      ) : (
                        "Inventaris sensor dinas"
                      )}
                    </dd>
                  </div>
                  <div><dt>Wilayah hidrologi</dt><dd>Sistem Sungai & Saluran Surabaya</dd></div>
                  <div><dt>Status data</dt><dd>Live Database · {statusLabel}</dd></div>
                </dl>
                <figure className="location-figure">
                  {station.latitude !== null && station.longitude !== null ? (
                    <LocationMap
                      station={{
                        name: station.name,
                        latitude: station.latitude,
                        longitude: station.longitude,
                        color: "#0284c7",
                      }}
                    />
                  ) : (
                    <div style={{ height: "350px", display: "grid", placeItems: "center", background: "var(--paper)", border: "1px solid var(--line)", borderRadius: "var(--panel-radius)" }}>
                      Koordinat peta belum tersedia
                    </div>
                  )}
                  <figcaption>{station.name} · peta dasar OpenStreetMap/OpenTopoMap</figcaption>
                </figure>
              </div>
            </section>
          </>
        )}
      </div>
      <SiteFooter />
    </main>
  );
}
