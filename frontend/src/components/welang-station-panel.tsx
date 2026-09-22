"use client";

import { useMemo, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { StationSnapshot } from "@/lib/types";
import { chartTheme } from "@/lib/presentation";
import "./surabaya.css";

type Period = "6h" | "24h" | "7d";

const periods: Record<Period, { label: string; hours: number; intervalMinutes: number }> = {
  "6h": { label: "6 jam", hours: 6, intervalMinutes: 5 },
  "24h": { label: "24 jam", hours: 24, intervalMinutes: 15 },
  "7d": { label: "7 hari", hours: 168, intervalMinutes: 30 },
};

const clockFormat = new Intl.DateTimeFormat("id-ID", { timeZone: "Asia/Jakarta", hour: "2-digit", minute: "2-digit" });
const timeFormat = new Intl.DateTimeFormat("id-ID", { timeZone: "Asia/Jakarta", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

export default function WelangStationPanel({ station, showHeading = true }: { station: StationSnapshot; showHeading?: boolean }) {
  const [period, setPeriod] = useState<Period>("24h");
  const end = Date.now();
  const periodConfig = periods[period];
  const capacityPercentage = Math.min(100, Math.max(0, (station.valueM / station.dangerM) * 100));

  const history = useMemo(() => {
    const count = (periodConfig.hours * 60) / periodConfig.intervalMinutes;
    return Array.from({ length: count + 1 }, (_, index) => {
      const ageMinutes = (count - index) * periodConfig.intervalMinutes;
      const ageHours = ageMinutes / 60;
      const recentTrend = (station.delta3hM * Math.min(ageHours, 3)) / 3;
      const longCycle = Math.sin(ageHours / 5) * 0.08 + Math.sin(ageHours / 17) * 0.04;
      const rainPulse = Math.max(0, station.rainfall30mMm * Math.exp(-((ageHours - 2.5) ** 2) / 10));
      return {
        time: end - ageMinutes * 60000,
        valueM: Math.max(0.01, station.valueM - recentTrend - longCycle),
        rainMm: rainPulse,
      };
    });
  }, [end, periodConfig.hours, periodConfig.intervalMinutes, station.delta3hM, station.rainfall30mMm, station.valueM]);

  const forecasts = useMemo(() => {
    return Array.from({ length: 5 }, (_, index) => {
      const leadHours = index + 1;
      const valueM = station.valueM + (station.delta3hM / 3) * leadHours;
      return { leadHours, valueM, time: end + leadHours * 3600000 };
    });
  }, [end, station.delta3hM, station.valueM]);

  return (
    <section className="live-station-panel welang-preview" aria-label={`Preview detail ${station.name}`}>
      {showHeading && <header className="live-heading">
        <div>
          <span className="live-eyebrow">DAS Welang · Pasuruan</span>
          <h2>{station.name}</h2>
        </div>
      </header>}

      <div className="live-gauge-container" role="group" aria-label="Ringkasan muka air">
        <div className="live-level-visual">
          <div className="live-tank-card">
            <div className="live-tank-tube" title={`Muka air ${station.valueM.toFixed(2)} m dari ambang bahaya ${station.dangerM.toFixed(2)} m`}>
              <div className="live-tank-water" style={{ height: `${capacityPercentage}%` }}>
                <div className="live-tank-wave" />
              </div>
            </div>
            <span className="live-tank-label">{capacityPercentage.toFixed(0)}% menuju ambang bahaya</span>
          </div>
          <div className="live-reading-block">
            <span className="live-tank-label" style={{ color: "var(--blue)" }}>Tinggi Muka Air Sungai</span>
            <div className="live-water-headline">
              <span className="live-water-value">{(station.valueM * 100).toFixed(1)}</span>
              <span className="live-water-unit">cm</span>
            </div>
            <span className="live-meter-sub">{station.valueM.toFixed(2)} meter</span>
            <dl className="live-calc-breakdown">
              <div className="live-calc-item">
                <dt>Perubahan 3 jam</dt>
                <dd>{station.delta3hM >= 0 ? "+" : ""}{(station.delta3hM * 100).toFixed(1)} cm</dd>
              </div>
              <div className="live-calc-item">
                <dt>Status muka air</dt>
                <dd>{station.status}</dd>
              </div>
            </dl>
          </div>
        </div>
      </div>

      <h3 id="riwayat-sensor">Riwayat muka air</h3>
      <p className="live-note">
        Rentang waktu riwayat pengamatan muka air
      </p>

      <div style={{ display: "flex", gap: "6px", marginBottom: "12px" }}>
        {(Object.keys(periods) as Period[]).map((val) => (
          <button
            key={val}
            type="button"
            style={{
              padding: "6px 12px",
              fontSize: "var(--text-caption)",
              borderRadius: "var(--radius)",
              border: "1px solid var(--color-control-border)",
              background: period === val ? "var(--color-selection-bg)" : "var(--surface)",
              color: period === val ? "var(--blue)" : "inherit",
              borderColor: period === val ? "var(--blue)" : "var(--color-control-border)",
              cursor: "pointer",
            }}
            onClick={() => setPeriod(val)}
          >
            {periods[val].label}
          </button>
        ))}
      </div>

      <div className="live-chart">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={history} margin={{ top: 12, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid stroke={chartTheme.grid} vertical={false} />
            <XAxis
              dataKey="time"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(value) => clockFormat.format(value)}
              minTickGap={45}
              tick={chartTheme.tick}
            />
            <YAxis domain={["auto", "auto"]} tick={chartTheme.tick} width={50} tickFormatter={(v) => Number(v).toFixed(2)} />
            <Tooltip
              contentStyle={chartTheme.tooltip}
              labelFormatter={(value) => `${timeFormat.format(Number(value))} WIB`}
              formatter={(value) => [`${Number(value).toFixed(2)} m`, "Muka air"]}
            />
            <Area yAxisId={0} dataKey="valueM" fill={chartTheme.rainFill} stroke="none" isAnimationActive={false} />
            <Line dataKey="valueM" name="Muka air" stroke={chartTheme.water} strokeWidth={2.5} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <h3>Prakiraan +1 sampai +5 jam</h3>
      <p className="live-note">Simulasi prakiraan model hidrologi DAS Welang</p>
      <div className="live-table-wrap">
        <table>
          <caption>Prakiraan simulasi 5 jam ke depan</caption>
          <thead>
            <tr>
              <th>Waktu berlaku</th>
              <th>Horizon</th>
              <th>Tinggi air</th>
            </tr>
          </thead>
          <tbody>
            {forecasts.map((pt) => (
              <tr key={pt.leadHours}>
                <td>{clockFormat.format(pt.time)} WIB</td>
                <td>+{pt.leadHours} jam</td>
                <td>{pt.valueM.toFixed(2)} m</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <details className="live-extra">
        <summary>Parameter lingkungan & lokasi</summary>
        <dl>
          <div><dt>Curah hujan 30 mnt</dt><dd>{station.rainfall30mMm.toFixed(1)} mm</dd></div>
          <div><dt>Kualitas air</dt><dd>{station.quality.status}</dd></div>
          <div><dt>pH</dt><dd>{station.quality.ph.toFixed(2)}</dd></div>
          <div><dt>Suhu air</dt><dd>{station.quality.temperatureC.toFixed(1)} °C</dd></div>
          <div><dt>Koordinat</dt><dd>{station.latitude.toFixed(6)}, {station.longitude.toFixed(6)}</dd></div>
          <div><dt>Keyakinan koordinat</dt><dd>{station.confidence}</dd></div>
        </dl>
      </details>
    </section>
  );
}
