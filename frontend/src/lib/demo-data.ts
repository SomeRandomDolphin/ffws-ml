import type { DashboardSnapshot, StationSnapshot, StationStatus, WaterQualityStatus } from "./types";
import { stationGeography } from "./stations";
import { statusColors } from "./presentation";
export { statusColors, qualityColors } from "./presentation";
const statusFor = (i: number, lead: number): StationStatus => i === 13 ? (lead >= 2 ? "Bahaya" : "Waspada") : i % 7 === 0 ? "Meningkat" : i % 5 === 0 ? "Waspada" : "Normal";
export function stationHistory(station: StationSnapshot) {
  return Array.from({ length: 25 }, (_, index) => ({ time: `${index - 24 < 0 ? index - 24 : `+${index - 24}`}j`, valueM: station.valueM - 0.72 + index * 0.03 + Math.sin(index / 2) * 0.08, rainMm: Math.max(0, station.rainfall30mMm * 1.8 * Math.exp(-((index - 18) ** 2) / 20)) }));
}
export function getDemoSnapshot(leadHours = 0): DashboardSnapshot {
  const stations: StationSnapshot[] = stationGeography.map((geo, i) => {
    const base = geo.name === "Dhompo" ? 8.65 : 2.15 + (i % 5) * 0.48;
    const valueM = base + leadHours * (geo.name === "Dhompo" ? 0.12 : 0.035) + Math.sin(i) * 0.08;
    const delta3hM = (i % 3 === 0 ? 1 : -1) * (0.08 + (i % 4) * 0.04);
    const status = statusFor(i, leadHours);
    const rainfall30mMm = Math.max(0, 2.4 + (i % 6) * 1.7 + Math.sin(i * 1.3) * 1.4 + leadHours * .35);
    const rainfall24hMm = rainfall30mMm * (5.8 + (i % 4) * .7);
    const qualityStatus: WaterQualityStatus = i % 8 === 5 ? "Buruk" : i % 4 === 1 ? "Sedang" : "Baik";
    const quality = {
      ph: 6.55 + (i % 5) * .23 - (qualityStatus === "Buruk" ? .38 : 0),
      temperatureC: 25.2 + (i % 6) * .62,
      dissolvedOxygenMgL: 6.7 - (i % 4) * .55 - (qualityStatus === "Buruk" ? 1.5 : 0),
      turbidityNtu: 9 + (i % 6) * 5.8 + (qualityStatus === "Buruk" ? 28 : 0),
      tdsMgL: 118 + (i % 7) * 34 + (qualityStatus === "Buruk" ? 125 : 0),
      status: qualityStatus,
    };
    return { ...geo, dataMode: "simulation" as const, valueM, delta3hM, status, alertM: base + 0.7, dangerM: base + 1.32, color: statusColors[status], rainfall30mMm, rainfall24hMm, quality };
  });
  const dhompo = stations.find((station) => station.name === "Dhompo")!;
  const history = stationHistory(dhompo);
  return { timestamp: "18 Feb 2026 · 12:00:00 WIB", leadHours, stations, forecasts: Array.from({ length: 5 }, (_, index) => ({ leadHours: index + 1, valueM: 8.65 + (index + 1) * 0.1, status: index > 1 ? "Bahaya" : "Waspada" })), history };
}
