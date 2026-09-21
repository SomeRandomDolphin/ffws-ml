export const surabayaNames = ["Hang Tuah", "Kalibokor", "Rumah Pompa Pucang", "Sebrang Perpustaan Jawa Timur"] as const;
export type LiveForecast = { method: "urban_file" | "persistence" | "unavailable"; issuedAt: string | null; reason: string; points: { leadHours: number; time: string; valueCm: number }[] };
export type LiveSensor = { id: string; valueCm: number | null; history: { time: string; valueCm: number | null }[]; forecast: LiveForecast };
export type LiveStation = {
  id: string; name: string; latitude: number | null; longitude: number | null;
  coordinateSource: string | null; coordinateVerified: boolean;
  observedAt: string | null; lastSyncAt: string | null;
  state: "live" | "stale" | "cached" | "unavailable" | "future"; error: string | null;
  sensors: LiveSensor[];
  rainfall: { observedAt: string; values: Record<string, number | null>; error: string | null } | null;
  telemetry: { observedAt: string | null; windir?: number | null; windavg?: number | null; windmax?: number | null; lightlevel?: number | null; flowrate?: number | null; flowvelocity?: number | null; rpm?: number | null } | null;
};
export type LiveSnapshot = { generatedAt: string; unit: "cm"; verification: "unverified"; pollSeconds: number; stations: LiveStation[] };

// Keep the configured facility locations visible while the live service is unavailable.
export const surabayaStationFallback: LiveStation[] = [
  { id: "lokasi_1_hang_tuah", name: "Hang Tuah", latitude: -7.2935747, longitude: 112.7931574, coordinateSource: "https://www.google.com/maps/place/Universitas+Hang+Tuah/@-7.2935747,112.7931574,17z", coordinateVerified: false, observedAt: null, lastSyncAt: null, state: "unavailable", error: "Pembacaan live belum tersedia.", sensors: [], rainfall: null, telemetry: null },
  { id: "lokasi_2_kalibokor", name: "Kalibokor", latitude: -7.2837221, longitude: 112.7500171, coordinateSource: "https://www.google.com/maps/search/Jalan+Kalibokor+Surabaya/@-7.2837221,112.7500171,17z", coordinateVerified: false, observedAt: null, lastSyncAt: null, state: "unavailable", error: "Pembacaan live belum tersedia.", sensors: [], rainfall: null, telemetry: null },
  { id: "lokasi_3_rumah_pompa_pucang", name: "Rumah Pompa Pucang", latitude: -7.2818936, longitude: 112.7560149, coordinateSource: "https://www.google.com/maps/search/Pucang+Anom+Timur+Surabaya/@-7.2818936,112.7560149,17z", coordinateVerified: false, observedAt: null, lastSyncAt: null, state: "unavailable", error: "Pembacaan live belum tersedia.", sensors: [], rainfall: null, telemetry: null },
  { id: "lokasi_4_pln_menur", name: "Sebrang Perpustaan Jawa Timur", latitude: -7.289108, longitude: 112.768395, coordinateSource: "https://www.google.com/maps/search/?api=1&query=-7.289108,112.768395", coordinateVerified: false, observedAt: null, lastSyncAt: null, state: "unavailable", error: "Pembacaan live belum tersedia.", sensors: [], rainfall: null, telemetry: null },
];
export const liveStateLabel: Record<LiveStation["state"], string> = { live: "Terkini", stale: "Data terlambat", cached: "Data tersimpan", unavailable: "Belum ada data", future: "Waktu tidak valid" };
export function formatSensorLabel(id: string, stationName?: string) {
  const isPucang = stationName ? stationName.toLowerCase().includes("pucang") : false;
  if (isPucang) {
    if (id.toLowerCase() === "distance1" || id.toLowerCase() === "distance_1") return "Pompa 1";
    if (id.toLowerCase() === "distance2" || id.toLowerCase() === "distance_2") return "Pompa 2";
    if (id.toLowerCase() === "distance") return "Pompa 1";
  }
  return id;
}

export function liveTime(value: string | null) {
  return value ? new Intl.DateTimeFormat("id-ID", { timeZone: "Asia/Jakarta", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }).format(new Date(value)) + " WIB" : "—";
}
