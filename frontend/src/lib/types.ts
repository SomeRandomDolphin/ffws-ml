export type StationStatus = "Normal" | "Meningkat" | "Waspada" | "Bahaya";
export type WaterQualityStatus = "Baik" | "Sedang" | "Buruk";
export type WaterQualityReading = { ph: number; temperatureC: number; dissolvedOxygenMgL: number; turbidityNtu: number; tdsMgL: number; status: WaterQualityStatus };
export type StationGeo = { name: string; latitude: number; longitude: number; confidence: string; source: string; description: string };
export type StationSnapshot = StationGeo & { valueM: number; delta3hM: number; status: StationStatus; alertM: number; dangerM: number; color: string; rainfall30mMm: number; rainfall24hMm: number; quality: WaterQualityReading };
export type DashboardSnapshot = { timestamp: string; leadHours: number; stations: StationSnapshot[]; forecasts: Array<{ leadHours: number; valueM: number; status: StationStatus }>; history: Array<{ time: string; valueM: number; rainMm: number }> };
export type LayerKey = "topology" | "regionalRivers" | "subdas" | "labels" | "basin" | "admin" | "rainfall" | "quality" | "relief";
export type RegionKey = "jawa-timur" | "welang";
