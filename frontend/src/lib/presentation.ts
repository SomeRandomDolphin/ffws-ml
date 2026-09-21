import type { StationStatus, WaterQualityStatus } from "./types";

export const statusColors: Record<StationStatus, string> = {
  Normal: "#32956b", Meningkat: "#d9ae32", Waspada: "#e48732", Bahaya: "#cb4949",
};
export const qualityColors: Record<WaterQualityStatus, string> = {
  Baik: "#28998d", Sedang: "#718093", Buruk: "#765477",
};
export const connectionColors = { live: "#19846d", delayed: "#9a7945" };
export const chartTheme = {
  water: "var(--color-action)",
  rain: "var(--color-rain)",
  rainFill: "var(--color-rain-fill)",
  grid: "var(--color-divider)",
  tick: { fontSize: 12, fill: "var(--color-text-muted)" },
  warning: "var(--color-warning)",
  danger: "var(--color-danger)",
  tooltip: {
    borderRadius: 8, border: "1px solid var(--color-divider)",
    background: "var(--color-surface)", color: "var(--color-text)",
    boxShadow: "var(--shadow)", padding: 12, fontSize: 12,
  },
};
