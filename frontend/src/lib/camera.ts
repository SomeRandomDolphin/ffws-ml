import type { RegionKey } from "./types";

export type RegionPreset = {
  key: RegionKey;
  label: string;
  asset: "regencies" | "basin";
  maxZoom: number;
  description: string;
  fallbackBounds: [number, number, number, number];
};

export const regionPresets: Record<RegionKey, RegionPreset> = {
  surabaya: {
    key: "surabaya", label: "Surabaya", asset: "regencies", maxZoom: 14,
    description: "Empat lokasi pemantauan Surabaya; posisi sensor belum terverifikasi.",
    fallbackBounds: [112.73, -7.31, 112.81, -7.26],
  },
  "jawa-timur": {
    key: "jawa-timur",
    label: "Jawa Timur",
    asset: "regencies",
    maxZoom: 8.8,
    description: "Konteks provinsi: batas kabupaten/kota dan sungai besar.",
    fallbackBounds: [110.8, -8.95, 114.8, -5.5],
  },
  welang: {
    key: "welang",
    label: "DAS Welang",
    asset: "basin",
    maxZoom: 12.5,
    description: "Fokus DAS: jaringan sungai, sub-DAS, dan 15 stasiun.",
    fallbackBounds: [112.587, -7.952, 112.932, -7.589],
  },
};

export const defaultRegion: RegionKey = "jawa-timur";

export const regionOrder: RegionKey[] = ["jawa-timur", "welang", "surabaya"];
