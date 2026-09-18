import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { getDemoSnapshot } from "@/lib/demo-data";
import StationDetail from "@/components/station-detail";

export async function generateMetadata({ params }: { params: Promise<{ name: string }> }): Promise<Metadata> {
  const { name } = await params;
  const stationName = decodeURIComponent(name);
  return { title: `${stationName} · ITS Water Dashboard`, description: `Detail stasiun pemantauan muka air ${stationName} di DAS Welang, Pasuruan.` };
}

export default async function StationPage({ params }: { params: Promise<{ name: string }> }) {
  const { name } = await params;
  const stationName = decodeURIComponent(name);
  const snapshot = getDemoSnapshot(0);
  const station = snapshot.stations.find((item) => item.name === stationName);
  if (!station) notFound();
  const forecasts = Array.from({ length: 5 }, (_, index) => {
    const leadHours = index + 1;
    const valueM = station.valueM + (station.delta3hM / 3) * leadHours;
    const status = valueM >= station.dangerM ? "Bahaya" : valueM >= station.alertM ? "Waspada" : valueM > station.valueM ? "Meningkat" : "Normal";
    return { leadHours, valueM, status } as const;
  });
  return <StationDetail station={station} forecasts={forecasts} timestamp="2026-02-18T05:00:00Z" />;
}
