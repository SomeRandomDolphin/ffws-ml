"use client";
import { useEffect, useState } from "react";
import type { LiveSnapshot } from "@/lib/surabaya";

export function useSurabaya() {
  const [data, setData] = useState<LiveSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController;
    async function refresh() {
      controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15000);
      try {
        const response = await fetch("/api/surabaya", { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error("Layanan data Surabaya belum terhubung.");
        const snapshot = await response.json() as LiveSnapshot;
        if (!Array.isArray(snapshot.stations)) throw new Error("Format data tidak valid.");
        if (!disposed) { setData(snapshot); setError(null); }
      } catch {
        if (!disposed) setError("Koneksi data Surabaya terputus. Pembacaan yang tampil adalah data terakhir.");
      } finally {
        clearTimeout(timeout);
        if (!disposed) timer = setTimeout(refresh, 30000);
      }
    }
    void refresh();
    return () => { disposed = true; clearTimeout(timer); controller?.abort(); };
  }, []);
  return { data, error };
}
