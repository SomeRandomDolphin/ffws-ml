export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const base = process.env.SURABAYA_API_URL ?? "http://127.0.0.1:8000";
    const response = await fetch(base + "/surabaya/snapshot", {
      cache: "no-store", signal: AbortSignal.timeout(12000),
    });
    if (!response.ok) return Response.json({ error: "Layanan Surabaya belum tersedia." }, { status: 503 });
    return Response.json(await response.json(), { headers: { "Cache-Control": "no-store" } });
  } catch {
    return Response.json({ error: "Backend Surabaya belum terhubung." }, { status: 503 });
  }
}
