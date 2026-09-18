# Welang Water Monitor frontend

Next.js migration of the active Dash monitor. The current version is intentionally demo-first: `src/lib/demo-data.ts` provides deterministic station, history, forecast, rainfall, and five-parameter water-quality data while the FastAPI service remains available separately on port 8000.

## Run locally

```powershell
npm install
npm run dev
```

Open `http://localhost:3000`. The map starts at the East Java overview and drills into the Welang basin. Bundled layers include BIG administrative/basin boundaries, OSM rivers, DEMNAS-derived station sub-basins, rainfall symbols, water-quality rings, and the 15 monitoring stations.

If the primary basemap fails or takes more than 10 seconds to load, the main map switches to OpenTopoMap automatically. If both providers fail, use “Coba muat peta lagi” after the connection recovers; this refreshes tile sources without resetting the station selection or camera. Basemap recovery is independent of station markers and illustrative river/basin layers.

The 15 station markers and their HTML labels render independently of tile loading. Search selects and centers a station; clicking a marker opens its current simulated reading, detail link, and chart action. The legend starts open, while Overview and the history chart start closed. On mobile the side panel becomes a scrollable bottom panel. The timeline changes simulated readings without rebuilding the map or resetting its camera.

Station status colors and deterministic rainfall/quality values come from `src/lib/demo-data.ts`. Basin and administrative boundaries come from BIG, rivers from OpenStreetMap, and sub-basins from a reproducible DEMNAS/WhiteboxTools pipeline in `scripts/build_geodata.py`. Sensor values remain simulation data; no backend connection is required.

Validation: `npx tsc --noEmit --incremental false` and `npm run build`. Browser checks should cover all 15 markers on desktop/mobile, keyboard and search selection, chart controls, layer toggles, timeline updates, and failed tile requests.

## Frontend presentation

Shared palette, typography, controls, and station-detail styles live in `src/app/globals.css`; dashboard layout and map UI live in `src/app/map-dashboard.css`. Icons are local SVGs with no added package dependency. Closing panels remain mounted for their 220 ms transition; reduced-motion preferences disable animation. On mobile, opening the chart closes the information panel to preserve map space.

Station labels are placed around markers with collision detection, prioritizing the selected station. Labels without enough room are hidden; their markers remain focusable and searchable. After a user pans, zooms, or searches, panel resizing preserves the camera. “Tampilkan semua stasiun” restores framing of all stations.

The Dash fallback remains available with:

```powershell
python run_dashboard.py
```
