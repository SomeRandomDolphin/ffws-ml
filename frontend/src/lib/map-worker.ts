import { setWorkerUrl } from "maplibre-gl";

// An explicit same-origin module avoids the worker URL being rewritten to a
// Next.js bundle URL. Keep the worker version matched via prepare-map-worker.
export function configureMapWorker() {
  setWorkerUrl("/vendor/maplibre-gl-worker.mjs");
}
