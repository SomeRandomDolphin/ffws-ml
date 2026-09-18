import { copyFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const root = dirname(dirname(fileURLToPath(import.meta.url)));
const destination = join(root, "public", "vendor");
mkdirSync(destination, { recursive: true });
const packageRoot = dirname(require.resolve("maplibre-gl/package.json"));
copyFileSync(join(packageRoot, "dist", "maplibre-gl-worker.mjs"), join(destination, "maplibre-gl-worker.mjs"));
copyFileSync(join(packageRoot, "dist", "maplibre-gl-shared.mjs"), join(destination, "maplibre-gl-shared.mjs"));
copyFileSync(join(packageRoot, "LICENSE.txt"), join(destination, "maplibre-LICENSE.txt"));
