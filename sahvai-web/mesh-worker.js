// SAHVAI 3D-4D: builds the 3D surfaces off the main thread so the page stays responsive.
// In: {id, grid: {nx, ny, nz, sp, hu}, mask, labels}   Out: {id, surf} | {id, error}
import { buildSurfaces } from './lib/mesh.js';
self.onmessage = ({ data }) => {
  try {
    const surf = buildSurfaces(data.grid, data.mask, data.labels);
    const bufs = Object.values(surf.layers).flatMap(l => [l.pos.buffer, l.nrm.buffer]);
    self.postMessage({ id: data.id, surf }, bufs);
  } catch (e) { self.postMessage({ id: data.id, error: e && e.message ? e.message : String(e) }); }
};
