// BLAST-CT (Monteiro, Newcombe et al., Lancet Digit Health 2020; github.com/biomedia-mira/blast-ct, Apache-2.0)
// in the browser. Mirrors blast_ct inference: reorient, resample to 1 mm isotropic (SimpleITK linear, corner
// aligned), window [-15, 100] HU to [-1, 1], tile the volume with non-overlapping 60^3 output blocks (the
// network sees a 140^3 edge-padded input around each; output is identical to the tool's 110^3 -> 30^3 tiles),
// softmax, average over the chosen models, argmax.
export const BLAST_CLASSES = ['background', 'IPH', 'EAH', 'Oedema', 'IVH'];
export const BLAST_NAMES = { 1: 'Intraparenchymal haemorrhage', 2: 'Extra-axial haemorrhage', 3: 'Perilesional oedema', 4: 'Intraventricular haemorrhage' };
export const TILE = 60, TILE_IN = 140, NCLS = 5;
export const BLAST_SPACING = () => [1, 1, 1];
export const BLAST_GRID = { spacing: BLAST_SPACING, align: 'corner', interp: 'linear', label: 'BLAST-CT' };

export function normalise(hu) {
  const lo = -15, hi = 100, x = new Float32Array(hu.length);
  for (let i = 0; i < hu.length; i++) { let v = hu[i]; v = v <= lo ? lo : v >= hi ? hi : v; x[i] = (2 * v - lo - hi) / (hi - lo); }
  return x;
}

// call(name, {x: {data, dims}}) -> [{data: probs (1,5,60,60,60)}]
export async function runBlast(g, { call, models = ['blast_m1'], log = () => {}, stage = () => {} } = {}) {
  const { nx, ny, nz } = g, S = [nz, ny, nx], t0 = performance.now();
  const x = normalise(g.hu);
  const centres = S.map(s => Array.from({ length: Math.floor(s / TILE) + 1 }, (_, i) => TILE * i + (TILE >> 1)));
  const tiles = [];
  for (const cz of centres[0]) for (const cy of centres[1]) for (const cx of centres[2]) tiles.push([cz, cy, cx]);
  log(`BLAST-CT: ${nx}x${ny}x${nz} at 1 mm, ${tiles.length} tiles of ${TILE}^3 (input ${TILE_IN}^3), ${models.length} model${models.length > 1 ? 's' : ''}`);
  const labels = new Uint8Array(nx * ny * nz);
  const inp = new Float32Array(TILE_IN ** 3), P = TILE ** 3, sum = new Float32Array(NCLS * P);
  const clamp = (v, n) => v < 0 ? 0 : v >= n ? n - 1 : v;
  const half = TILE_IN >> 1;
  for (let k = 0; k < tiles.length; k++) {
    const [cz, cy, cx] = tiles[k];
    // edge-padded input block (np.pad mode='edge' == clamped indices)
    const z0 = cz - half, y0 = cy - half, x0 = cx - half;
    const xi = new Int32Array(TILE_IN); for (let i = 0; i < TILE_IN; i++) xi[i] = clamp(x0 + i, nx);
    let o = 0;
    for (let dz = 0; dz < TILE_IN; dz++) {
      const zb = clamp(z0 + dz, nz) * ny * nx;
      for (let dy = 0; dy < TILE_IN; dy++) {
        const b = zb + clamp(y0 + dy, ny) * nx;
        for (let dx = 0; dx < TILE_IN; dx++) inp[o++] = x[b + xi[dx]];
      }
    }
    sum.fill(0);
    for (const m of models) {
      const out = await call(m, { x: { data: inp, dims: [1, 1, TILE_IN, TILE_IN, TILE_IN] } });
      const pr = out[0].data; for (let i = 0; i < sum.length; i++) sum[i] += pr[i];
    }
    // argmax into the output block (tiles start at TILE * i and do not overlap)
    const oz = cz - (TILE >> 1), oy = cy - (TILE >> 1), ox = cx - (TILE >> 1);
    for (let dz = 0; dz < TILE; dz++) {
      const z = oz + dz; if (z >= nz) break;
      for (let dy = 0; dy < TILE; dy++) {
        const y = oy + dy; if (y >= ny) break;
        for (let dx = 0; dx < TILE; dx++) {
          const xx = ox + dx; if (xx >= nx) break;
          const p = (dz * TILE + dy) * TILE + dx;
          let best = 0, bv = sum[p];
          for (let c = 1; c < NCLS; c++) { const v = sum[c * P + p]; if (v > bv) { bv = v; best = c; } }
          labels[(z * ny + y) * nx + xx] = best;
        }
      }
    }
    stage(`Segmenting with BLAST-CT · tile ${k + 1} of ${tiles.length}`, (k + 1) / tiles.length);
  }
  const vox = g.sp[0] * g.sp[1] * g.sp[2] / 1000, n = new Array(NCLS).fill(0);
  for (let i = 0; i < labels.length; i++) n[labels[i]]++;
  const volumes = {}; for (let c = 1; c < NCLS; c++) volumes[BLAST_CLASSES[c]] = n[c] * vox;
  log(`BLAST-CT volumes: ${Object.entries(volumes).map(([k, v]) => `${k} ${v.toFixed(1)} mL`).join(', ')} (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
  return { labels, volumes, voxels: n, tiles: tiles.length };
}
