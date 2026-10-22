// CLAIM aSAH multiclass segmentation (Kiewitz et al., Front Neurol 2024; github.com/claim-berlin/aSAH-multiclass-segmentation;
// weights CC BY-NC 4.0, Zenodo 10.5281/zenodo.12579491) in the browser: nnU-Net v2 "2d" configuration.
// Mirrors nnU-Net inference: in-plane resampling to the plans spacing (0.4453 mm; slice spacing kept),
// CTNormalization (clip to the 0.5/99.5 percentiles of the training foreground, z-score with its mean/std),
// slice-by-slice prediction with 512 x 512 tiles. "exact" = nnU-Net sliding window (step 0.5, Gaussian
// importance weighting); "fast" = the same sliding window restricted to the skull + 8 mm (one centred tile on an axis where that fits).
// Optional test-time mirroring over both in-plane axes (nnU-Net default). Argmax of the averaged logits.
// Orientation: our grid is L-P-S; nnU-Net sees the NIfTI storage order, which for the CLAIM data (dcm2niix-style
// L-A-S, as the BHSD demo cases) runs anterior along y. flipY (default) feeds the slices in that order.
export const CLAIM_CLASSES = ['background', 'SAH', 'IVH', 'Ventricles', 'ICH', 'Aneurysm', 'SDH'];
export const CLAIM_BLOOD = [1, 2, 4, 6];
export const CLAIM_PLANS = { spacing: 0.4453119933605194, patch: 512, clip: [-3, 88], mean: 32.46885299682617, std: 27.41019630432129 };
// thin slices are averaged to 5 mm (the model was trained on 5.5-6.5 mm scans); thicker slices are kept as they are
export const CLAIM_GRID = { spacing: sp => [CLAIM_PLANS.spacing, CLAIM_PLANS.spacing, sp[2] >= 4 ? sp[2] : 5], align: 'extent', interp: 'auto', label: 'CLAIM' };
const NC = 7, PS = CLAIM_PLANS.patch;

export function normalise(hu) {
  const [lo, hi] = CLAIM_PLANS.clip, { mean, std } = CLAIM_PLANS, x = new Float32Array(hu.length);
  for (let i = 0; i < hu.length; i++) { let v = hu[i]; v = v < lo ? lo : v > hi ? hi : v; x[i] = (v - mean) / std; }
  return x;
}

// nnU-Net v2 compute_steps_for_sliding_window
export function steps(size, tile, step = 0.5) {
  if (size <= tile) return [0];
  const n = Math.ceil((size - tile) / (tile * step)) + 1, actual = (size - tile) / (n - 1);
  return Array.from({ length: n }, (_, i) => Math.round(actual * i));
}

// nnU-Net v2 compute_gaussian (sigma = patch / 8, normalised to max 1, zeros replaced by the smallest value)
export function gaussian(n) {
  const s = n / 8, c = (n - 1) / 2, g1 = new Float64Array(n);
  // scipy gaussian_filter of a centred delta (constant mode): a sampled, truncated (4 sigma), renormalised kernel
  const r = Math.floor(4 * s + 0.5), k = []; let ks = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * i * i / (s * s)); k.push(v); ks += v; }
  const ci = n >> 1;   // nnU-Net puts the 1 at patch_size // 2
  for (let i = 0; i < n; i++) { const d = i - ci; g1[i] = Math.abs(d) <= r ? k[d + r] / ks : 0; }
  const g = new Float32Array(n * n); let mx = 0;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const v = g1[y] * g1[x]; g[y * n + x] = v; if (v > mx) mx = v; }
  let mn = Infinity;
  for (let i = 0; i < g.length; i++) { g[i] /= mx; if (g[i] > 0 && g[i] < mn) mn = g[i]; }
  for (let i = 0; i < g.length; i++) if (g[i] === 0) g[i] = mn;
  void c; return g;
}

function flip(src, n, fx, fy) {
  const out = new Float32Array(src.length), C = src.length / (n * n);
  for (let c = 0; c < C; c++) for (let y = 0; y < n; y++) {
    const yy = fy ? n - 1 - y : y;
    for (let x = 0; x < n; x++) out[(c * n + y) * n + x] = src[(c * n + yy) * n + (fx ? n - 1 - x : x)];
  }
  return out;
}

// call('claim2d', {x}) -> [{data: logits (1,7,512,512)}]; mode 'fast' | 'exact'; mirror bool
export async function runClaim(g, { call, mode = 'fast', mirror = false, flipY = true, log = () => {}, stage = () => {} } = {}) {
  const { nx, ny, nz } = g, P = nx * ny, t0 = performance.now();
  const row = y => flipY ? ny - 1 - y : y;   // network row -> grid row
  const x = normalise(g.hu), labels = new Uint8Array(P * nz);
  // head bounding box for the single-tile mode: the skull (HU > 300) plus an 8 mm margin, so the table and
  // head holder do not count (the labels outside the tile stay background)
  let x0 = nx, x1 = -1, y0 = ny, y1 = -1;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let xx = 0; xx < nx; xx++) if (g.hu[z * P + y * nx + xx] > 300) {
    const yr = row(y); if (xx < x0) x0 = xx; if (xx > x1) x1 = xx; if (yr < y0) y0 = yr; if (yr > y1) y1 = yr;
  }
  if (x1 < 0) { x0 = 0; x1 = nx - 1; y0 = 0; y1 = ny - 1; }
  const mg = Math.round(8 / g.sp[0]);
  x0 = Math.max(0, x0 - mg); x1 = Math.min(nx - 1, x1 + mg); y0 = Math.max(0, y0 - mg); y1 = Math.min(ny - 1, y1 + mg);
  const W = Math.max(nx, PS), H = Math.max(ny, PS);   // padded slice size (nnU-Net pads with 0 after normalisation, centred)
  const ox = (W - nx) >> 1, oy = (H - ny) >> 1;
  // fast: nnU-Net's sliding window restricted to the head box (one centred tile on an axis where the box fits)
  const span = (a0, a1, n) => {
    const L = a1 - a0 + 1;
    if (L <= PS) return [Math.max(0, Math.min(n - PS, Math.round((a0 + a1 + 1) / 2 - PS / 2)))];
    return steps(L, PS).map(t => Math.min(n - PS, a0 + t));
  };
  let tilesX, tilesY;
  if (mode === 'fast') { tilesX = span(x0 + ox, x1 + ox, W); tilesY = span(y0 + oy, y1 + oy, H); }
  else { tilesX = steps(W, PS); tilesY = steps(H, PS); }
  const gauss = mode === 'exact' && tilesX.length * tilesY.length > 1 ? gaussian(PS) : null;
  const flips = mirror ? [[0, 0], [1, 0], [0, 1], [1, 1]] : [[0, 0]];
  const nPass = nz * tilesX.length * tilesY.length * flips.length;
  log(`CLAIM: ${nx}x${ny}x${nz} at ${g.sp.map(s => s.toFixed(3)).join(' x ')} mm; ${mode} mode, ${tilesX.length * tilesY.length} tile(s) per slice, mirroring ${mirror ? 'on (x4)' : 'off'}: ${nPass} network passes`);
  const inp = new Float32Array(PS * PS);
  const acc = new Float32Array(NC * W * H), wsum = new Float32Array(W * H);
  let done = 0;
  for (let z = 0; z < nz; z++) {
    acc.fill(0); wsum.fill(0);
    for (const ty of tilesY) for (const tx of tilesX) {
      inp.fill(0);
      for (let y = 0; y < PS; y++) {
        const sy = ty + y - oy; if (sy < 0 || sy >= ny) continue;
        const rb = z * P + row(sy) * nx;
        for (let xx = 0; xx < PS; xx++) { const sx = tx + xx - ox; if (sx >= 0 && sx < nx) inp[y * PS + xx] = x[rb + sx]; }
      }
      const logit = new Float32Array(NC * PS * PS);
      for (const [fx, fy] of flips) {
        const xin = fx || fy ? flip(inp, PS, fx, fy) : inp;
        const out = await call('claim2d', { x: { data: xin, dims: [1, 1, PS, PS] } });
        const o = fx || fy ? flip(out[0].data, PS, fx, fy) : out[0].data;
        for (let i = 0; i < logit.length; i++) logit[i] += o[i];
        stage(`Segmenting with CLAIM nnU-Net · slice ${z + 1} of ${nz}`, ++done / nPass);
      }
      for (let c = 0; c < NC; c++) for (let y = 0; y < PS; y++) for (let xx = 0; xx < PS; xx++) {
        const w = gauss ? gauss[y * PS + xx] : 1, p = (ty + y) * W + tx + xx;
        acc[c * W * H + p] += logit[(c * PS + y) * PS + xx] / flips.length * w;
        if (c === 0) wsum[p] += w;
      }
    }
    for (let y = 0; y < ny; y++) for (let xx = 0; xx < nx; xx++) {
      const p = (y + oy) * W + xx + ox; if (!wsum[p]) continue;
      let best = 0, bv = acc[p];
      for (let c = 1; c < NC; c++) { const v = acc[c * W * H + p]; if (v > bv) { bv = v; best = c; } }
      labels[z * P + row(y) * nx + xx] = best;
    }
  }
  const vox = g.sp[0] * g.sp[1] * g.sp[2] / 1000, n = new Array(NC).fill(0);
  for (let i = 0; i < labels.length; i++) n[labels[i]]++;
  const volumes = {}; for (let c = 1; c < NC; c++) volumes[CLAIM_CLASSES[c]] = n[c] * vox;
  log(`CLAIM volumes: ${Object.entries(volumes).map(([k, v]) => `${k} ${v.toFixed(1)} mL`).join(', ')} (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
  return { labels, volumes, voxels: n, mode, mirror, flipY, passes: nPass };
}
