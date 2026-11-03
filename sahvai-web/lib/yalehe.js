// Yale hematoma-expansion model (Tran AT, Zeevi T, Haider SP, ... Payabvash S. npj Digit Med 2024;7:26;
// code and weights github.com/anhtrnyaleedu/HE): SegResNet hematoma segmentation, then a two-channel 3D
// DenseNet121 (CT + CT within 40 mm of the hematoma) with Monte-Carlo dropout, trained on ATACH-2 + Yale
// (n = 634) to predict expansion >= 6 mL. This file reproduces model512v9.ipynb's preprocessing.
//
// Volumes here are NIfTI-ordered (x fastest): index = x + X*(y + Y*z), with x -> patient Left, y -> Anterior,
// z -> Superior (the LAS layout dcm2niix writes). ONNX tensors are C-ordered [.., X, Y, Z].
import { label2d, fillHoles, sqdist, dilate, erode, removeSmallObjects } from './morph.js';

export const YALE = { band2: 41.5, seg: [512, 512, 48], roi: [256, 256, 32], cls: [128, 128, 96], mcPasses: 10 };
const roundHalfEven = v => { const f = Math.floor(v), d = v - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1); };

function minmax(a) { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } const r = hi - lo || 1; for (let i = 0; i < a.length; i++) a[i] = (a[i] - lo) / r; return a; }

// scipy.ndimage.zoom(order=1, grid_mode=False) along one axis of an x-fastest volume
function zoomAxis(src, dims, axis, n2) {
  const [X, Y, Z] = dims, n = dims[axis], out = new Float32Array(src.length / n * n2), d2 = dims.slice(); d2[axis] = n2;
  const st = [1, X, X * Y][axis], st2 = [1, d2[0], d2[0] * d2[1]][axis], scale = n2 > 1 ? (n - 1) / (n2 - 1) : 0;
  const w = new Float32Array(n2), i0 = new Int32Array(n2);
  for (let o = 0; o < n2; o++) { const u = o * scale, i = Math.min(n - 2, Math.floor(u)); i0[o] = Math.max(0, i); w[o] = n > 1 ? u - i0[o] : 0; }
  const [A, B] = axis === 0 ? [Y, Z] : axis === 1 ? [X, Z] : [X, Y];
  for (let a = 0; a < A; a++) for (let b = 0; b < B; b++) {
    const base = axis === 0 ? (a * X + b * X * Y) : axis === 1 ? (a + b * X * Y) : (a + b * X);
    const base2 = axis === 0 ? (a * d2[0] + b * d2[0] * d2[1]) : axis === 1 ? (a + b * d2[0] * d2[1]) : (a + b * d2[0]);
    for (let o = 0; o < n2; o++) { const i = i0[o], f = w[o]; out[base2 + o * st2] = src[base + i * st] * (1 - f) + (f ? src[base + (i + 1) * st] * f : 0); }
  }
  return { data: out, dims: d2 };
}
// resample3d(): each axis to round(n * spacing / target) voxels, linear
function resample(vol, dims, sp, target) {
  let v = { data: vol, dims: dims.slice() };
  // np.round rounds half to even
  for (let a = 0; a < 3; a++) { const n2 = roundHalfEven(dims[a] * sp[a] / target[a]); if (n2 !== v.dims[a]) v = zoomAxis(v.data, v.dims, a, n2); }
  return v;
}

// skull removal of model512v9.ipynb, slice by slice (img: brain-window volume scaled 0..1)
function removeSkull(img, dims) {
  const [X, Y, Z] = dims, P = X * Y, edge = roundHalfEven(Z / 20), out = new Float32Array(img.length);
  for (let z = 0; z < Z; z++) {
    if (z <= edge || z > Z - edge) continue;
    let m = new Uint8Array(P), any = 0; for (let p = 0; p < P; p++) if (img[z * P + p] > 0) { m[p] = 1; any = 1; }
    if (!any) continue;
    m = dilate(removeSmallObjects(erode(m, X, Y, 4), X, Y, 5000), X, Y, 4);
    m = fillHoles(m, X, Y);
    const L = label2d(m, X, Y, true); if (!L.n) continue;
    let best = 1; for (let i = 2; i <= L.n; i++) if (L.sizes[i] > L.sizes[best]) best = i;
    const keep = new Uint8Array(P); for (let p = 0; p < P; p++) keep[p] = L.lab[p] === best ? 1 : 0;
    const k2 = fillHoles(keep, X, Y);
    // cv2.drawContours(thickness=10) on the outer contour removes a band inside the edge; a Euclidean band of
    // distance <= sqrt(41.5) px to the outside matches it to 0.3% of brain voxels (BHSD SAH_002)
    const inv = new Uint8Array(P); for (let p = 0; p < P; p++) inv[p] = k2[p] ? 0 : 1;
    const d2 = sqdist(inv, X, Y);
    for (let p = 0; p < P; p++) if (k2[p] && d2[p] > YALE.band2) out[z * P + p] = img[z * P + p];
  }
  return minmax(out);
}

// MONAI ResizeWithPadOrCrop (symmetric pad, centre crop) of an x-fastest volume
function padCrop(src, dims, size) {
  const out = new Float32Array(size[0] * size[1] * size[2]), off = dims.map((n, a) => n < size[a] ? Math.floor((size[a] - n) / 2) : -(Math.floor(n / 2) - Math.floor(size[a] / 2)));
  for (let z = 0; z < size[2]; z++) { const zs = z - off[2]; if (zs < 0 || zs >= dims[2]) continue;
    for (let y = 0; y < size[1]; y++) { const ys = y - off[1]; if (ys < 0 || ys >= dims[1]) continue;
      for (let x = 0; x < size[0]; x++) { const xs = x - off[0]; if (xs < 0 || xs >= dims[0]) continue;
        out[x + size[0] * (y + size[1] * z)] = src[xs + dims[0] * (ys + dims[1] * zs)]; } } }
  return out;
}

// step 1: CT (HU, LAS x-fastest) -> the 512 x 512 x 48 skull-stripped brain window the segmentation sees
export function prepBrain(hu, dims, sp) {
  let w = new Float32Array(hu.length);
  for (let i = 0; i < hu.length; i++) { const v = hu[i]; w[i] = v < 0 || v > 200 ? 0 : v > 80 ? 80 : v; }
  minmax(w);
  let d = dims.slice(), nsp = sp.slice();
  if (dims[2] > 48) { const r = resample(w, dims, sp, [sp[0], sp[1], 5]); w = r.data; d = r.dims; nsp = [sp[0], sp[1], 5]; }
  w = removeSkull(w, d);
  return { brain: padCrop(w, d, YALE.seg), spacing: nsp, preDims: d };
}

// step 2: SegResNet by sliding window (256 x 256 x 32, 50% overlap, averaged logits) -> mask (x-fastest)
export async function segment(brain, run, progress = () => {}) {
  const [X, Y, Z] = YALE.seg, [rx, ry, rz] = YALE.roi, x = Float32Array.from(brain); minmax(x);
  const starts = (n, r) => { const s = []; const step = r / 2, k = Math.ceil((n - r) / step) + 1; for (let i = 0; i < k; i++) s.push(Math.min(i * step, n - r)); return [...new Set(s)]; };
  const sx = starts(X, rx), sy = starts(Y, ry), sz = starts(Z, rz), acc = new Float32Array(X * Y * Z), cnt = new Uint8Array(X * Y * Z);
  const total = sx.length * sy.length * sz.length; let done = 0;
  const win = new Float32Array(rx * ry * rz);
  for (const z0 of sz) for (const y0 of sy) for (const x0 of sx) {
    for (let i = 0; i < rx; i++) for (let j = 0; j < ry; j++) for (let k = 0; k < rz; k++) win[(i * ry + j) * rz + k] = x[(x0 + i) + X * ((y0 + j) + Y * (z0 + k))];
    const out = await run(win);
    for (let i = 0; i < rx; i++) for (let j = 0; j < ry; j++) for (let k = 0; k < rz; k++) { const q = (x0 + i) + X * ((y0 + j) + Y * (z0 + k)); acc[q] += out[(i * ry + j) * rz + k]; cnt[q]++; }
    progress(++done / total);
  }
  const mask = new Uint8Array(X * Y * Z); for (let q = 0; q < mask.length; q++) mask[q] = acc[q] / cnt[q] > 0 ? 1 : 0;
  return mask;
}

// step 3: the classifier's two-channel input, C-ordered [2, 128, 128, 96]
export function clsInput(brain, mask, sp) {
  const [X, Y, Z] = YALE.seg, P = X * Y, m = new Float32Array(mask.length);
  for (let z = 0; z < Z; z++) { let s = 0; for (let p = 0; p < P; p++) s += mask[z * P + p]; if (s >= 50) for (let p = 0; p < P; p++) m[z * P + p] = mask[z * P + p]; }
  const S = resample(m, YALE.seg, sp, [1, 1, 1]), B = resample(Float32Array.from(brain), YALE.seg, sp, [1, 1, 1]);
  const [x1, y1, z1] = S.dims, Q = x1 * y1; for (let i = 0; i < S.data.length; i++) S.data[i] = S.data[i] > 0 ? 1 : 0;
  // selectGroupSlice: hematoma slices +-16 (at most 96), placed at the start of a 96-slice block
  let s1 = 0, s2 = z1 - 1; const has = z => { for (let p = 0; p < Q; p++) if (S.data[z * Q + p]) return true; return false; };
  for (s1 = 0; s1 < z1; s1++) if (has(s1)) break; if (s1 === z1) s1 = z1 - 1;
  for (s2 = z1 - 1; s2 > 0; s2--) if (has(s2)) break;
  const img = new Float32Array(Q * 96), seg = new Float32Array(Q * 96);
  if (s1 < s2) {
    s1 = Math.max(0, s1 - 16); s2 = Math.min(z1 - 1, s2 + 16); if (s2 - s1 >= 96) s2 = s1 + 96;
    img.set(B.data.subarray(s1 * Q, s2 * Q), 0); seg.set(S.data.subarray(s1 * Q, s2 * Q), 0);
  } else { const c = roundHalfEven(z1 / 2); img.set(B.data.subarray((c - 32) * Q, (c + 32) * Q), 0); }
  minmax(img);
  // getDilation: CT within a 40-voxel (40 mm) disk of the hematoma, per slice
  const dil = new Float32Array(Q * 96); let anySeg = false;
  for (let z = 0; z < 96; z++) {
    const sl = seg.subarray(z * Q, (z + 1) * Q); let any = false; for (const v of sl) if (v) { any = true; break; }
    if (!any) continue; anySeg = true;
    const dm = dilate(Uint8Array.from(sl, v => (v ? 1 : 0)), x1, y1, 40);
    for (let p = 0; p < Q; p++) if (dm[p]) dil[z * Q + p] = img[z * Q + p];
  }
  if (anySeg) minmax(dil);
  // MONAI Resize([192, 192, 96], mode 'area' = adaptive average pooling), CenterSpatialCrop(128, 128, 96), ScaleIntensity
  const out = new Float32Array(2 * 128 * 128 * 96), off = 32;
  const bins = (n, o) => Array.from({ length: o }, (_, i) => [Math.floor(i * n / o), Math.ceil((i + 1) * n / o)]);
  const bx = bins(x1, 192), by = bins(y1, 192);
  for (let c = 0; c < 2; c++) { const src = c ? dil : img;
    for (let i = 0; i < 128; i++) { const [a0, a1] = bx[i + off];
      for (let j = 0; j < 128; j++) { const [b0, b1] = by[j + off];
        for (let z = 0; z < 96; z++) { let s = 0; for (let a = a0; a < a1; a++) for (let b = b0; b < b1; b++) s += src[a + x1 * (b + y1 * z)];
          out[((c * 128 + i) * 128 + j) * 96 + z] = s / ((a1 - a0) * (b1 - b0)); } } } }
  return minmax(out);
}

// the whole model from the SAHVAI working grid (x -> Left, y -> Posterior, z -> Superior; idx = x + nx*(y + ny*z))
export function gridToLAS(g) {
  const { nx, ny, nz } = g, out = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) out.set(g.hu.subarray(z * nx * ny + (ny - 1 - y) * nx, z * nx * ny + (ny - y) * nx), z * nx * ny + y * nx);
  return { hu: out, dims: [nx, ny, nz], sp: g.sp.slice() };
}
// the 512 x 512 x 48 mask back onto the SAHVAI grid (for display)
export function maskToGrid(mask, preDims, g) {
  const [X, Y, Z] = YALE.seg, [px, py, pz] = preDims, out = new Uint8Array(g.nx * g.ny * g.nz);
  const off = [px, py, pz].map((n, a) => n < YALE.seg[a] ? Math.floor((YALE.seg[a] - n) / 2) : -(Math.floor(n / 2) - Math.floor(YALE.seg[a] / 2)));
  const zr = pz === g.nz ? 1 : (g.nz - 1) / Math.max(1, pz - 1);
  for (let z = 0; z < g.nz; z++) { const zp = Math.round(z / zr) + off[2]; if (zp < 0 || zp >= Z) continue;
    for (let y = 0; y < g.ny; y++) { const yp = (g.ny - 1 - y) + off[1]; if (yp < 0 || yp >= Y) continue;
      for (let x = 0; x < g.nx; x++) { const xp = x + off[0]; if (xp < 0 || xp >= X) continue; if (mask[xp + X * (yp + Y * zp)]) out[z * g.nx * g.ny + y * g.nx + x] = 1; } } }
  return out;
}
