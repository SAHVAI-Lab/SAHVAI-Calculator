// Pre- and post-processing around the Hybrid 2D/3D UNet, ported from BrainHemoAI
// (github.com/BrainHemo/BrainHemoAI: demo.py, ml/comm_tools.py, ml/tri_hybrid_unet_wrapper.py,
// utils/data_orgnizer.py, utils/postprocess.py). Grid layout: z-major, x = column.
import { closing, removeSmallObjects, label2d } from './morph.js';

export const H = 352, W = 288, DEPTH = 18;
export const WINDOWS = [[0, 100], [10, 90], [20, 80], [30, 80], [40, 80]];
export const CLASSES = ['background', 'SAH', 'IPH', 'IVH'];

// ---------- alignment (the upstream platform lets the user set the rotation by hand) ----------
// Pick the in-plane rotation (degrees, skimage convention) that makes the brain outline most
// left-right symmetric. The network is not very sensitive to this (on the demo case the SAH
// volume changed by <1.5% between 0 and 25 degrees).
export function estimateAngle(mask, nx, ny, nz) {
  const P = nx * ny, areas = [];
  for (let z = 0; z < nz; z++) { let a = 0; for (let p = 0; p < P; p++) a += mask[z * P + p]; areas.push(a); }
  const amax = Math.max(...areas); if (!amax) return 0;
  // 4x downsampled masks of the larger slices; each is rotated (inverse mapping, nearest) and
  // scored by how much of it overlaps its own mirror image about its centre column
  const B = 4, bx = Math.ceil(nx / B), by = Math.ceil(ny / B), cx = nx / 2 / B, cy = ny / 2 / B, sl = [];
  for (let z = 0; z < nz; z++) {
    if (areas[z] < 0.3 * amax) continue;
    const m = new Uint8Array(bx * by);
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) if (mask[z * P + y * nx + x]) m[((y / B) | 0) * bx + ((x / B) | 0)]++;
    for (let i = 0; i < m.length; i++) m[i] = m[i] * 2 >= B * B ? 1 : 0;
    sl.push(m);
  }
  const rot = new Uint8Array(bx * by);
  const score = a => {
    const th = a * Math.PI / 180, c = Math.cos(th), s = Math.sin(th);
    let tot = 0;
    for (const m of sl) {
      let n = 0, sx = 0;
      for (let y = 0; y < by; y++) for (let x = 0; x < bx; x++) {
        const dx = x - cx, dy = y - cy, X = Math.round(c * dx - s * dy + cx), Y = Math.round(s * dx + c * dy + cy);
        const v = X >= 0 && Y >= 0 && X < bx && Y < by ? m[Y * bx + X] : 0;
        rot[y * bx + x] = v; if (v) { n++; sx += x; }
      }
      if (!n) continue;
      const mx2 = 2 * sx / n; let hit = 0;
      for (let y = 0; y < by; y++) for (let x = 0; x < bx; x++) {
        if (!rot[y * bx + x]) continue;
        const X = Math.round(mx2 - x); if (X >= 0 && X < bx && rot[y * bx + X]) hit++;
      }
      tot += hit / n;
    }
    return tot;
  };
  let best = 0, bestScore = -1;
  for (let a = -30; a <= 30; a += 1) { const s = score(a); if (s > bestScore) { bestScore = s; best = a; } }
  return best;
}

// skimage.transform.rotate(img, angle, center=(cx, cy)): out(p) = in(R(angle)(p - c) + c)
// order 1 (bilinear, cval 0) for images, order 0 (nearest) for masks.
export function rotateSlice(src, nx, ny, angle, cx, cy, nearest = false) {
  const out = new Float32Array(nx * ny);
  if (!angle) { out.set(src); return out; }
  const th = angle * Math.PI / 180, c = Math.cos(th), s = Math.sin(th);
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const dx = x - cx, dy = y - cy, xi = c * dx - s * dy + cx, yi = s * dx + c * dy + cy;
    if (nearest) {
      const X = Math.round(xi), Y = Math.round(yi);
      if (X >= 0 && Y >= 0 && X < nx && Y < ny) out[y * nx + x] = src[Y * nx + X];
      continue;
    }
    const x0 = Math.floor(xi), y0 = Math.floor(yi), fx = xi - x0, fy = yi - y0;
    const v = (X, Y) => (X >= 0 && Y >= 0 && X < nx && Y < ny) ? src[Y * nx + X] : 0;
    if (x0 < -1 || y0 < -1 || x0 >= nx || y0 >= ny) continue;
    out[y * nx + x] = (v(x0, y0) * (1 - fx) + v(x0 + 1, y0) * fx) * (1 - fy) + (v(x0, y0 + 1) * (1 - fx) + v(x0 + 1, y0 + 1) * fx) * fy;
  }
  return out;
}

// skimage.transform.resize(order=1, mode='reflect' -> ndimage 'mirror', grid_mode, anti-aliasing when shrinking)
function gauss1d(sigma) {
  const r = Math.floor(4 * sigma + 0.5), k = []; let s = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * i * i / (sigma * sigma)); k.push(v); s += v; }
  return k.map(v => v / s);
}
const mirror = (i, n) => { if (n === 1) return 0; const p = 2 * (n - 1); i = ((i % p) + p) % p; return i >= n ? p - i : i; };
function blur(src, h, w, sy, sx) {
  let a = src;
  if (sx > 0.05) { const k = gauss1d(sx), r = (k.length - 1) / 2, o = new Float32Array(h * w);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let v = 0; for (let j = 0; j < k.length; j++) v += k[j] * a[y * w + mirror(x + j - r, w)]; o[y * w + x] = v; } a = o; }
  if (sy > 0.05) { const k = gauss1d(sy), r = (k.length - 1) / 2, o = new Float32Array(h * w);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let v = 0; for (let j = 0; j < k.length; j++) v += k[j] * a[mirror(y + j - r, h) * w + x]; o[y * w + x] = v; } a = o; }
  return a;
}
export function resize2d(src, h, w, H2, W2) {
  const fy = h / H2, fx = w / W2;
  const a = blur(src, h, w, Math.max(0, (fy - 1) / 2), Math.max(0, (fx - 1) / 2));
  let mn = Infinity, mx = -Infinity; for (let i = 0; i < src.length; i++) { if (src[i] < mn) mn = src[i]; if (src[i] > mx) mx = src[i]; }
  const out = new Float32Array(H2 * W2);
  const xi0 = new Int32Array(W2), xi1 = new Int32Array(W2), xw = new Float32Array(W2);
  for (let X = 0; X < W2; X++) { const u = (X + 0.5) * fx - 0.5, f = Math.floor(u); xi0[X] = mirror(f, w); xi1[X] = mirror(f + 1, w); xw[X] = u - f; }
  for (let Y = 0; Y < H2; Y++) {
    const v = (Y + 0.5) * fy - 0.5, g = Math.floor(v), y0 = mirror(g, h), y1 = mirror(g + 1, h), wy = v - g;
    for (let X = 0; X < W2; X++) {
      const t = a[y0 * w + xi0[X]] * (1 - xw[X]) + a[y0 * w + xi1[X]] * xw[X];
      const b = a[y1 * w + xi0[X]] * (1 - xw[X]) + a[y1 * w + xi1[X]] * xw[X];
      let r = t * (1 - wy) + b * wy; if (r < mn) r = mn; else if (r > mx) r = mx;
      out[Y * W2 + X] = r;
    }
  }
  return out;
}

// ---------- the network input (tri_hybrid_unet_wrapper.predict) ----------
// hu: grid HU, mask: brain mask (after rectify). Returns everything needed to map back.
export function prepare(hu, mask, nx, ny, nz, angle) {
  const P = nx * ny, cx = nx / 2, cy = ny / 2;
  const brain = new Float32Array(P * nz);
  for (let z = 0; z < nz; z++) {
    const h = rotateSlice(hu.subarray(z * P, (z + 1) * P), nx, ny, angle, cx, cy);
    const m = rotateSlice(Float32Array.from(mask.subarray(z * P, (z + 1) * P)), nx, ny, angle, cx, cy, true);
    for (let p = 0; p < P; p++) if (m[p] > 0.5) brain[z * P + p] = h[p];
  }
  // crop_data: bounding box of non-zero voxels over all slices (end index exclusive of the max, as upstream)
  let top = ny, bot = -1, left = nx, right = -1;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) if (brain[z * P + y * nx + x] !== 0) {
    if (y < top) top = y; if (y > bot) bot = y; if (x < left) left = x; if (x > right) right = x;
  }
  if (bot < 0) throw new Error('no brain found in this scan (is it a head CT?)');
  const ch = bot - top, cw = right - left;
  const x2 = new Float32Array(nz * H * W), crop = new Float32Array(ch * cw);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) crop[y * cw + x] = brain[z * P + (top + y) * nx + left + x];
    x2.set(resize2d(crop, ch, cw, H, W), z * H * W);
  }
  // area_clip: first/last slice with >= 20% brain pixels (both inclusive here; upstream drops the last)
  const HW = H * W, fg = z => { let n = 0; for (let i = z * HW; i < (z + 1) * HW; i++) if (x2[i] !== 0) n++; return n; };
  let s = 0, e = nz - 1;
  while (s < nz && fg(s) < 0.2 * HW) s++;
  while (e >= 0 && fg(e) < 0.2 * HW) e--;
  if (s > e) throw new Error('too little brain tissue found to run the model');
  return { x2, s, e, top, left, ch, cw, angle, cx, cy };
}

// slices [s, e] in windows of 18: first window from s, later ones fill what is left (the last is
// aligned to e). Each slice takes its label from the first window that covers it.
export function windowsFor(s, e) {
  const n = e - s + 1, starts = [s];
  if (n > DEPTH) { for (let k = s + DEPTH; k + DEPTH <= e + 1; k += DEPTH) starts.push(k); if (starts[starts.length - 1] + DEPTH < e + 1) starts.push(e + 1 - DEPTH); }
  return starts;
}

// get_win_data for one window of up to 18 slices starting at z0 (zero padded at the end).
// Returns Float32 (18, 5, H, W) in slice-major order, the layout the ONNX parts use.
export function windowInput(x2, z0, e) {
  const HW = H * W, nreal = Math.min(DEPTH, e + 1 - z0), raw = new Float32Array(DEPTH * HW);
  raw.set(x2.subarray(z0 * HW, (z0 + nreal) * HW));
  let sum = 0, n = 0; for (let i = 0; i < raw.length; i++) if (raw[i] !== 0) { sum += raw[i]; n++; }
  const avg = sum / n; let ss = 0; for (let i = 0; i < raw.length; i++) if (raw[i] !== 0) { const d = raw[i] - avg; ss += d * d; }
  const std = Math.sqrt(ss / n);
  const x = new Float32Array(DEPTH * 5 * HW);
  for (let d = 0; d < DEPTH; d++) for (let i = 0; i < HW; i++) {
    const v = raw[d * HW + i], zv = (v - avg) / std;
    for (let c = 0; c < 5; c++) if (v > WINDOWS[c][0] && v < WINDOWS[c][1]) x[(d * 5 + c) * HW + i] = zv;
  }
  return { x, nreal };
}

// ---------- back to the grid ----------
// labels: Uint8 (nz, H, W) on the resized grid -> Uint8 (nz, ny, nx) on the working grid.
export function toGridLabels(labels, prep, nx, ny, nz) {
  const { top, left, ch, cw, angle, cx, cy } = prep, P = nx * ny, HW = H * W, out = new Uint8Array(P * nz);
  for (let z = 0; z < nz; z++) {
    const lab = labels.subarray(z * HW, (z + 1) * HW), rot = new Uint8Array(P);
    let any = false; for (let i = 0; i < HW; i++) if (lab[i]) { any = true; break; }
    if (!any) continue;
    for (let t = 1; t <= 3; t++) {        // resize_each_blood_type
      const b = new Float32Array(HW); let has = false;
      for (let i = 0; i < HW; i++) if (lab[i] === t) { b[i] = 1; has = true; }
      if (!has) continue;
      const r = resize2d(b, H, W, ch, cw);
      for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) if (r[y * cw + x] > 0.5) rot[(top + y) * nx + left + x] = t;
    }
    for (let t = 1; t <= 3; t++) {        // rotate_each_blood_type(-angle)
      const b = new Float32Array(P); let has = false;
      for (let p = 0; p < P; p++) if (rot[p] === t) { b[p] = 1; has = true; }
      if (!has) continue;
      const r = rotateSlice(b, nx, ny, -angle, cx, cy);
      for (let p = 0; p < P; p++) if (r[p] > 0.5) out[z * P + p] = t;
    }
  }
  return out;
}

// ---------- utils/postprocess.py ('connect' mode) ----------
export function postprocess(lab, nx, ny, nz) {
  const P = nx * ny, out = new Uint8Array(P * nz);
  for (let z = 0; z < nz; z++) {
    let l = lab.slice(z * P, (z + 1) * P);
    let any = false; for (let p = 0; p < P; p++) if (l[p]) { any = true; break; }
    if (!any) continue;
    // closing (disk 3): gaps filled with the provisional type 4
    const m = new Uint8Array(P); for (let p = 0; p < P; p++) m[p] = l[p] ? 1 : 0;
    const c = closing(m, nx, ny, 3), l2 = new Uint8Array(P);
    for (let p = 0; p < P; p++) if (c[p]) l2[p] = l[p] || 4;
    // remove_small_object (min 100 px, 4-connectivity)
    const m2 = new Uint8Array(P); for (let p = 0; p < P; p++) m2[p] = l2[p] ? 1 : 0;
    const keep = removeSmallObjects(m2, nx, ny, 100);
    for (let p = 0; p < P; p++) l[p] = keep[p] ? l2[p] : 0;
    // rectify_blood_type, run twice
    l = rectifySlice(rectifySlice(l, nx, ny), nx, ny);
    for (let p = 0; p < P; p++) if (l[p] === 4) l[p] = 1;   // gap pixels not absorbed by a neighbour: count as SAH
    out.set(l, z * P);
  }
  return out;
}

function rectifySlice(blood, w, h) {
  const P = w * h, fg = new Uint8Array(P); for (let p = 0; p < P; p++) fg[p] = blood[p] ? 1 : 0;
  const { lab, n } = label2d(fg, w, h, true), result = new Uint8Array(P);
  // pixels of each region
  const members = Array.from({ length: n + 1 }, () => []);
  for (let p = 0; p < P; p++) if (lab[p]) members[lab[p]].push(p);
  for (let r = 1; r <= n; r++) {
    const px = members[r], types = new Set(px.map(p => blood[p]));
    if (types.size < 2) { for (const p of px) result[p] = blood[p]; continue; }
    const temp = new Uint8Array(P); for (const p of px) temp[p] = blood[p];
    const rc = regionConnection(temp, px, w, h);
    for (const p of px) result[p] = rc[p];
  }
  return result;
}

function regionConnection(lm, px, w, h) {
  // label_region: components of each type (8-connectivity), numbered type by type
  const P = w * h, labels = new Int32Array(P), types = [...new Set(px.map(p => lm[p]))].sort((a, b) => a - b);
  const props = [{ area: 0, nb: new Set(), type: 0 }];
  for (const t of types) {
    const m = new Uint8Array(P); for (const p of px) if (lm[p] === t) m[p] = 1;
    const { lab, n } = label2d(m, w, h, true), base = props.length - 1;
    for (let k = 0; k < n; k++) props.push({ area: 0, nb: new Set(), type: t });
    for (const p of px) if (lab[p]) labels[p] = lab[p] + base;
  }
  for (const p of px) {
    const L = labels[p]; if (!L) continue;
    props[L].area++;
    const x = p % w, y = (p - x) / w;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue; const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= w || Y >= h) continue;
      const q = labels[Y * w + X]; if (q && q !== L) props[L].nb.add(q);
    }
  }
  const relabel = props.map((_, i) => i); let total = 0; for (const pr of props) total += pr.area;
  for (let i = 1; i < props.length; i++) {
    if (props[i].area / total < 0.25 || props[i].type === 4) {
      let best = -1, bestArea = 0;
      for (const j of props[i].nb) if (props[j].area > props[i].area && props[j].area > bestArea) { bestArea = props[j].area; best = j; }
      if (best !== -1) relabel[i] = best;
    }
  }
  const out = new Uint8Array(P);
  for (const p of px) { let l = relabel[labels[p]]; while (l !== relabel[l]) l = relabel[l]; out[p] = props[l].type; }
  return out;
}

export function volumes(lab, sp) {
  const v = sp[0] * sp[1] * sp[2] / 1000, n = [0, 0, 0, 0];
  for (let i = 0; i < lab.length; i++) n[lab[i]]++;
  return { SAH: n[1] * v, IPH: n[2] * v, IVH: n[3] * v, total: (n[1] + n[2] + n[3]) * v, voxels: n.slice(1), voxel_mL: v };
}
