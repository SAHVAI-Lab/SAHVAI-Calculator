// SAH Swin UNETR (García-García S, Cepeda S, Müller D, et al. Mortality prediction of patients with subarachnoid
// hemorrhage using a deep learning model based on an initial brain CT scan. Brain Sci 2024;14:10; and World Neurosurg
// 2024, doi:10.1016/j.wneu.2024.07.216; github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction;
// Hospital Universitario Río Hortega, Valladolid, and University of Augsburg; academic non-commercial licence) in the browser.
//
// The original pipeline (SAH_mortality_prediction.py + inference_2.py): brain extraction (FSL, as DeepBleed) ->
// ANTs registration (affine + SyN) of the brain to a 1 mm MNI CT template (193 x 229 x 193) -> MONAI Spacingd to
// 1.5 x 1.5 x 2 mm -> z-score over non-zero voxels -> crop to the voxels above the mean -> sliding window
// (128 x 128 x 64, overlap 0.25, constant blending) with Swin UNETR (feature size 48, 2 classes) -> inverted back
// to the template (bilinear logits) -> argmax.
// Here: brain extraction is SAHVAI's; registration is a 9-parameter (rotation, translation, per-axis scale) mutual-
// information fit to the template (in tests the network's output changed little between rigid, affine and SyN
// registration); the network is the original, converted to ONNX with the weights unchanged; the sliding window,
// normalisation and crop reproduce MONAI's; the result is mapped back to the scan, where volumes are measured
// (the original reports in template space, which scales volumes by the size of the head).
// A scan that is already in the template space (the RHUH dataset's preprocessed scans) skips the registration.

export const ROI = [128, 128, 64];
const PIX = [1.5, 1.5, 2.0], OVERLAP = 0.25;

const mul = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2], M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2], M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]];
const mm = (A, B) => [0, 1, 2].map(r => [0, 1, 2].map(c => A[r][0] * B[0][c] + A[r][1] * B[1][c] + A[r][2] * B[2][c]));
function inv3(M) {
  const [a, b, c] = M[0], [d, e, f] = M[1], [g, h, i] = M[2], A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det], [B / det, (a * i - c * g) / det, -(a * f - c * d) / det], [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
}
function rot([ax, ay, az]) {   // Rz Ry Rx
  const cx = Math.cos(ax), sx = Math.sin(ax), cy = Math.cos(ay), sy = Math.sin(ay), cz = Math.cos(az), sz = Math.sin(az);
  return [[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], [-sy, cy * sx, cy * cx]];
}
function geom(g) {
  const A = g.affine ? [0, 1, 2].map(r => g.affine[r].slice(0, 3)) : [[g.sp[0], 0, 0], [0, g.sp[1], 0], [0, 0, g.sp[2]]];
  const o = g.affine ? [g.affine[0][3], g.affine[1][3], g.affine[2][3]] : [0, 0, 0];
  return { A, o, Ai: inv3(A) };
}
// trilinear sample of a z-major grid (x fastest) at fractional index; outside -> 0
function tri(vol, nx, ny, nz, x, y, z) {
  if (x < 0 || y < 0 || z < 0 || x > nx - 1 || y > ny - 1 || z > nz - 1) return 0;
  const x0 = Math.floor(x), y0 = Math.floor(y), z0 = Math.floor(z), fx = x - x0, fy = y - y0, fz = z - z0;
  const x1 = Math.min(x0 + 1, nx - 1), y1 = Math.min(y0 + 1, ny - 1), z1 = Math.min(z0 + 1, nz - 1), P = nx * ny;
  const a = z0 * P, b = z1 * P;
  const c00 = vol[a + y0 * nx + x0] * (1 - fx) + vol[a + y0 * nx + x1] * fx, c10 = vol[a + y1 * nx + x0] * (1 - fx) + vol[a + y1 * nx + x1] * fx;
  const c01 = vol[b + y0 * nx + x0] * (1 - fx) + vol[b + y0 * nx + x1] * fx, c11 = vol[b + y1 * nx + x0] * (1 - fx) + vol[b + y1 * nx + x1] * fx;
  return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
}
function blur(src, nx, ny, nz, sig) {
  let a = Float32Array.from(src), b = new Float32Array(src.length);
  const dims = [nx, ny, nz], strides = [1, nx, nx * ny];
  for (let ax = 0; ax < 3; ax++) {
    const s = sig[ax]; if (!(s > 0.3)) continue;
    const r = Math.ceil(3 * s), k = []; let ks = 0;
    for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * i * i / (s * s)); k.push(v); ks += v; }
    for (let i = 0; i < k.length; i++) k[i] /= ks;
    const n = dims[ax], st = strides[ax];
    for (let idx = 0; idx < a.length; idx++) {
      const c = Math.floor(idx / st) % n; let v = 0;
      for (let d = -r; d <= r; d++) { const cc = c + d; if (cc >= 0 && cc < n) v += k[d + r] * a[idx + d * st]; }
      b[idx] = v;
    }
    [a, b] = [b, a];
  }
  return a;
}
function mi(a, b, n, bins = 48) {
  let la = Infinity, ha = -Infinity, lb = Infinity, hb = -Infinity;
  for (let i = 0; i < n; i++) { const x = a[i], y = b[i]; if (x < la) la = x; if (x > ha) ha = x; if (y < lb) lb = y; if (y > hb) hb = y; }
  const h = new Float64Array(bins * bins), sa = (bins - 1e-6) / (ha - la + 1e-6), sb = (bins - 1e-6) / (hb - lb + 1e-6);
  for (let i = 0; i < n; i++) h[((a[i] - la) * sa | 0) * bins + ((b[i] - lb) * sb | 0)]++;
  const px = new Float64Array(bins), py = new Float64Array(bins);
  for (let i = 0; i < bins; i++) for (let j = 0; j < bins; j++) { const v = h[i * bins + j]; px[i] += v; py[j] += v; }
  let s = 0;
  for (let i = 0; i < bins; i++) for (let j = 0; j < bins; j++) { const v = h[i * bins + j]; if (v) s += v / n * Math.log(v * n / (px[i] * py[j])); }
  return s;
}

// template: niftiVolume() of models/sahswin/ct_template2mni_2mm.nii.gz (the repository's 1 mm template, smoothed and
// subsampled to 2 mm for the registration only) -> {T (x fastest), n, A (LPS, columns = axes * spacing), o (LPS), D (unit axes)}
export async function loadSwinTemplate(vol) {
  const [nx, ny, nz] = vol.dims, T = new Float32Array(nx * ny * nz);
  for (let k = 0; k < nz; k++) T.set(await vol.slice(k), k * nx * ny);
  const A = [0, 1, 2].map(r => [0, 1, 2].map(c => vol.dir[c][r] * vol.spacing[c]));
  const D = [0, 1, 2].map(r => [0, 1, 2].map(c => vol.dir[c][r]));
  return { T, n: [nx, ny, nz], A, o: vol.origin || [0, 0, 0], D };
}
// the 1 mm template the network works in: 193 x 229 x 193, same origin and axes as the 2 mm file
const T1 = [193, 229, 193];
export const NET = [0, 1, 2].map(a => Math.ceil(((T1[a] - 1) * 1) / PIX[a] - 1e-9) + 1);   // MONAI Spacingd: 129 x 153 x 97

// For each network-grid index along each axis, the 1 mm template voxels MONAI's Spacingd (torch grid_sample, bilinear,
// float64 grid, border padding) gives a non-zero weight: code = 2 * d + f with first voxel floor(n * spacing) - d and a
// second voxel (+1) when f = 1. Recorded from MONAI 1.6 for this geometry (template 193 x 229 x 193 -> 129 x 153 x 97).
const NZ_CODES = { i: '113111311131013101310131013101310101010101010101010101010101010101110131011101310111010101010101010101010101010101010101010101010', j: '111111111131110111113101310111111131313101310101013131010131010101310131010101011101010101010101110101110101010101010111013101010101013101010101310101310', k: '3313313313310310010310310133133100103003003003000010030010030030000001000000300000300001300001300' };
function nzSets(code, step, n) {
  return Array.from(code, (ch, m) => { const c = +ch, l = Math.floor(m * step) - (c >> 1), s = [Math.min(n - 1, Math.max(0, l))];
    if (c & 1) { const l2 = Math.min(n - 1, Math.max(0, l + 1)); if (l2 !== s[0]) s.push(l2); } return s; });
}
// network-grid voxels MONAI would count as non-zero, given nz1(a, b, c) = is the 1 mm template-space image non-zero there
function nonZeroNet(nz1) {
  const [A, B, C] = T1, [X0, Y0, Z0] = NET;
  const si = nzSets(NZ_CODES.i, PIX[0], A), sj = nzSets(NZ_CODES.j, PIX[1], B), sk = nzSets(NZ_CODES.k, PIX[2], C);
  const full = new Uint8Array(A * B * C);   // index (a*B + b)*C + c
  for (let a = 0; a < A; a++) for (let b = 0; b < B; b++) for (let c = 0; c < C; c++) full[(a * B + b) * C + c] = nz1(a, b, c) ? 1 : 0;
  const s1 = new Uint8Array(X0 * B * C);
  for (let i = 0; i < X0; i++) for (const a of si[i]) for (let r = 0; r < B * C; r++) s1[i * B * C + r] |= full[a * B * C + r];
  const s2 = new Uint8Array(X0 * Y0 * C);
  for (let i = 0; i < X0; i++) for (let j = 0; j < Y0; j++) for (const b of sj[j]) for (let c = 0; c < C; c++) s2[(i * Y0 + j) * C + c] |= s1[(i * B + b) * C + c];
  const out = new Uint8Array(X0 * Y0 * Z0);
  for (let i = 0; i < X0; i++) for (let j = 0; j < Y0; j++) for (let k = 0; k < Z0; k++) { let v = 0; for (const c of sk[k]) v |= s2[(i * Y0 + j) * C + c]; out[(i * Y0 + j) * Z0 + k] = v; }
  return out;
}

// is this grid already the 1 mm template space (the RHUH preprocessed scans)?
export function inTemplateSpace(gi, tpl) {
  if (!gi.affine) return false;
  const G = geom(gi), corners = [[0, 0, 0], [gi.nx - 1, gi.ny - 1, gi.nz - 1]].map(i => mul(G.A, i).map((v, a) => v + G.o[a]));
  const t = [[0, 0, 0], T1.map(n => n - 1)].map(i => mul(tpl.D, i).map((v, a) => v + tpl.o[a]));
  const lo = (p, q) => p.map((v, a) => Math.min(v, q[a])), hi = (p, q) => p.map((v, a) => Math.max(v, q[a]));
  const d = Math.max(...lo(...corners).map((v, a) => Math.abs(v - lo(...t)[a])), ...hi(...corners).map((v, a) => Math.abs(v - hi(...t)[a])));
  return d < 1.01 && Math.abs(gi.sp[0] - 1) < 0.02 && Math.abs(gi.sp[2] - 1) < 0.02;
}

// MONAI sliding-window starts along one axis (image already padded to at least the ROI)
export function starts(size, roi, overlap = OVERLAP) {
  const iv = size === roi ? roi : Math.max(1, Math.floor(roi * (1 - overlap)));
  const num = Math.ceil(size / iv); let n = 1;
  for (let d = 0; d < num; d++) if (d * iv + roi >= size) { n = d + 1; break; }
  return Array.from({ length: n }, (_, i) => i * iv - Math.max(i * iv + roi - size, 0));
}

// gi: the scan grid (HU), brain: its brain mask (1/0); returns {labels on gi, volumes, transform}
export async function runSahSwin(gi, brain, tpl, { call, identity = false, debug = false, log = () => {}, stage = () => {} } = {}) {
  const t0 = performance.now();
  const { nx, ny, nz } = gi, P = nx * ny, N = P * nz, G = geom(gi);
  const img = new Float32Array(N);
  for (let i = 0; i < N; i++) img[i] = brain[i] ? gi.hu[i] : 0;
  const [tx, ty, tz] = tpl.n, TP = tx * ty;
  const tp = (i, j, k) => { const v = mul(tpl.A, [i, j, k]); return [v[0] + tpl.o[0], v[1] + tpl.o[1], v[2] + tpl.o[2]]; };
  // transform: template point p (LPS) -> scan point q = M p + b
  let M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], b = [0, 0, 0], params = null;
  if (!identity) {
    // centres of mass and brain volumes (initial scale)
    let cf = [0, 0, 0], sf = 0, nt = 0;
    for (let k = 0; k < tz; k++) for (let j = 0; j < ty; j++) for (let i = 0; i < tx; i++) {
      const v = tpl.T[k * TP + j * tx + i]; if (v > 5) nt++; if (v <= 0) continue; const p = tp(i, j, k); cf[0] += v * p[0]; cf[1] += v * p[1]; cf[2] += v * p[2]; sf += v;
    }
    cf = cf.map(x => x / sf);
    let cm = [0, 0, 0], sm = 0, nm = 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const v = img[z * P + y * nx + x]; if (v > 5) nm++; if (v <= 0) continue; cm[0] += v * x; cm[1] += v * y; cm[2] += v * z; sm += v;
    }
    if (!sm) throw new Error('no brain was found in this scan');
    cm = mul(G.A, cm.map(x => x / sm)).map((v, a) => v + G.o[a]);
    const vt = nt * Math.abs(tpl.A[0][0] * tpl.A[1][1] * tpl.A[2][2] || 8), vm = nm * gi.sp[0] * gi.sp[1] * gi.sp[2];
    const ls0 = Math.log(Math.min(1.25, Math.max(0.8, Math.cbrt(vm / vt))));
    const c = cf, t0v = cm.map((v, a) => v - cf[a]);
    const build = y => {
      const R = rot(y.slice(0, 3)), S = [[Math.exp(y[6]), 0, 0], [0, Math.exp(y[7]), 0], [0, 0, Math.exp(y[8])]], Mx = mm(R, S);
      const Mc = mul(Mx, c), t = [t0v[0] + y[3], t0v[1] + y[4], t0v[2] + y[5]];
      return { M: Mx, b: [c[0] + t[0] - Mc[0], c[1] + t[1] - Mc[1], c[2] + t[2] - Mc[2]] };
    };
    let x = [0, 0, 0, 0, 0, 0, ls0, ls0, ls0];
    for (const [s, sig] of [[2, 2], [1, 1]]) {   // 4 mm then 2 mm template points
      stage('Registering to the template', s === 2 ? 0.05 : 0.2);
      const Ts = blur(tpl.T, tx, ty, tz, [sig / 2, sig / 2, sig / 2]);
      const pts = [], vals = [];
      for (let k = 0; k < tz; k += s) for (let j = 0; j < ty; j += s) for (let i = 0; i < tx; i += s) { pts.push(...tp(i, j, k)); vals.push(Ts[k * TP + j * tx + i]); }
      const Mn = vals.length, Pl = Float32Array.from(pts), Tl = Float32Array.from(vals), buf = new Float32Array(Mn);
      const vs = blur(img, nx, ny, nz, gi.sp.map(v => sig * 1.5 / v));
      const cost = y => {
        const { M: Mx, b: bx } = build(y), Q = mm(G.Ai, Mx), qb = mul(G.Ai, [bx[0] - G.o[0], bx[1] - G.o[1], bx[2] - G.o[2]]);
        for (let m = 0; m < Mn; m++) {
          const p0 = Pl[3 * m], p1 = Pl[3 * m + 1], p2 = Pl[3 * m + 2];
          buf[m] = tri(vs, nx, ny, nz, Q[0][0] * p0 + Q[0][1] * p1 + Q[0][2] * p2 + qb[0], Q[1][0] * p0 + Q[1][1] * p1 + Q[1][2] * p2 + qb[1], Q[2][0] * p0 + Q[2][1] * p1 + Q[2][2] * p2 + qb[2]);
        }
        return -mi(Tl, buf, Mn);
      };
      let step = [0.04, 0.04, 0.04, 4, 4, 4, 0.04, 0.04, 0.04].map(v => v * (s === 2 ? 1 : 0.5)), best = cost(x), it = 0;
      while (step[3] > (s === 2 ? 0.5 : 0.2) && it < 1200) {
        let improved = false;
        for (let k = 0; k < 9; k++) for (const sg of [1, -1]) {
          const y = x.slice(); y[k] += sg * step[k];
          if (Math.max(Math.abs(y[0]), Math.abs(y[1]), Math.abs(y[2])) > 0.6 || Math.abs(y[k] - (k >= 6 ? 0 : y[k])) > 0.35) continue;
          const cc = cost(y); it++;
          if (cc < best - 1e-7) { best = cc; x = y; improved = true; break; }
        }
        if (!improved) step = step.map(v => v / 2);
        if (it % 40 === 0) await new Promise(r => setTimeout(r, 0));
      }
      log(`SAH Swin UNETR: registration at ${2 * s} mm: ${it} evaluations, MI ${(-best).toFixed(3)}`);
    }
    ({ M, b } = build(x));
    params = { rotation_deg: x.slice(0, 3).map(v => v * 180 / Math.PI), shift_mm: x.slice(3, 6), scale: x.slice(6).map(Math.exp) };
    log(`SAH Swin UNETR: template registration: rotation ${params.rotation_deg.map(v => v.toFixed(1)).join('/')} deg, scale ${params.scale.map(v => v.toFixed(3)).join('/')}, shift ${params.shift_mm.map(v => v.toFixed(1)).join('/')} mm from the centres of mass`);
  } else log('SAH Swin UNETR: the scan is already in the 1 mm template space (RHUH preprocessing): no registration');
  // network grid (MONAI Spacingd of the 1 mm template): point of index (i, j, k) = o + D [1.5 i, 1.5 j, 2 k]
  stage('Preparing the network input', 0.35);
  const [X0, Y0, Z0] = NET, NN = X0 * Y0 * Z0;
  const Q = mm(G.Ai, M), qb = mul(G.Ai, [b[0] - G.o[0], b[1] - G.o[1], b[2] - G.o[2]]);
  const netPt = (i, j, k) => { const v = mul(tpl.D, [PIX[0] * i, PIX[1] * j, PIX[2] * k]); return [v[0] + tpl.o[0], v[1] + tpl.o[1], v[2] + tpl.o[2]]; };
  const X = new Float32Array(NN);   // index (i*Y0 + j)*Z0 + k (MONAI array order)
  for (let i = 0; i < X0; i++) for (let j = 0; j < Y0; j++) for (let k = 0; k < Z0; k++) {
    const p = netPt(i, j, k);
    X[(i * Y0 + j) * Z0 + k] = tri(img, nx, ny, nz, Q[0][0] * p[0] + Q[0][1] * p[1] + Q[0][2] * p[2] + qb[0], Q[1][0] * p[0] + Q[1][1] * p[1] + Q[1][2] * p[2] + qb[1], Q[2][0] * p[0] + Q[2][1] * p[1] + Q[2][2] * p[2] + qb[2]);
  }
  // NormalizeIntensity(nonzero): z-score over the non-zero voxels (population std); zeros stay zero.
  // "Non-zero" as MONAI sees it after Spacingd: its bilinear resampling (float64 grid) leaves values of ~1e-14 in voxels
  // whose sampling position is a hair off a voxel of the 1 mm image and touches brain, and those count. The network is
  // sensitive to this (it shifts the mean and SD, and the volume by up to 10%), so the same voxels are counted here.
  const nzNet = nonZeroNet((a, bb, c) => {
    const p = mul(tpl.D, [a, bb, c]); p[0] += tpl.o[0]; p[1] += tpl.o[1]; p[2] += tpl.o[2];
    const fx = Q[0][0] * p[0] + Q[0][1] * p[1] + Q[0][2] * p[2] + qb[0], fy = Q[1][0] * p[0] + Q[1][1] * p[1] + Q[1][2] * p[2] + qb[1], fz = Q[2][0] * p[0] + Q[2][1] * p[1] + Q[2][2] * p[2] + qb[2];
    if (identity) { const x = Math.round(fx), y = Math.round(fy), z = Math.round(fz); return x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz && img[z * P + y * nx + x] !== 0; }
    return tri(img, nx, ny, nz, fx, fy, fz) !== 0;   // the ANTs-warped 1 mm image is non-zero where the brain reaches
  });
  let s1 = 0, s2 = 0, cnt = 0;
  for (let i = 0; i < NN; i++) if (nzNet[i] || X[i] !== 0) { nzNet[i] = 1; const v = X[i]; s1 += v; s2 += v * v; cnt++; }
  if (!cnt) throw new Error('the brain fell outside the template');
  const mean = s1 / cnt, sd = Math.sqrt(Math.max(s2 / cnt - mean * mean, 1e-12));
  for (let i = 0; i < NN; i++) if (nzNet[i]) X[i] = (X[i] - mean) / sd;
  // CropForeground: bounding box of the voxels > 0 (after normalisation: above the mean)
  const lo = [X0, Y0, Z0], hi = [-1, -1, -1];
  for (let i = 0; i < X0; i++) for (let j = 0; j < Y0; j++) for (let k = 0; k < Z0; k++) if (X[(i * Y0 + j) * Z0 + k] > 0) {
    if (i < lo[0]) lo[0] = i; if (i > hi[0]) hi[0] = i; if (j < lo[1]) lo[1] = j; if (j > hi[1]) hi[1] = j; if (k < lo[2]) lo[2] = k; if (k > hi[2]) hi[2] = k;
  }
  const S = [0, 1, 2].map(a => hi[a] - lo[a] + 1), Sp = S.map((v, a) => Math.max(v, ROI[a])), pad = S.map((v, a) => Math.floor((Sp[a] - v) / 2));
  const st = [0, 1, 2].map(a => starts(Sp[a], ROI[a])), nWin = st[0].length * st[1].length * st[2].length;
  log(`SAH Swin UNETR: network grid ${NET.join('x')} at 1.5 x 1.5 x 2 mm, crop ${S.join('x')}, ${nWin} window(s) of ${ROI.join('x')}`);
  // padded-crop coordinate (u, v, w) -> net index (u - pad + lo)
  const D = new Float32Array(Sp[0] * Sp[1] * Sp[2]), C = new Uint8Array(D.length);   // summed logit difference (class 1 - class 0), window count
  const win = new Float32Array(ROI[0] * ROI[1] * ROI[2]);
  let w = 0;
  for (const a0 of st[0]) for (const b0 of st[1]) for (const c0 of st[2]) {
    stage(`Segmenting with Swin UNETR · window ${w + 1} of ${nWin}`, 0.4 + 0.55 * w / nWin);
    win.fill(0);
    for (let u = 0; u < ROI[0]; u++) { const i = a0 + u - pad[0] + lo[0]; if (a0 + u - pad[0] < 0 || a0 + u - pad[0] >= S[0]) continue;
      for (let v = 0; v < ROI[1]; v++) { const j = b0 + v - pad[1] + lo[1]; if (b0 + v - pad[1] < 0 || b0 + v - pad[1] >= S[1]) continue;
        const src = (i * Y0 + j) * Z0, dst = (u * ROI[1] + v) * ROI[2];
        for (let x = 0; x < ROI[2]; x++) { const kk = c0 + x - pad[2]; if (kk >= 0 && kk < S[2]) win[dst + x] = X[src + kk + lo[2]]; }
      } }
    const out = await call('sahswin', { x: { data: win, dims: [1, 1, ...ROI] } });
    const y = out[0].data, R3 = ROI[0] * ROI[1] * ROI[2];
    for (let u = 0; u < ROI[0]; u++) for (let v = 0; v < ROI[1]; v++) for (let x = 0; x < ROI[2]; x++) {
      const r = (u * ROI[1] + v) * ROI[2] + x, d = ((a0 + u) * Sp[1] + b0 + v) * Sp[2] + c0 + x;
      D[d] += y[R3 + r] - y[r]; C[d]++;
    }
    w++;
  }
  // logit difference on the full network grid (0 outside the crop, as MONAI's inverse pads with zeros)
  const L = new Float32Array(NN); let nT = 0;
  for (let i = 0; i < S[0]; i++) for (let j = 0; j < S[1]; j++) for (let k = 0; k < S[2]; k++) {
    const d = ((i + pad[0]) * Sp[1] + j + pad[1]) * Sp[2] + k + pad[2], v = C[d] ? D[d] / C[d] : 0;
    L[((i + lo[0]) * Y0 + j + lo[1]) * Z0 + k + lo[2]] = v; if (v > 0) nT++;
  }
  // back to the scan: network-grid coordinates of each voxel (inverse transform), trilinear logit difference > 0
  stage('Mapping the segmentation back to the scan', 0.97);
  const Mi = inv3(M), Di = inv3(tpl.D), labels = new Uint8Array(N);
  const tl = (i, j, k) => (i < 0 || j < 0 || k < 0 || i >= X0 || j >= Y0 || k >= Z0) ? 0 : L[(i * Y0 + j) * Z0 + k];
  for (let z = 0; z < nz; z++) for (let yy = 0; yy < ny; yy++) for (let xx = 0; xx < nx; xx++) {
    const q = mul(G.A, [xx, yy, z]), p = mul(Mi, [q[0] + G.o[0] - b[0], q[1] + G.o[1] - b[1], q[2] + G.o[2] - b[2]]);
    const t = mul(Di, [p[0] - tpl.o[0], p[1] - tpl.o[1], p[2] - tpl.o[2]]), fi = t[0] / PIX[0], fj = t[1] / PIX[1], fk = t[2] / PIX[2];
    const i0 = Math.floor(fi), j0 = Math.floor(fj), k0 = Math.floor(fk);
    if (i0 < -1 || j0 < -1 || k0 < -1 || i0 >= X0 || j0 >= Y0 || k0 >= Z0) continue;
    const a = fi - i0, bb = fj - j0, cc = fk - k0;
    let v = 0;
    for (let e = 0; e < 2; e++) for (let f = 0; f < 2; f++) for (let h = 0; h < 2; h++) {
      const ww = (e ? a : 1 - a) * (f ? bb : 1 - bb) * (h ? cc : 1 - cc); if (ww) v += ww * tl(i0 + e, j0 + f, k0 + h);
    }
    if (v > 0) labels[z * P + yy * nx + xx] = 1;
  }
  const vox = gi.sp[0] * gi.sp[1] * gi.sp[2] / 1000; let n = 0;
  for (let i = 0; i < N; i++) n += labels[i];
  const volumes = { SAH: n * vox };
  log(`SAH Swin UNETR: ${nT} network voxels (${(nT * 4.5 / 1000).toFixed(1)} mL in template space); on the scan ${volumes.SAH.toFixed(1)} mL (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
  return { labels, volumes, voxels: [N - n, n], windows: nWin, template_mL: nT * 4.5 / 1000, transform: params || { identity: true }, ...(debug ? { debug: { L, X, lo, S } } : {}) };
}
