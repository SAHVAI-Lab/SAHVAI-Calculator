// DeepBleed (Sharrock MF, Mould WA, Ali H, et al. 3D deep neural network segmentation of intracerebral hemorrhage:
// development and validation for clinical trials. Neuroinformatics 2021;19:403-415; github.com/msharrock/deepbleed,
// CC BY-NC-ND 4.0; MISTIE-2 weights) in the browser.
//
// DeepBleed's predict.py: brain extraction (FSL BET for CT) -> rigid registration (ANTs, Mattes MI) of the brain to a
// 1.5 mm, 128^3 CT template -> 3D V-Net -> mask > 0.5 -> inverse transform back to the scan (genericLabel).
// Here: brain extraction is SAHVAI's (lib/brain.js; FSL is not available in a browser); the network is the original,
// converted to ONNX without changes; the geometry reproduces predict.py, including one quirk that the network was
// trained with: its nii2ants() hands ANTs the NIfTI qform in RAS numbers, which ANTs reads as LPS, so the scan's frame
// is turned 180 degrees about the vertical axis relative to the template, and the rigid registration (started from
// the centres of mass with no rotation) keeps it so. Moving-image coordinates below are therefore q_a = F * q_LPS,
// F = diag(-1, -1, 1); the transform is ITK's: q_a = R (p - c) + c + t for template points p (LPS).
// Registration: centre-of-mass initialisation as ANTs, then (optional) a short Mattes-style mutual-information
// refinement with rotations kept small, as the ANTs optimiser does from that start.

const F = [-1, -1, 1];
const NT = 128;

// template: niftiVolume() of templates/scct_unsmooth_SS_0.01_128x128x128.nii.gz -> {T (Float32, index (i*128+j)*128+k), A (3x3 LPS), o (LPS)}
export async function loadTemplate(vol) {
  const [n0, n1, n2] = vol.dims;
  if (n0 !== NT || n1 !== NT || n2 !== NT) throw new Error('unexpected DeepBleed template size');
  const T = new Float32Array(NT * NT * NT);
  for (let k = 0; k < NT; k++) {
    const s = await vol.slice(k);   // x fastest
    for (let j = 0; j < NT; j++) for (let i = 0; i < NT; i++) T[(i * NT + j) * NT + k] = s[j * NT + i];
  }
  const A = [0, 1, 2].map(r => [0, 1, 2].map(c => vol.dir[c][r] * vol.spacing[c]));   // column c = axis c
  return { T, A, o: vol.origin || [0, 0, 0] };
}

const mul = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2], M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2], M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]];
function inv3(M) {
  const [a, b, c] = M[0], [d, e, f] = M[1], [g, h, i] = M[2], A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det], [B / det, (a * i - c * g) / det, -(a * f - c * d) / det], [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
}
function rot([ax, ay, az]) {   // Rz Ry Rx
  const cx = Math.cos(ax), sx = Math.sin(ax), cy = Math.cos(ay), sy = Math.sin(ay), cz = Math.cos(az), sz = Math.sin(az);
  return [[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], [-sy, cy * sx, cy * cx]];
}
const T3 = M => [[M[0][0], M[1][0], M[2][0]], [M[0][1], M[1][1], M[2][1]], [M[0][2], M[1][2], M[2][2]]];

// grid geometry (LPS affine rows) -> index<->physical helpers
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

// separable Gaussian blur of a z-major grid, sigma in voxels per axis
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

// mutual information of two samples (bins x bins joint histogram over each sample's range)
function mi(a, b, n, bins = 64) {
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

// gi: grid the network input is sampled from (HU), brain: its brain mask; returns {labels (on gi), prob stats, transform}
export async function runDeepBleed(gi, brain, tpl, { call, refine = true, log = () => {}, stage = () => {} } = {}) {
  const t0 = performance.now();
  const { nx, ny, nz } = gi, P = nx * ny, N = P * nz;
  const img = new Float32Array(N);
  for (let i = 0; i < N; i++) img[i] = brain[i] ? gi.hu[i] : 0;
  const G = geom(gi);
  // template points p (LPS) for index (i, j, k) at shrink s
  const tp = (i, j, k) => { const v = mul(tpl.A, [i, j, k]); return [v[0] + tpl.o[0], v[1] + tpl.o[1], v[2] + tpl.o[2]]; };
  // centres of mass (intensity-weighted, as ITK's moments initialiser): template in LPS, moving in the 'ants' frame
  let cf = [0, 0, 0], sf = 0;
  for (let i = 0; i < NT; i++) for (let j = 0; j < NT; j++) for (let k = 0; k < NT; k++) {
    const v = tpl.T[(i * NT + j) * NT + k]; if (!v) continue; const p = tp(i, j, k); cf[0] += v * p[0]; cf[1] += v * p[1]; cf[2] += v * p[2]; sf += v;
  }
  cf = cf.map(x => x / sf);
  let cm = [0, 0, 0], sm = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const v = img[z * P + y * nx + x]; if (!v) continue; cm[0] += v * x; cm[1] += v * y; cm[2] += v * z; sm += v;
  }
  if (!sm) throw new Error('no brain was found in this scan');
  cm = cm.map(x => x / sm);
  const cmL = mul(G.A, cm).map((v, a) => v + G.o[a]), cma = cmL.map((v, a) => v * F[a]);
  const c = cf, t0v = [cma[0] - cf[0], cma[1] - cf[1], cma[2] - cf[2]];
  // q_a -> moving grid index
  const toIdx = qa => { const l = [qa[0] * F[0] - G.o[0], qa[1] * F[1] - G.o[1], qa[2] * F[2] - G.o[2]]; return mul(G.Ai, l); };
  const sampleAt = (vol, R, t, p) => {
    const d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]], r = mul(R, d), q = [r[0] + c[0] + t[0], r[1] + c[1] + t[1], r[2] + c[2] + t[2]];
    const ix = toIdx(q); return tri(vol, nx, ny, nz, ix[0], ix[1], ix[2]);
  };
  let x = [0, 0, 0, 0, 0, 0];
  if (refine) {
    for (const [s, sig] of [[4, 2], [2, 1]]) {
      stage('Registering to the DeepBleed template', s === 4 ? 0.1 : 0.4);
      const n = NT / s, M = n * n * n;
      // template at shrink s: block-mean of the Gaussian-smoothed template; points at block centres
      const Ts = blur(tpl.T, NT, NT, NT, [sig, sig, sig]);   // symmetric, so index order does not matter
      const Tl = new Float32Array(M), Pl = new Float32Array(M * 3);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
        const m = (i * n + j) * n + k, ci = i * s + (s - 1) / 2, cj = j * s + (s - 1) / 2, ck = k * s + (s - 1) / 2;
        Tl[m] = Ts[(Math.round(ci) * NT + Math.round(cj)) * NT + Math.round(ck)];
        const p = tp(ci, cj, ck); Pl[3 * m] = p[0]; Pl[3 * m + 1] = p[1]; Pl[3 * m + 2] = p[2];
      }
      const vs = blur(img, nx, ny, nz, gi.sp.map(v => sig * 1.5 / v));
      const buf = new Float32Array(M);
      const cost = y => {
        const R = rot(y.slice(0, 3)), t = [t0v[0] + y[3], t0v[1] + y[4], t0v[2] + y[5]];
        for (let m = 0; m < M; m++) buf[m] = sampleAt(vs, R, t, [Pl[3 * m], Pl[3 * m + 1], Pl[3 * m + 2]]);
        return -mi(Tl, buf, M);
      };
      let step = [0.01, 0.01, 0.01, 4, 4, 4].map(v => v * (s / 6 + 0.25)), best = cost(x), it = 0;
      while (step[3] > 0.05 * Math.max(1, s / 2) && it < 400) {
        let improved = false;
        for (let k = 0; k < 6 && !improved; k++) for (const sg of [1, -1]) {
          const y = x.slice(); y[k] += sg * step[k];
          if (Math.max(Math.abs(y[0]), Math.abs(y[1]), Math.abs(y[2])) > 0.1) continue;
          const cc = cost(y); it++;
          if (cc < best) { best = cc; x = y; improved = true; break; }
        }
        if (!improved) step = step.map(v => v / 2);
      }
      await new Promise(r => setTimeout(r, 0));
    }
  }
  const R = rot(x.slice(0, 3)), t = [t0v[0] + x[3], t0v[1] + x[4], t0v[2] + x[5]];
  log(`DeepBleed: rigid registration ${refine ? 'centre of mass + mutual information' : 'centre of mass'}: rotation ${x.slice(0, 3).map(v => (v * 180 / Math.PI).toFixed(1)).join('/')} deg, shift ${x.slice(3).map(v => v.toFixed(1)).join('/')} mm from the centres of mass`);
  // network input: the brain resampled onto the template grid (linear, outside 0), index (i*128 + j)*128 + k
  stage('Preparing the network input', 0.7);
  const X = new Float32Array(NT * NT * NT);
  for (let i = 0; i < NT; i++) for (let j = 0; j < NT; j++) for (let k = 0; k < NT; k++) X[(i * NT + j) * NT + k] = sampleAt(img, R, t, tp(i, j, k));
  stage('Segmenting with DeepBleed V-Net', 0.75);
  const out = await call('deepbleed', { x: { data: X, dims: [1, 1, NT, NT, NT] } });
  const prob = out[0].data, pm = new Float32Array(NT * NT * NT); let nT = 0;
  for (let i = 0; i < pm.length; i++) if (prob[i] > 0.5) { pm[i] = 1; nT++; }
  // back to the scan grid: template index of each voxel, trilinear of the binary mask >= 0.5 (as ANTs' genericLabel)
  stage('Mapping the segmentation back to the scan', 0.95);
  const Rt = T3(R), TAi = inv3(tpl.A), labels = new Uint8Array(N);
  const tm = (i, j, k) => (i < 0 || j < 0 || k < 0 || i > NT - 1 || j > NT - 1 || k > NT - 1) ? 0 : pm[(i * NT + j) * NT + k];
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let xx = 0; xx < nx; xx++) {
    const l = mul(G.A, [xx, y, z]), qa = [(l[0] + G.o[0]) * F[0] - c[0] - t[0], (l[1] + G.o[1]) * F[1] - c[1] - t[1], (l[2] + G.o[2]) * F[2] - c[2] - t[2]];
    const r = mul(Rt, qa), p = [r[0] + c[0] - tpl.o[0], r[1] + c[1] - tpl.o[1], r[2] + c[2] - tpl.o[2]], ti = mul(TAi, p);
    const i0 = Math.floor(ti[0]), j0 = Math.floor(ti[1]), k0 = Math.floor(ti[2]);
    if (i0 < -1 || j0 < -1 || k0 < -1 || i0 > NT - 1 || j0 > NT - 1 || k0 > NT - 1) continue;
    const fi = ti[0] - i0, fj = ti[1] - j0, fk = ti[2] - k0;
    let v = 0;
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let d = 0; d < 2; d++) {
      const w = (a ? fi : 1 - fi) * (b ? fj : 1 - fj) * (d ? fk : 1 - fk); if (w) v += w * tm(i0 + a, j0 + b, k0 + d);
    }
    if (v >= 0.5) labels[z * P + y * nx + xx] = 1;
  }
  const vox = gi.sp[0] * gi.sp[1] * gi.sp[2] / 1000; let n = 0;
  for (let i = 0; i < N; i++) n += labels[i];
  const volumes = { ICH: n * vox };
  log(`DeepBleed: ${nT} template voxels (${(nT * 3.375 / 1000).toFixed(1)} mL at 1.5 mm); on the scan ${volumes.ICH.toFixed(1)} mL (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
  return { labels, volumes, voxels: [N - n, n], transform: { rotation_deg: x.slice(0, 3).map(v => v * 180 / Math.PI), shift_mm: x.slice(3), refine } };
}
