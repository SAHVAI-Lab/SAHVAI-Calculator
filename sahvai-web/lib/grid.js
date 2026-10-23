// Volume (series.js / nifti.js / nrrd.js form) -> working grid for the model.
// Axes are reoriented so x runs towards the patient's Left, y towards Posterior and z
// towards Superior (the order SimpleITK gives for a standard axial head CT, which is what
// the BrainHemoAI network was trained on), and resampled to the training geometry
// (Hu et al., NeuroImage 2023): 0.5 x 0.5 mm in plane and 5 mm slices. A native spacing
// already within 0.45-0.55 mm (in plane) or 4.5-5.5 mm (slices) is kept as is.
// Thin slices are averaged into 5 mm slabs (area weighting), which mimics a thick
// reconstruction; otherwise linear interpolation is used.
// Output: { nx, ny, nz, sp:[sx,sy,sz], hu: Float32Array (z-major), affine (LPS, 4x4 rows), info }

export function orientation(vol) {
  const perm = [-1, -1, -1], flip = [false, false, false];
  const order = [0, 1, 2].sort((p, q) => Math.max(...vol.dir[q].map(Math.abs)) - Math.max(...vol.dir[p].map(Math.abs)));
  for (const s of order) {
    let best = -1;
    for (let a = 0; a < 3; a++) if (perm[a] < 0 && (best < 0 || Math.abs(vol.dir[s][a]) > Math.abs(vol.dir[s][best]))) best = a;
    perm[best] = s;
  }
  for (let a = 0; a < 3; a++) flip[a] = (vol.dir[perm[a]][a] || 1) < 0;
  return { perm, flip };
}

// weights for one axis: output t -> [[src, w], ...] in (unflipped) LPS-ordered index.
// align 'extent': voxel centres at (t + 0.5) * r - 0.5 (pixel-centred, keeps the physical extent; skimage/nnU-Net style)
// align 'corner': voxel centres at t * r (first voxel centres coincide; SimpleITK Resample style). An empty list = outside.
// interp 'auto': area average when shrinking by > 1.5x, else linear; 'linear'; 'nearest' (label maps);
// 'cubic': cubic B-spline in plane (scipy/skimage order 3, as nnU-Net), 'auto' along the slices.
function axisWeights(n, sp, tgt, T, align = 'extent', interp = 'auto') {
  const r = tgt / sp, out = [];
  for (let t = 0; t < T; t++) {
    const list = [];
    if (interp === 'auto' && r > 1.5 && align === 'extent') {   // area average
      const a = t * r, b = (t + 1) * r;
      for (let i = Math.max(0, Math.floor(a)); i < Math.min(n, Math.ceil(b)); i++) {
        const w = Math.min(b, i + 1) - Math.max(a, i); if (w > 1e-6) list.push([i, w]);
      }
      const s = list.reduce((x, y) => x + y[1], 0) || 1; list.forEach(p => p[1] /= s);
      if (!list.length) list.push([Math.min(n - 1, Math.max(0, Math.round(a))), 1]);
    } else {
      let u = align === 'corner' ? t * r : (t + 0.5) * r - 0.5;
      if (align === 'corner' && u > n - 1 + 1e-6) { out.push(list); continue; }
      u = Math.min(Math.max(u, 0), n - 1);
      if (interp === 'nearest') { list.push([Math.min(n - 1, Math.round(u)), 1]); out.push(list); continue; }
      if (interp === 'cubic') {   // taps on B-spline coefficients (the plane is prefiltered first)
        const i = Math.floor(u), f = u - i, f2 = f * f, f3 = f2 * f, cl = k => k < 0 ? 0 : k >= n ? n - 1 : k;
        const w = [(1 - f) ** 3 / 6, (4 - 6 * f2 + 3 * f3) / 6, (1 + 3 * f + 3 * f2 - 3 * f3) / 6, f3 / 6];
        for (let k = 0; k < 4; k++) if (w[k]) list.push([cl(i - 1 + k), w[k]]);
        out.push(list); continue;
      }
      const i = Math.floor(u), f = u - i;
      if (f > 1e-6 && i + 1 < n) list.push([i, 1 - f], [i + 1, f]); else list.push([i, 1]);
    }
    out.push(list);
  }
  return out;
}

// cubic B-spline prefilter (Unser) along a strided line, mirror boundaries
const POLE = Math.sqrt(3) - 2;
function splineFilter(a, off, stride, n) {
  if (n < 2) return;
  const z = POLE, at = k => off + k * stride;
  for (let k = 0; k < n; k++) a[at(k)] *= 6;
  let s = a[at(0)], zn = z;
  const K = Math.min(n, 30);
  for (let k = 1; k < K; k++) { s += zn * a[at(k)]; zn *= z; }
  a[at(0)] = s;
  for (let k = 1; k < n; k++) a[at(k)] += z * a[at(k - 1)];
  a[at(n - 1)] = (z / (z * z - 1)) * (a[at(n - 1)] + z * a[at(n - 2)]);
  for (let k = n - 2; k >= 0; k--) a[at(k)] = z * (a[at(k + 1)] - a[at(k)]);
}

export function targetSpacing(sp) {
  return [
    sp[0] >= 0.45 && sp[0] <= 0.55 ? sp[0] : 0.5,
    sp[1] >= 0.45 && sp[1] <= 0.55 ? sp[1] : 0.5,
    sp[2] >= 4.5 && sp[2] <= 5.5 ? sp[2] : 5,
  ];
}

// opts: spacing(sp) -> target spacing (default: SAHVAI rule above); align 'extent' | 'corner'; interp 'auto' | 'linear' | 'nearest' | 'cubic';
//       pre(v) -> v applied to source voxels before resampling
export async function toGrid(vol, log = () => {}, progress = () => {}, opts = {}) {
  const t0 = performance.now();
  const { perm, flip } = orientation(vol);
  const inv = [0, 1, 2].map(s => perm.indexOf(s));
  const n = perm.map(s => vol.dims[s]), sp = perm.map(s => vol.spacing[s]);
  const align = opts.align || 'extent', interp = opts.interp || 'auto';
  const tgt = (opts.spacing || targetSpacing)(sp);
  const T = align === 'corner'
    ? [0, 1, 2].map(a => Math.max(1, Math.round((n[a] - 1) * sp[a] / tgt[a]) + 1))
    : [0, 1, 2].map(a => Math.max(1, Math.round(n[a] * sp[a] / tgt[a])));
  const tgtExact = align === 'corner' ? tgt.slice() : [0, 1, 2].map(a => n[a] * sp[a] / T[a]);   // extent: keep the physical extent
  const W = [0, 1, 2].map(a => axisWeights(n[a], sp[a], tgtExact[a], T[a], align, interp === 'cubic' && perm[a] === 2 ? 'auto' : interp).map(l => l.map(([i, w]) => [flip[a] ? n[a] - 1 - i : i, w])));
  log(`${opts.label ? opts.label + ': ' : ''}reorient to L-P-S: source axes ${perm.join(',')} flips ${flip.map(Number).join(',')}; ${n.join('x')} @ ${sp.map(s => s.toFixed(3)).join(', ')} mm -> ${T.join('x')} @ ${tgtExact.map(s => s.toFixed(3)).join(', ')} mm`);

  const [nx, ny, nz] = T, stride = [1, nx, nx * ny];
  const hu = new Float32Array(nx * ny * nz);
  const aS = inv[2], a0 = inv[0], a1 = inv[1];
  const n0 = vol.dims[0], n1 = vol.dims[1];
  // which source slices each output plane needs
  const need = W[aS].map(l => l.map(p => p[0])), lastUse = new Map();
  need.forEach((ks, t) => ks.forEach(k => lastUse.set(k, t)));
  const uniq = [...new Set(need.flat())].sort((a, b) => a - b);
  log(`decoding ${uniq.length} of ${vol.dims[2]} source slices`);
  const cache = new Map(); let next = 0, done = 0;
  const pre = opts.pre;   // optional per-voxel map applied before resampling (e.g. nnU-Net clips before it resamples)
  const get = k => { if (!cache.has(k)) cache.set(k, vol.slice(k).then(v => { progress(++done / uniq.length); return pre ? v.map(pre) : v; })); return cache.get(k); };
  const prefetch = () => { while (next < uniq.length && cache.size < 12) get(uniq[next++]); };
  const plane = new Float32Array(n0 * n1), Tx = T[a0], Ty = T[a1], rows = new Float32Array(Tx * n1);
  for (let t = 0; t < T[aS]; t++) {
    prefetch();
    plane.fill(0);
    for (const [k, w] of W[aS][t]) { const s = await get(k); for (let i = 0; i < plane.length; i++) plane[i] += w * s[i]; }
    if (!W[aS][t].length) plane.fill(NaN);
    else if (interp === 'cubic') { for (let y = 0; y < n1; y++) splineFilter(plane, y * n0, 1, n0); for (let x = 0; x < n0; x++) splineFilter(plane, x, n0, n1); }
    // along source axis 0
    for (let y = 0; y < n1; y++) {
      const r = y * n0;
      for (let x = 0; x < Tx; x++) { const wl = W[a0][x]; let v = wl.length ? 0 : NaN; for (const [i, w] of wl) v += w * plane[r + i]; rows[y * Tx + x] = v; }
    }
    // along source axis 1, write into the L-P-S grid
    const base = t * stride[aS];
    for (let y = 0; y < Ty; y++) {
      const ob = base + y * stride[a1], wl = W[a1][y];
      for (let x = 0; x < Tx; x++) { let v = wl.length ? 0 : NaN; for (const [i, w] of wl) v += w * rows[i * Tx + x]; hu[ob + x * stride[a0]] = v; }
    }
    for (const k of need[t]) if (lastUse.get(k) === t) cache.delete(k);
  }
  if (align === 'corner') {
    let mn = Infinity, any = false;
    for (let i = 0; i < hu.length; i++) { const v = hu[i]; if (v !== v) any = true; else if (v < mn) mn = v; }
    if (any) for (let i = 0; i < hu.length; i++) if (hu[i] !== hu[i]) hu[i] = mn;
  }
  // affine (LPS) of the output grid: columns = axis direction * spacing, last column = centre of voxel 0
  const dirOut = [0, 1, 2].map(a => vol.dir[perm[a]].map(c => flip[a] ? -c : c));
  let affine = null;
  if (vol.origin) {
    const idx0 = [0, 0, 0];   // source (unflipped) index of output voxel 0's centre, per source axis
    for (let a = 0; a < 3; a++) { const u = align === 'corner' ? 0 : 0.5 * tgtExact[a] / sp[a] - 0.5; idx0[perm[a]] = flip[a] ? n[a] - 1 - u : u; }
    const o = [0, 1, 2].map(c => vol.origin[c] + idx0.reduce((acc, u, s) => acc + u * vol.spacing[s] * vol.dir[s][c], 0));
    affine = [0, 1, 2].map(c => [dirOut[0][c] * tgtExact[0], dirOut[1][c] * tgtExact[1], dirOut[2][c] * tgtExact[2], o[c]]).concat([[0, 0, 0, 1]]);
  }
  log(`working grid ${nx}x${ny}x${nz} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return { nx, ny, nz, sp: tgtExact, hu, affine, native: { dims: n, spacing: sp } };
}
