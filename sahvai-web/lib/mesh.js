// SAHVAI 3D-4D: surfaces for the 3D view, built from the working grid of the BrainHemoAI pipeline.
// Grid axes (lib/grid.js): x towards patient Left, y towards Posterior, z towards Superior.
// World axes for three.js: +X = Left, +Y = Superior, +Z = Anterior (a rotation of the grid, so winding is kept).
import { edgeTable, triTable } from './mc-tables.js';
import { sqdist } from './morph.js';

// Occupancy field: fraction of each f x f in-plane block whose voxel passes `test`, padded by `pad` empty voxels.
// The field index i maps to grid millimetres org + i * sp on each axis.
export function blockField(src, g, test, f = 2, pad = 2) {
  const { nx, ny, nz, sp } = g, bx = Math.ceil(nx / f), by = Math.ceil(ny / f);
  const NX = bx + 2 * pad, NY = by + 2 * pad, NZ = nz + 2 * pad, data = new Float32Array(NX * NY * NZ);
  const inv = 1 / (f * f);
  for (let z = 0; z < nz; z++) {
    const o = z * nx * ny, oz = (z + pad) * NX * NY;
    for (let y = 0; y < ny; y++) {
      const oy = oz + (((y / f) | 0) + pad) * NX + pad, row = o + y * nx;
      for (let x = 0; x < nx; x++) if (test(src[row + x], row + x)) data[oy + ((x / f) | 0)] += inv;
    }
  }
  return {
    data, nx: NX, ny: NY, nz: NZ,
    sp: [sp[0] * f, sp[1] * f, sp[2]],
    org: [((f - 1) / 2 - pad * f) * sp[0], ((f - 1) / 2 - pad * f) * sp[1], -pad * sp[2]],
  };
}

function gauss1d(sigma) {
  const r = Math.max(1, Math.ceil(sigma * 2.5)), k = new Float32Array(2 * r + 1); let s = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-0.5 * (i / sigma) ** 2); s += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return k;
}
// separable Gaussian, sigma in millimetres (an axis is skipped when sigma < 0.35 voxel)
export function blur(F, sigmaMm) {
  const dims = [F.nx, F.ny, F.nz], strides = [1, F.nx, F.nx * F.ny];
  let a = F.data, b = new Float32Array(a.length);
  for (let ax = 0; ax < 3; ax++) {
    const s = (Array.isArray(sigmaMm) ? sigmaMm[ax] : sigmaMm) / F.sp[ax];
    if (s < 0.35) continue;
    const k = gauss1d(s), r = (k.length - 1) >> 1, n = dims[ax], st = strides[ax];
    const o1 = dims[(ax + 1) % 3], o2 = dims[(ax + 2) % 3], s1 = strides[(ax + 1) % 3], s2 = strides[(ax + 2) % 3];
    for (let j = 0; j < o2; j++) for (let i = 0; i < o1; i++) {
      const base = i * s1 + j * s2;
      for (let t = 0; t < n; t++) {
        let acc = 0;
        for (let q = -r; q <= r; q++) { const u = t + q; if (u >= 0 && u < n) acc += k[q + r] * a[base + u * st]; }
        b[base + t * st] = acc;
      }
    }
    const tmp = a; a = b; b = tmp === F.data ? new Float32Array(a.length) : tmp;
  }
  return { ...F, data: a };
}

// linear upsampling along z by an integer factor (5 mm slices -> smoother surfaces)
export function upsampleZ(F, u) {
  if (u <= 1) return F;
  const P = F.nx * F.ny, NZ = (F.nz - 1) * u + 1, out = new Float32Array(P * NZ);
  for (let k = 0; k < NZ; k++) {
    const t = k / u, z0 = Math.min(F.nz - 1, Math.floor(t)), z1 = Math.min(F.nz - 1, z0 + 1), w = t - z0;
    const a = z0 * P, b = z1 * P, o = k * P;
    for (let p = 0; p < P; p++) out[o + p] = F.data[a + p] * (1 - w) + F.data[b + p] * w;
  }
  return { ...F, data: out, nz: NZ, sp: [F.sp[0], F.sp[1], F.sp[2] / u] };
}

// Marching cubes (Lorensen & Cline; tables and corner order as three.js). The field is "inside" above iso.
// Returns positions (grid mm) and outward normals from the field gradient.
const CORNER = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const EDGE = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
export function marchingCubes(F, iso = 0.5) {
  const { nx, ny, nz, data, sp, org } = F, P = nx * ny;
  const pos = [], nrm = [];
  const val = new Float32Array(8), vp = new Float32Array(36), vn = new Float32Array(36), gr = new Float32Array(24);
  const at = (x, y, z) => data[z * P + y * nx + x];
  const grad = (x, y, z, o) => {
    gr[o] = (at(Math.min(nx - 1, x + 1), y, z) - at(Math.max(0, x - 1), y, z)) / sp[0];
    gr[o + 1] = (at(x, Math.min(ny - 1, y + 1), z) - at(x, Math.max(0, y - 1), z)) / sp[1];
    gr[o + 2] = (at(x, y, Math.min(nz - 1, z + 1)) - at(x, y, Math.max(0, z - 1))) / sp[2];
  };
  for (let z = 0; z < nz - 1; z++) for (let y = 0; y < ny - 1; y++) for (let x = 0; x < nx - 1; x++) {
    let ci = 0;
    for (let c = 0; c < 8; c++) { const C = CORNER[c]; val[c] = at(x + C[0], y + C[1], z + C[2]); if (val[c] < iso) ci |= 1 << c; }
    const bits = edgeTable[ci];
    if (!bits) continue;
    for (let c = 0; c < 8; c++) { const C = CORNER[c]; grad(x + C[0], y + C[1], z + C[2], c * 3); }
    for (let e = 0; e < 12; e++) {
      if (!(bits & (1 << e))) continue;
      const [a, b] = EDGE[e], A = CORNER[a], B = CORNER[b], d = val[b] - val[a];
      const t = Math.abs(d) < 1e-9 ? 0.5 : (iso - val[a]) / d;
      for (let k = 0; k < 3; k++) {
        vp[e * 3 + k] = org[k] + (([x, y, z][k]) + A[k] + t * (B[k] - A[k])) * sp[k];
        vn[e * 3 + k] = -(gr[a * 3 + k] + t * (gr[b * 3 + k] - gr[a * 3 + k]));
      }
      const l = Math.hypot(vn[e * 3], vn[e * 3 + 1], vn[e * 3 + 2]) || 1;
      vn[e * 3] /= l; vn[e * 3 + 1] /= l; vn[e * 3 + 2] /= l;
    }
    const o = ci << 4;
    for (let i = 0; triTable[o + i] !== -1; i += 3) {
      // with "inside above iso" the table's own order (e0, e1, e2) is counter-clockwise seen from outside
      for (const e of [triTable[o + i], triTable[o + i + 1], triTable[o + i + 2]]) {
        pos.push(vp[e * 3], vp[e * 3 + 1], vp[e * 3 + 2]); nrm.push(vn[e * 3], vn[e * 3 + 1], vn[e * 3 + 2]);
      }
    }
  }
  return { pos: new Float32Array(pos), nrm: new Float32Array(nrm) };
}

// grid millimetres -> world (centred on c, +X left, +Y superior, +Z anterior)
export function toWorld(m, c) {
  const { pos, nrm } = m, P = new Float32Array(pos.length), N = new Float32Array(nrm.length);
  for (let i = 0; i < pos.length; i += 3) {
    P[i] = pos[i] - c[0]; P[i + 1] = pos[i + 2] - c[2]; P[i + 2] = -(pos[i + 1] - c[1]);
    N[i] = nrm[i]; N[i + 1] = nrm[i + 2]; N[i + 2] = -nrm[i + 1];
  }
  return { pos: P, nrm: N, tris: pos.length / 9 };
}

// centre of mass of a mask, in grid millimetres
export function centroid(mask, g) {
  const { nx, ny, nz, sp } = g; let sx = 0, sy = 0, sz = 0, n = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) { const o = (z * ny + y) * nx; for (let x = 0; x < nx; x++) if (mask[o + x]) { sx += x; sy += y; sz += z; n++; } }
  return n ? [sx / n * sp[0], sy / n * sp[1], sz / n * sp[2]] : [nx * sp[0] / 2, ny * sp[1] / 2, nz * sp[2] / 2];
}

// in-plane box mean of radius r (summed-area table), to tell CSF from noisy parenchyma
function boxMean(a, nx, ny, r) {
  const S = new Float64Array((nx + 1) * (ny + 1)), out = new Float32Array(nx * ny);
  for (let y = 0; y < ny; y++) { let row = 0; for (let x = 0; x < nx; x++) { row += a[y * nx + x]; S[(y + 1) * (nx + 1) + x + 1] = S[y * (nx + 1) + x + 1] + row; } }
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const x0 = Math.max(0, x - r), x1 = Math.min(nx, x + r + 1), y0 = Math.max(0, y - r), y1 = Math.min(ny, y + r + 1);
    out[y * nx + x] = (S[y1 * (nx + 1) + x1] - S[y0 * (nx + 1) + x1] - S[y1 * (nx + 1) + x0] + S[y0 * (nx + 1) + x0]) / ((x1 - x0) * (y1 - y0));
  }
  return out;
}

// Approximate ventricular system for context: CSF-density voxels (in-plane mean -15 to 8 HU) deep inside the brain
// (> 12 mm from its outline on that slice) plus IVH, kept as 3D components of at least 1 mL near the midline.
// HU-based and approximate; not a segmentation from the model.
export function ventricles(g, mask, labels) {
  const { nx, ny, nz, sp, hu } = g, P = nx * ny, cand = new Uint8Array(P * nz);
  const r2 = (12 / sp[0]) ** 2;
  let x0 = nx, x1 = 0, y0 = ny, y1 = 0;
  for (let z = 0; z < nz; z++) {
    const m = mask.subarray(z * P, (z + 1) * P); let any = false;
    for (let p = 0; p < P; p++) if (m[p]) { any = true; const y = (p / nx) | 0, x = p - y * nx; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    if (!any) continue;
    const inv = new Uint8Array(P); for (let p = 0; p < P; p++) inv[p] = m[p] ? 0 : 1;
    const d = sqdist(inv, nx, ny), sm = boxMean(hu.subarray(z * P, (z + 1) * P), nx, ny, Math.max(1, Math.round(1.7 / sp[0])));
    for (let p = 0; p < P; p++) {
      const q = z * P + p;
      if (labels[q] === 3) cand[q] = 1;
      else if (m[p] && d[p] > r2 && sm[p] >= -15 && sm[p] <= 8) cand[q] = 1;
    }
  }
  // 3D components (6-neighbour)
  const lab = new Int32Array(P * nz), stack = new Int32Array(P * nz), keep = new Uint8Array(P * nz);
  const vox = sp[0] * sp[1] * sp[2] / 1000, cx = (x0 + x1) / 2, w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
  let n = 0;
  for (let p0 = 0; p0 < cand.length; p0++) {
    if (!cand[p0] || lab[p0]) continue;
    n++; let s = 0, size = 0, mx = 0, my = 0; const members = [];
    stack[s++] = p0; lab[p0] = n;
    while (s) {
      const q = stack[--s]; size++; members.push(q);
      const z = (q / P) | 0, r = q - z * P, y = (r / nx) | 0, x = r - y * nx; mx += x; my += y;
      if (x > 0 && cand[q - 1] && !lab[q - 1]) { lab[q - 1] = n; stack[s++] = q - 1; }
      if (x < nx - 1 && cand[q + 1] && !lab[q + 1]) { lab[q + 1] = n; stack[s++] = q + 1; }
      if (y > 0 && cand[q - nx] && !lab[q - nx]) { lab[q - nx] = n; stack[s++] = q - nx; }
      if (y < ny - 1 && cand[q + nx] && !lab[q + nx]) { lab[q + nx] = n; stack[s++] = q + nx; }
      if (z > 0 && cand[q - P] && !lab[q - P]) { lab[q - P] = n; stack[s++] = q - P; }
      if (z < nz - 1 && cand[q + P] && !lab[q + P]) { lab[q + P] = n; stack[s++] = q + P; }
    }
    mx /= size; my /= size;
    const central = Math.abs(mx - cx) < 0.22 * w && my > y0 + 0.15 * h && my < y0 + 0.85 * h;
    if (size * vox >= 1 && central) for (const q of members) keep[q] = 1;
  }
  return keep;
}

// Everything the 3D view needs for one scan: surfaces in world coordinates plus the centre used.
export function buildSurfaces(g, mask, labels, { ventr = true } = {}) {
  const t0 = performance.now();
  const c = centroid(mask, g);
  const fBrain = Math.max(1, Math.round(2 / g.sp[0]));                            // about 2 mm blocks for the brain
  const fBlood = Math.max(1, Math.round(1 / g.sp[0]));                             // about 1 mm blocks for blood
  const uz = Math.max(1, Math.round(g.sp[2] / 1.25));                            // slices to about 1.25 mm
  // interpolate the 5 mm slices first, then smooth in millimetres so the slab steps are rounded off
  const surf = (src, test, f, sigma, iso) => toWorld(marchingCubes(blur(upsampleZ(blockField(src, g, test, f), uz), sigma), iso), c);
  const out = { centre: c, layers: {} };
  out.layers.brain = surf(mask, v => v > 0, fBrain, [2.2, 2.2, 2.8], 0.5);
  out.layers.SAH = surf(labels, v => v === 1, fBlood, [0.75, 0.75, 1.9], 0.3);
  out.layers.IPH = surf(labels, v => v === 2, fBlood, [0.9, 0.9, 1.9], 0.35);
  out.layers.IVH = surf(labels, v => v === 3, fBlood, [0.9, 0.9, 1.9], 0.35);
  if (ventr) {
    const v = ventricles(g, mask, labels);
    let nv = 0; for (let i = 0; i < v.length; i++) nv += v[i];
    out.ventricleMl = nv * g.sp[0] * g.sp[1] * g.sp[2] / 1000;
    out.layers.vent = surf(v, x => x > 0, fBlood, [1.2, 1.2, 2.2], 0.45);
  }
  out.ms = performance.now() - t0;
  return out;
}
