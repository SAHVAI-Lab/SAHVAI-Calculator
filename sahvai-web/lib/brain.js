// Brain (intracranial cavity) extraction for non-contrast head CT. The BrainHemoAI
// repository does not include its brain-extraction step, so this is a simple
// threshold-and-topology method that targets the same region as the demo mask
// (everything inside the inner table of the skull):
//   1. per slice: bone = HU >= 170; close small gaps in the skull ring (disk ~1.5 mm);
//      the intracranial candidate is soft tissue (-15..130 HU) that cannot be reached
//      from the image border without crossing bone
//   2. keep the largest 3D connected region after a ~2 mm erosion (cuts thin bridges
//      through foramina and the skull base), then grow it back inside the candidate
//   3. fill holes per slice
// followed by rectify_brain_mask() from the upstream code (-20..100 HU, fill holes
// < 10000 px, drop pieces < 1000 px) and removal of calcification (>= 120 HU).
import { floodFromBorder, dilate, erode, fillHoles, largest3d, removeSmallHoles, removeSmallObjects } from './morph.js';

export function extractBrain(g, log = () => {}, progress = () => {}) {
  const t0 = performance.now();
  const { nx, ny, nz, hu, sp } = g, P = nx * ny;
  const rBall = Math.max(2, Math.round(2.5 / sp[0])), rCut = Math.max(1, Math.round(2 / sp[0]));
  const cand = new Uint8Array(P * nz), core = new Uint8Array(P * nz);
  for (let z = 0; z < nz; z++) {
    const o = z * P;
    // soft tissue / fluid; bone, air and fat are barriers
    const pass = new Uint8Array(P), air = new Uint8Array(P);
    for (let p = 0; p < P; p++) { const v = hu[o + p]; pass[p] = v >= -40 && v <= 130 ? 1 : 0; air[p] = v < -200 ? 1 : 0; }
    // extracranial soft tissue: reachable by a ~5 mm ball from the skin surface (or the image
    // edge when the field of view cuts the head) without passing bone, air or fat
    const outside = floodFromBorder(air, nx, ny);
    for (let x = 0; x < nx; x++) { outside[x] = 1; outside[(ny - 1) * nx + x] = 1; }
    for (let y = 0; y < ny; y++) { outside[y * nx] = 1; outside[y * nx + nx - 1] = 1; }
    const near = dilate(outside, nx, ny, rBall + 3), pcore = erode(pass, nx, ny, rBall);
    const seedPass = new Uint8Array(P);
    for (let p = 0; p < P; p++) seedPass[p] = pcore[p];
    const reached = floodSeeds(seedPass, near, nx, ny);
    const reachedFull = dilate(reached, nx, ny, rBall);
    const encl = enclosure(hu.subarray(o, o + P), nx, ny, sp[0]);
    const c = new Uint8Array(P);
    for (let p = 0; p < P; p++) { const v = hu[o + p]; c[p] = pass[p] && !reachedFull[p] && v >= -15 && encl[p] ? 1 : 0; }
    cand.set(c, o);
    core.set(erode(c, nx, ny, rCut), o);
    progress(0.5 * (z + 1) / nz);
  }
  const keep = largest3d(core, nx, ny, nz);
  const mask = new Uint8Array(P * nz);
  for (let z = 0; z < nz; z++) {
    const o = z * P, k = keep.subarray(o, o + P);
    let any = 0; for (let p = 0; p < P; p++) any |= k[p];
    if (!any) continue;
    const grown = dilate(k, nx, ny, rCut + 1), m = new Uint8Array(P);
    for (let p = 0; p < P; p++) m[p] = grown[p] && cand[o + p] ? 1 : 0;
    mask.set(fillHoles(m, nx, ny), o);
    progress(0.5 + 0.5 * (z + 1) / nz);
  }
  const out = rectify(hu, mask, nx, ny, nz);
  let n = 0; for (let p = 0; p < out.length; p++) n += out[p];
  log(`brain extraction: ${(n * sp[0] * sp[1] * sp[2] / 1000).toFixed(0)} mL intracranial in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return out;
}

// Skull enclosure test on a 4x coarser grid: from each block cast 16 rays (up to 120 mm);
// a ray scores if it meets bone (>= 170 HU) before air or fat (< -40 HU) or the image edge.
// Intracranial tissue is walled in by bone in nearly every direction; face and neck
// muscles are not. Returns a full-resolution mask of blocks with >= 12 of 16 rays scoring.
export function enclosure(sl, nx, ny, px) {
  const B = 4, bx = Math.ceil(nx / B), by = Math.ceil(ny / B), cls = new Uint8Array(bx * by);   // 0 soft, 1 bone, 2 miss
  for (let j = 0; j < by; j++) for (let i = 0; i < bx; i++) {
    let bone = false, miss = false;
    for (let y = j * B; y < Math.min(ny, j * B + B); y++) for (let x = i * B; x < Math.min(nx, i * B + B); x++) {
      const v = sl[y * nx + x]; if (v >= 170) bone = true; else if (v < -40) miss = true;
    }
    cls[j * bx + i] = bone ? 1 : miss ? 2 : 0;
  }
  const R = 16, steps = Math.ceil(120 / (px * B)), dirs = [];
  for (let k = 0; k < R; k++) dirs.push([Math.cos(2 * Math.PI * k / R), Math.sin(2 * Math.PI * k / R)]);
  const ok = new Uint8Array(bx * by);
  for (let j = 0; j < by; j++) for (let i = 0; i < bx; i++) {
    if (cls[j * bx + i] !== 0) continue;
    let hits = 0;
    for (let k = 0; k < R && hits + (R - k) >= 12; k++) {
      const [dx, dy] = dirs[k];
      for (let s = 1; s <= steps; s++) {
        const x = Math.round(i + dx * s), y = Math.round(j + dy * s);
        if (x < 0 || y < 0 || x >= bx || y >= by) break;
        const c = cls[y * bx + x];
        if (c === 1) { hits++; break; }
        if (c === 2) break;
      }
    }
    ok[j * bx + i] = hits >= 12 ? 1 : 0;
  }
  // grow by one block so tissue in blocks that also hold bone (the brain rim) is kept
  const okG = new Uint8Array(bx * by);
  for (let j = 0; j < by; j++) for (let i = 0; i < bx; i++) {
    let v = 0;
    for (let b = Math.max(0, j - 1); b <= Math.min(by - 1, j + 1) && !v; b++) for (let a = Math.max(0, i - 1); a <= Math.min(bx - 1, i + 1); a++) if (ok[b * bx + a]) { v = 1; break; }
    okG[j * bx + i] = v;
  }
  const out = new Uint8Array(nx * ny);
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) out[y * nx + x] = okG[((y / B) | 0) * bx + ((x / B) | 0)];
  return out;
}

// flood (4-neighbour) through pass[p] = 1 starting from every pass pixel with seed[p] = 1
function floodSeeds(pass, seed, w, h) {
  const seen = new Uint8Array(w * h), stack = new Int32Array(w * h); let sp = 0;
  for (let p = 0; p < w * h; p++) if (pass[p] && seed[p]) { seen[p] = 1; stack[sp++] = p; }
  while (sp) {
    const q = stack[--sp], x = q % w;
    const nb = [x > 0 ? q - 1 : -1, x < w - 1 ? q + 1 : -1, q >= w ? q - w : -1, q + w < w * h ? q + w : -1];
    for (const r of nb) if (r >= 0 && pass[r] && !seen[r]) { seen[r] = 1; stack[sp++] = r; }
  }
  return seen;
}

// utils/data_orgnizer.py: rectify_brain_mask + "remove calcified regions"
export function rectify(hu, mask, nx, ny, nz) {
  const P = nx * ny, out = new Uint8Array(P * nz);
  for (let z = 0; z < nz; z++) {
    const o = z * P, f = new Uint8Array(P);
    for (let p = 0; p < P; p++) { const v = hu[o + p]; f[p] = mask[o + p] && v >= -20 && v <= 100 ? 1 : 0; }
    const m = removeSmallObjects(removeSmallHoles(f, nx, ny, 10000), nx, ny, 1000);
    for (let p = 0; p < P; p++) out[o + p] = m[p] && hu[o + p] < 120 ? 1 : 0;
  }
  return out;
}
