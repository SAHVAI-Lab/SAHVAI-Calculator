// ICH expansion signs measured on the SAHVAI working grid (0.5 x 0.5 mm in plane, 5 mm slices;
// x runs to the patient's Left, y to Posterior, z to Superior) from the SAHVAI (BrainHemoAI) labels
// (1 SAH, 2 IPH, 3 IVH), the HU values and the brain mask.
//
// Non-contrast CT signs follow the published definitions, as standardised by Morotti et al. (Ann Neurol 2019;86:480-492):
//   hypodensity    any hypodense region strictly encapsulated within the hematoma (Boulouis et al., JAMA Neurol 2016)
//   black hole     encapsulated hypodense region >= 28 HU below the surrounding hematoma (Li et al., Stroke 2016)
//   swirl          encapsulated region iso- or hypodense to brain within the hyperdense hematoma (Selariu et al., BMC Neurol 2012)
//   blend          hypodense region next to the hyperdense part, not encapsulated, >= 18 HU difference (Li et al., Stroke 2015)
//   fluid level    hypodense (supernatant) region above a hyperdense (dependent) one with a straight interface
//   heterogeneous  >= 3 hypodense foci in the hematoma on its largest axial section (Barras density III-V)
//   irregular      >= 2 focal margin irregularities, joined to or separate from the edge (Barras shape III-V)
//   island         >= 3 small hematomas separate from the main one, or >= 4 some of which connect to it, on one section (Li et al., Stroke 2017)
//   satellite      small hematoma < 10 mm separated from the main one by 1-20 mm on at least one section (Shimoda et al., AJNR 2017)
// Contrast signs are measured only when the scan is contrast-enhanced: spot sign candidates (>= 120 HU foci inside the
// hematoma, not touching its edge; Delgado Almandoz et al., Stroke 2009/2010). Leakage and iodine signs need a second
// (delayed) phase or dual-energy iodine maps and are reader-entered.
// Every number here is an operational translation of a visual definition: the output is a set of candidates for a reader.

import { label2d, fillHoles, sqdist, dilate, erode } from './morph.js';

export const CRITERIA = {
  minHematoma_mL: 1,          // smaller main hematomas are not analysed
  seedDepth_mm: 1.5,          // a hypodense region must reach this far inside the hematoma edge (partial-volume rim excluded)
  growLimit_mm: 1.5,          // it is then grown this far through hypodense pixels to see whether it opens to the edge
  candidate_dHU: 12,          // a pixel is hypodense when >= 12 HU below the hematoma reference (smoothed)
  minHU: 12,                  // below this a region is CSF / air, not blood
  hypodensity_dHU: 15, blackHole_dHU: 28, blend_dHU: 18, fluid_dHU: 15,
  minRegion_mm2: 7, minWidth_mm: 2,
  blendMinArea_mm2: 15, blendMinAboveBrain_HU: 8, blendMaxShare: 0.6, blendMinInterface_mm: 5, blendMinEdge_HUmm: 4,
  densityMinSection: 0.3,     // density signs are read on sections with >= 30% of the largest section's area (end slices are partial-volumed)
  swirlIsoMargin_HU: 8, swirlMinSlices: 2,
  fluidMaxTilt_deg: 10, fluidMaxRms_mm: 0.75, fluidMinLen_mm: 8, fluidMinShare: 0.5,
  heteroMinFoci: 3, heteroFocus_mm2: 2, heteroFocus_dHU: 12,
  irregularMinMargins: 2, lobeMin_mm2: 4, lobeMinDepth_mm: 1, islandLobeMax_mm2: 80, islandNeckRatio: 0.8,
  satelliteMaxDiam_mm: 10, satelliteGap_mm: [1, 20],
  islandSeparate: 3, islandTotal: 4, islandMaxArea_mm2: 80, islandMaxGap_mm: 20,
  focusMin_mm2: 2, focusMinHU: 50, focusAboveBrain_HU: 15, focusMaxHU: 100, focusCalcHU: 110, focusSkull_mm: 4, focusMaxElong: 3, focusMetal_mm: 3, metalHU: 2000, metalMinPx: 5, catheterHU: 1000,
  spotHU: 120, spotMinDiam_mm: 1.5, contrast_mL: 2,
  calcHU: 150, calcMaxHU: 800, calcNear_mm: 5,   // NCCT calcification in or near the clot (a spot-sign mimic on CTA; Gazzola et al., Stroke 2008)
};

// overlay labels written into the result's label map
export const OVERLAY = {
  1: { key: 'Hematoma', name: 'main hematoma', color: [255, 45, 120] },
  2: { key: 'Other blood', name: 'other blood (SAH / other IPH)', color: [255, 176, 0] },
  3: { key: 'IVH', name: 'intraventricular blood', color: [30, 200, 255] },
  4: { key: 'Hypodensity', name: 'encapsulated hypodensity', color: [120, 190, 255] },
  5: { key: 'Black hole', name: 'black hole (>= 28 HU)', color: [40, 70, 235] },
  6: { key: 'Blend', name: 'blend sign (hypodense part)', color: [40, 210, 120] },
  7: { key: 'Fluid level', name: 'fluid level (supernatant)', color: [255, 235, 40] },
  8: { key: 'Satellite / island', name: 'separate small hematoma', color: [255, 120, 0] },
  9: { key: 'Spot?', name: 'spot sign candidate (contrast)', color: [255, 0, 0] },
};

// ---------- small helpers ----------
const sum = a => { let s = 0; for (const v of a) s += v; return s; };
function smooth(src, w, h) {   // separable binomial [1 4 6 4 1] / 16 (sigma ~ 1 pixel), edges clamped
  const k = [1, 4, 6, 4, 1], tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0; for (let i = -2; i <= 2; i++) { const X = Math.min(w - 1, Math.max(0, x + i)); s += k[i + 2] * src[y * w + X]; } tmp[y * w + x] = s / 16;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0; for (let i = -2; i <= 2; i++) { const Y = Math.min(h - 1, Math.max(0, y + i)); s += k[i + 2] * tmp[Y * w + x]; } out[y * w + x] = s / 16;
  }
  return out;
}
function percentile(vals, q) {
  if (!vals.length) return NaN;
  const a = Float32Array.from(vals).sort(), i = (a.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
}
// shape of a 2D pixel set: centroid, principal axes (mm), elongation
function moments(px, w, sx, sy) {
  let mx = 0, my = 0; for (const p of px) { mx += p % w; my += (p - p % w) / w; } mx /= px.length; my /= px.length;
  let xx = 0, yy = 0, xy = 0;
  for (const p of px) { const x = (p % w - mx) * sx, y = ((p - p % w) / w - my) * sy; xx += x * x; yy += y * y; xy += x * y; }
  xx = xx / px.length + sx * sx / 12; yy = yy / px.length + sy * sy / 12; xy /= px.length;
  const tr = xx + yy, det = xx * yy - xy * xy, d = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + d, l2 = Math.max(1e-6, tr / 2 - d);
  return { cx: mx, cy: my, major: 4 * Math.sqrt(l1), minor: 4 * Math.sqrt(l2), elong: Math.sqrt(l1 / l2) };
}
// 3D components with 26-connectivity
function label3d(m, nx, ny, nz) {
  const N = nx * ny * nz, P = nx * ny, lab = new Int32Array(N), stack = new Int32Array(N), sizes = [0];
  let n = 0;
  for (let p = 0; p < N; p++) {
    if (!m[p] || lab[p]) continue;
    n++; let sp = 0, size = 0; stack[sp++] = p; lab[p] = n;
    while (sp) {
      const q = stack[--sp]; size++;
      const z = Math.floor(q / P), r = q - z * P, y = Math.floor(r / nx), x = r - y * nx;
      for (let dz = -1; dz <= 1; dz++) { const Z = z + dz; if (Z < 0 || Z >= nz) continue;
        for (let dy = -1; dy <= 1; dy++) { const Y = y + dy; if (Y < 0 || Y >= ny) continue;
          for (let dx = -1; dx <= 1; dx++) { const X = x + dx; if (X < 0 || X >= nx) continue;
            const t = Z * P + Y * nx + X; if (m[t] && !lab[t]) { lab[t] = n; stack[sp++] = t; } } } }
    }
    sizes.push(size);
  }
  return { lab, n, sizes };
}

// ---------- main entry ----------
// g: {nx, ny, nz, sp, hu}; labels, raw (before small fragments were removed; may be null), mask: Uint8 on the grid
// opts: { includeIVH: false }
export function analyseSigns(g, labels, raw, mask, opts = {}) {
  const C = { ...CRITERIA, ...(opts.criteria || {}) };
  const { nx, ny, nz, sp, hu } = g, P = nx * ny, N = P * nz, [sx, sy, sz] = sp;
  const pxA = sx * sy, vox = pxA * sz / 1000, mm = d => d / Math.sqrt(pxA);   // mm -> pixels (in plane)
  raw = raw || labels;
  const overlay = new Uint8Array(N);
  const out = { criteria: C, overlay, signs: {}, regions: [], warnings: [], hematoma: null, status: 'ok' };

  // ---- blood types and the main hematoma (largest 26-connected IPH component)
  let vSAH = 0, vIPH = 0, vIVH = 0;
  const hem = new Uint8Array(N);
  for (let p = 0; p < N; p++) {
    const l = labels[p];
    if (l === 1) vSAH++; else if (l === 2) vIPH++; else if (l === 3) vIVH++;
    if (l === 2 || (opts.includeIVH && l === 3)) hem[p] = 1;
  }
  vSAH *= vox; vIPH *= vox; vIVH *= vox;
  const L3 = label3d(hem, nx, ny, nz);
  let best = 0; for (let i = 1; i <= L3.n; i++) if (L3.sizes[i] > (L3.sizes[best] || 0)) best = i;
  const main = new Uint8Array(N); let nMain = 0;
  if (best) for (let p = 0; p < N; p++) if (L3.lab[p] === best) { main[p] = 1; nMain++; }
  const mainVol = nMain * vox;
  for (let p = 0; p < N; p++) overlay[p] = main[p] ? 1 : labels[p] === 3 ? 3 : labels[p] ? 2 : 0;
  out.volumes = { SAH: vSAH, IPH: vIPH, IVH: vIVH, main: mainVol, iphComponents: L3.n };

  // ---- references: normal brain and the hyperdense hematoma
  const brainVals = [];
  for (let p = 0; p < N; p += 7) if (mask[p] && !labels[p] && hu[p] > 15 && hu[p] < 50) brainVals.push(hu[p]);
  const brainRef = percentile(brainVals, 0.5);

  // contrast-enhanced? many voxels of 130-600 HU inside the brain (vessels) outside the hematoma
  let hot = 0;
  for (let k = 0; k < nz; k++) {
    let any = false; for (let p = k * P; p < (k + 1) * P; p++) if (mask[p]) { any = true; break; }
    if (!any) continue;
    const core = erode(mask.subarray(k * P, (k + 1) * P), nx, ny, Math.round(mm(3)));
    for (let p = 0; p < P; p++) { const q = k * P + p; if (core[p] && !labels[q] && hu[q] >= 130 && hu[q] <= 600) hot++; }
  }
  const contrast_mL = hot * vox, contrast = contrast_mL >= C.contrast_mL;
  out.contrast = { enhanced: contrast, vessel_mL: contrast_mL };

  if (mainVol < C.minHematoma_mL) {
    out.status = 'no-ich';
    out.hematoma = { volume_mL: mainVol, brainRef };
    return finish(out, C);
  }
  if (vSAH > Math.max(5, 0.5 * vIPH)) out.warnings.push(`The model also labelled ${vSAH.toFixed(1)} mL of subarachnoid blood: island and satellite counts may include subarachnoid blood, and these signs were defined for spontaneous ICH.`);

  // ---- working box: the main hematoma plus 25 mm, slices it spans
  let x0 = nx, x1 = -1, y0 = ny, y1 = -1, z0 = nz, z1 = -1;
  for (let k = 0; k < nz; k++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) if (main[k * P + y * nx + x]) {
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; if (k < z0) z0 = k; if (k > z1) z1 = k;
  }
  const marg = Math.ceil(mm(25));
  x0 = Math.max(0, x0 - marg); x1 = Math.min(nx - 1, x1 + marg); y0 = Math.max(0, y0 - marg); y1 = Math.min(ny - 1, y1 + marg);
  const w = x1 - x0 + 1, h = y1 - y0 + 1, Q = w * h;
  const sub = (arr, k, T = arr.constructor) => { const o = new T(Q); for (let y = 0; y < h; y++) o.set(arr.subarray(k * P + (y + y0) * nx + x0, k * P + (y + y0) * nx + x0 + w), y * w); return o; };
  const toGrid = (k, q) => k * P + (Math.floor(q / w) + y0) * nx + x0 + q % w;

  // reference density of the hematoma: 75th percentile of the smoothed HU in its core
  const coreVals = [], sliceData = [];
  for (let k = z0; k <= z1; k++) {
    const M = sub(main, k), area = sum(M);
    const H = sub(hu, k, Float32Array), S = smooth(H, w, h);
    const mkf = fillHoles(dilate(sub(mask, k), w, h, 2), w, h);
    sliceData[k] = { M, H, S, area, mk: mkf };
    if (area * pxA < 20) continue;
    const core = erode(M, w, h, Math.round(mm(1)));
    for (let q = 0; q < Q; q++) if (core[q]) coreVals.push(S[q]);
  }
  const hemRef = percentile(coreVals, 0.75), allVals = [];
  for (let k = z0; k <= z1; k++) { const { M, H } = sliceData[k]; for (let q = 0; q < Q; q++) if (M[q]) allVals.push(H[q]); }
  const meanHU = sum(allVals) / allVals.length, sdHU = Math.sqrt(allVals.reduce((s, v) => s + (v - meanHU) ** 2, 0) / allVals.length);

  // max-area slice
  let zMax = z0; for (let k = z0; k <= z1; k++) if (sliceData[k].area > sliceData[zMax].area) zMax = k;

  // ---- density: hypodense regions, slice by slice
  const maxArea = sliceData[zMax].area, metalSlices = [], catheterSlices = [], denseMetal = new Uint8Array(nz);
  for (let k = z0; k <= z1; k++) { let n = 0; for (let p = k * P; p < (k + 1) * P; p++) if (hu[p] >= C.metalHU && mask[p]) n++;
    // the brain mask excludes the metal itself, so also count dense voxels enclosed by it
    if (n < C.metalMinPx) { const fm = fillHoles(dilate(mask.subarray(k * P, (k + 1) * P), nx, ny, 2), nx, ny); n = 0; for (let p = 0; p < P; p++) if (fm[p] && hu[k * P + p] >= C.metalHU) n++; }
    denseMetal[k] = n >= C.metalMinPx ? 1 : 0; }
  const regions = [], seedPx = Math.max(1, Math.round(mm(C.seedDepth_mm))), growPx = Math.max(1, Math.round(mm(C.growLimit_mm)));
  for (let k = z0; k <= z1; k++) {
    const d = sliceData[k]; if (d.area * pxA < 20) continue;
    const { M, H, S } = d, E = fillHoles(M, w, h);
    d.E = E;
    if (d.area < C.densityMinSection * maxArea) continue;
    // dense metal (coil, clip) anywhere in this section's brain: its streak artefacts cross the whole section and mimic hypodensities, so it is not read;
    // a catheter (about 1000-1500 HU on 5 mm slabs) gives milder streaks and is only flagged
    const met = new Uint8Array(Q), cat = new Uint8Array(Q); let nm = 0, nc = 0;
    for (let q = 0; q < Q; q++) if (d.mk[q]) { if (H[q] >= C.metalHU) { met[q] = 1; nm++; } if (H[q] >= C.catheterHU) { cat[q] = 1; nc++; } }
    if (nm >= C.metalMinPx || denseMetal[k]) { metalSlices.push(k); continue; }
    if (nc && sqdistMin(cat, E, w, h) <= mm(15) ** 2) catheterSlices.push(k);
    // this section's hyperdense reference (its partial-volume rim excluded); the whole-clot value caps it
    const cv = [], cr = erode(M, w, h, Math.round(mm(1))); for (let q = 0; q < Q; q++) if (cr[q]) cv.push(S[q]);
    const ref = Math.min(hemRef + 5, cv.length > 20 ? percentile(cv, 0.75) : hemRef);
    let eArea = 0; for (let q = 0; q < Q; q++) eArea += E[q];
    const notE = new Uint8Array(Q); for (let q = 0; q < Q; q++) notE[q] = E[q] ? 0 : 1;
    const depth2 = sqdist(notE, w, h);   // squared distance of each pixel to the nearest pixel outside the hematoma
    const cand = new Uint8Array(Q), seed = new Uint8Array(Q);
    for (let q = 0; q < Q; q++) if (E[q] && S[q] <= ref - C.candidate_dHU && S[q] >= C.minHU) { cand[q] = 1; if (depth2[q] >= seedPx * seedPx) seed[q] = 1; }
    d.cand = cand; d.depth2 = depth2;
    const { lab, n, sizes } = label2d(seed, w, h, true);
    if (!n) continue;
    const members = Array.from({ length: n + 1 }, () => []);
    for (let q = 0; q < Q; q++) if (lab[q]) members[lab[q]].push(q);
    const insideD = sqdist(seed.map(v => 1 - v), w, h);   // width check: distance of seed pixels to the nearest non-seed pixel
    for (let r = 1; r <= n; r++) {
      const px = members[r]; if (sizes[r] * pxA < C.focusMin_mm2) continue;
      // grow through hypodense pixels (at most growPx steps) to test whether the region opens to the hematoma edge
      const grown = new Set(px); let front = px.slice(), open = false;
      for (let step = 0; step < growPx && front.length && !open; step++) {
        const next = [];
        for (const q of front) {
          const x = q % w, y = (q - x) / w;
          for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= w || Y >= h) { open = true; continue; }
            const t = Y * w + X;
            if (!E[t]) { open = true; continue; }
            if (cand[t] && !grown.has(t)) { grown.add(t); next.push(t); }
          }
        }
        front = next;
      }
      if (!open) for (const q of grown) { const x = q % w, y = (q - x) / w;
        if ((x > 0 && !E[q - 1]) || (x < w - 1 && !E[q + 1]) || (y > 0 && !E[q - w]) || (y < h - 1 && !E[q + w])) { open = true; break; } }
      // the hyperdense ring around it (1-2 mm), inside the hematoma, not hypodense
      const reg = new Uint8Array(Q); for (const q of grown) reg[q] = 1;
      const d1 = dilate(reg, w, h, 1), d4 = dilate(reg, w, h, Math.max(2, Math.round(mm(2))));
      let rs = 0, rn = 0; for (let q = 0; q < Q; q++) if (d4[q] && !d1[q] && E[q] && !cand[q]) { rs += H[q]; rn++; }
      let ss = 0; for (const q of px) ss += H[q];
      const regHU = ss / px.length, ringHU = rn >= 8 ? rs / rn : ref, dHU = ringHU - regHU;
      // the interface with the hyperdense part: its length and the density gradient across it (HU per mm)
      let ifn = 0, grad = 0;
      for (const q of grown) { const x = q % w, y = (q - x) / w;
        for (const t of [x > 0 ? q - 1 : -1, x < w - 1 ? q + 1 : -1, y > 0 ? q - w : -1, y < h - 1 ? q + w : -1]) if (t >= 0 && E[t] && !cand[t] && !grown.has(t)) {
          const gx = (S[Math.min(Q - 1, q + 1)] - S[Math.max(0, q - 1)]) / (2 * sx), gy = (S[Math.min(Q - 1, q + w)] - S[Math.max(0, q - w)]) / (2 * sy);
          grad += Math.hypot(gx, gy); ifn++; break; } }
      let width = 0; for (const q of px) width = Math.max(width, insideD[q]);
      const area = grown.size * pxA, mo = moments(px, w, sx, sy);
      const R = { k, px, grown: [...grown], open, share: grown.size / eArea, interface_mm: ifn * Math.sqrt(pxA), edge_HUmm: ifn ? grad / ifn : 0, area_mm2: area, seed_mm2: px.length * pxA, regionHU: regHU, ringHU, dHU, width_mm: 2 * Math.sqrt(width) * Math.sqrt(pxA), cx: mo.cx, cy: mo.cy, flags: [] };
      if (regHU < C.minHU) continue;
      // fluid level: hypodense part anterior to a straight, near-horizontal interface with hyperdense blood behind it
      if (dHU >= C.fluid_dHU && area >= C.minRegion_mm2) R.fluid = fluidLevel(R.grown, E, cand, w, h, sx, sy, C);
      regions.push(R);
    }
  }

  // link regions across slices (pixel overlap on adjacent slices) so a region spanning slices counts once
  let nextId = 1;
  for (const R of regions) {
    const prev = regions.filter(o => o.k === R.k - 1 && o.id);
    for (const o of prev) { const s = new Set(o.grown); if (R.grown.some(q => s.has(q))) { R.id = R.id ?? o.id; } }
    if (!R.id) R.id = nextId++;
  }
  const slicesOf = id => [...new Set(regions.filter(r => r.id === id).map(r => r.k))].sort((a, b) => a - b);

  // classify
  for (const R of regions) {
    const big = R.area_mm2 >= C.minRegion_mm2 && R.width_mm >= C.minWidth_mm;
    if (!big) continue;
    if (!R.open && R.dHU >= C.hypodensity_dHU) R.flags.push('hypodensity');
    if (!R.open && R.dHU >= C.blackHole_dHU) R.flags.push('black hole');
    if (!R.open && R.dHU >= C.hypodensity_dHU && R.regionHU <= brainRef + C.swirlIsoMargin_HU && slicesOf(R.id).length >= C.swirlMinSlices) R.flags.push('swirl');
    if (R.open && R.dHU >= C.blend_dHU && R.area_mm2 >= C.blendMinArea_mm2 && R.share <= C.blendMaxShare && R.interface_mm >= C.blendMinInterface_mm
      && R.edge_HUmm >= C.blendMinEdge_HUmm && R.regionHU >= brainRef + C.blendMinAboveBrain_HU) R.flags.push('blend');
    if (R.fluid && R.fluid.ok) R.flags.push('fluid level');
  }
  for (const R of regions) {
    const f = R.flags, lab = f.includes('fluid level') ? 7 : f.includes('blend') ? 6 : f.includes('black hole') ? 5 : f.includes('hypodensity') ? 4 : 0;
    if (lab) for (const q of R.grown) overlay[toGrid(R.k, q)] = lab;
  }

  // ---- heterogeneity on the largest section: number of hypodense foci (Barras density, INTERACT2 rule)
  const heteroFoci = regions.filter(r => r.k === zMax && r.seed_mm2 >= C.heteroFocus_mm2 && r.dHU >= C.heteroFocus_dHU);

  // ---- morphology: lobules joined to the edge and small hematomas separate from it
  const brainCoreR = Math.round(mm(C.focusSkull_mm)), focusThr = Math.max(C.focusMinHU, brainRef + C.focusAboveBrain_HU);
  const perSlice = [];
  for (let k = z0; k <= z1; k++) {
    const d = sliceData[k]; if (!d.area) { perSlice[k] = null; continue; }
    const { M, H, S } = d, E = d.E || fillHoles(M, w, h);
    // lobules: what an opening with a disk removes from the section (bubble- or sprout-like parts)
    const eqR = Math.sqrt(d.area / Math.PI), rO = Math.max(Math.round(mm(2)), Math.min(Math.round(mm(8)), Math.round(0.35 * eqR)));
    const O = dilate(erode(E, w, h, rO), w, h, rO), lob = new Uint8Array(Q);
    for (let q = 0; q < Q; q++) if (E[q] && !O[q]) lob[q] = 1;
    const dO = sqdist(O, w, h), L = label2d(lob, w, h, false), lobules = [];
    const lm = Array.from({ length: L.n + 1 }, () => []); for (let q = 0; q < Q; q++) if (L.lab[q]) lm[L.lab[q]].push(q);
    for (let r = 1; r <= L.n; r++) {
      const px = lm[r]; let dep = 0; for (const q of px) dep = Math.max(dep, dO[q]);
      const depth = Math.sqrt(dep) * Math.sqrt(pxA);
      if (px.length * pxA >= Math.max(C.lobeMin_mm2, 0.005 * d.area * pxA) && depth >= C.lobeMinDepth_mm + rO * Math.sqrt(pxA) * 0.3) {
        // bubble- or sprout-like: small, and joined to the clot by a neck narrower than itself
        let neck = 0; for (const q of px) { const x = q % w, y = (q - x) / w; if ((x > 0 && O[q - 1]) || (x < w - 1 && O[q + 1]) || (y > 0 && O[q - w]) || (y < h - 1 && O[q + w])) neck++; }
        const eqD = 2 * Math.sqrt(px.length / Math.PI);
        lobules.push({ px, area_mm2: px.length * pxA, depth_mm: depth, island: px.length * pxA <= C.islandLobeMax_mm2 && neck <= C.islandNeckRatio * eqD });
      }
    }
    // separate small hematomas: other blood labels (before fragment removal) or compact hyperdense foci in the brain
    const mk = sub(mask, k), core = erode(mk, w, h, brainCoreR), Rw = sub(raw, k), Lb = sub(labels, k);
    const nearE = dilate(E, w, h, 1), cand = new Uint8Array(Q), dense = new Uint8Array(Q);
    for (let q = 0; q < Q; q++) if (H[q] > 200) dense[q] = 1;
    const nearDense = dilate(dense, w, h, Math.round(mm(C.focusMetal_mm)));
    for (let q = 0; q < Q; q++) {
      if (nearE[q] || nearDense[q] || Lb[q] === 3 || Rw[q] === 3) continue;
      if ((Rw[q] === 1 || Rw[q] === 2 || Lb[q] === 1 || Lb[q] === 2) && mk[q]) cand[q] = 1;
      else if (core[q] && !denseMetal[k] && S[q] >= focusThr && S[q] <= C.focusMaxHU) cand[q] = 1;   // not on metal-streaked sections
    }
    const dE = sqdist(E, w, h), F = label2d(cand, w, h, true), foci = [];
    const fm = Array.from({ length: F.n + 1 }, () => []); for (let q = 0; q < Q; q++) if (F.lab[q]) fm[F.lab[q]].push(q);
    for (let r = 1; r <= F.n; r++) {
      const px = fm[r], area = px.length * pxA; if (area < C.focusMin_mm2 || area > C.islandMaxArea_mm2) continue;
      let gap = Infinity, mx = -Infinity, s = 0; for (const q of px) { gap = Math.min(gap, dE[q]); mx = Math.max(mx, H[q]); s += H[q]; }
      gap = Math.sqrt(gap) * Math.sqrt(pxA) - Math.sqrt(pxA) / 2;   // edge-to-edge separation in mm
      if (gap > C.islandMaxGap_mm || mx > C.focusCalcHU || s / px.length < Math.min(focusThr, 45)) continue;
      const mo = moments(px, w, sx, sy); if (mo.elong > C.focusMaxElong) continue;
      foci.push({ px, area_mm2: area, gap_mm: gap, diam_mm: mo.major, meanHU: s / px.length, maxHU: mx, cx: mo.cx, cy: mo.cy });
    }
    perSlice[k] = { lobules, foci, rO };
  }
  const satellites = [], islandSlices = [];
  for (let k = z0; k <= z1; k++) {
    const s = perSlice[k]; if (!s) continue;
    for (const f of s.foci) if (f.diam_mm < C.satelliteMaxDiam_mm && f.gap_mm >= C.satelliteGap_mm[0] && f.gap_mm <= C.satelliteGap_mm[1]) satellites.push({ k, ...f });
    const sep = s.foci.length, con = s.lobules.filter(l => l.island).length, tot = sep + con;
    if (sep >= C.islandSeparate || tot >= C.islandTotal) islandSlices.push({ k, separate: sep, connected: con });
  }
  for (const f of satellites) for (const q of f.px) overlay[toGrid(f.k, q)] = 8;
  for (const s of islandSlices) for (const f of perSlice[s.k].foci) for (const q of f.px) overlay[toGrid(s.k, q)] = 8;
  const shapeMax = perSlice[zMax], margins = shapeMax ? shapeMax.lobules.length + shapeMax.foci.filter(f => f.gap_mm <= 10).length : 0;

  // solidity of the largest section (area / convex hull area), reported only
  if (metalSlices.length) out.warnings.push(`Dense metal (a coil or clip?) is in the brain on slice${metalSlices.length > 1 ? 's' : ''} ${metalSlices.map(k => k + 1).join(', ')}: density signs were not read there because streak artefacts mimic hypodensities.`);
  if (catheterSlices.length) out.warnings.push(`A catheter or other dense object (≥ ${C.catheterHU} HU) lies within 15 mm of the hematoma on slice${catheterSlices.length > 1 ? 's' : ''} ${catheterSlices.map(k => k + 1).join(', ')}: check density findings there for streak artefact.`);
  const solidity = convexSolidity(sliceData[zMax].M, w, h);

  // ---- NCCT calcification in or within 5 mm of the clot: on a CTA it would mimic a spot sign (Gazzola et al., Stroke 2008)
  const calcs = [];
  if (!contrast) for (let k = z0; k <= z1; k++) {
    const d = sliceData[k]; if (!d.area || denseMetal[k]) continue;
    // denser than 800 HU on a 5 mm slab, or within 3 mm of such a voxel: a catheter, coil or bone, not a calcification
    const E = d.E || fillHoles(d.M, w, h), near = dilate(E, w, h, Math.round(mm(C.calcNear_mm))), cm = new Uint8Array(Q), dense = new Uint8Array(Q);
    for (let q = 0; q < Q; q++) if (d.H[q] >= C.calcMaxHU) dense[q] = 1;
    const nearDense = dilate(dense, w, h, Math.round(mm(3)));
    for (let q = 0; q < Q; q++) if (near[q] && !nearDense[q] && d.H[q] >= C.calcHU) cm[q] = 1;
    const T = label2d(cm, w, h, true), tm = Array.from({ length: T.n + 1 }, () => []);
    for (let q = 0; q < Q; q++) if (T.lab[q]) tm[T.lab[q]].push(q);
    for (let r = 1; r <= T.n; r++) { const px = tm[r]; if (px.length < 2) continue; let mx = 0; for (const q of px) mx = Math.max(mx, d.H[q]);
      const mo = moments(px, w, sx, sy); calcs.push({ slice: k, x: Math.round(mo.cx + x0), y: Math.round(mo.cy + y0), maxHU: Math.round(mx), inside: px.some(q => E[q]) }); }
  }
  out.calcifications = calcs;

  // ---- contrast: spot sign candidates inside the hematoma
  const spots = [];
  if (contrast) {
    for (let k = z0; k <= z1; k++) {
      const d = sliceData[k]; if (!d.area) continue;
      const E = d.E || fillHoles(d.M, w, h), inner = erode(E, w, h, 1), hotm = new Uint8Array(Q);
      for (let q = 0; q < Q; q++) if (E[q] && d.H[q] >= C.spotHU) hotm[q] = 1;
      const T = label2d(hotm, w, h, true), tm = Array.from({ length: T.n + 1 }, () => []);
      for (let q = 0; q < Q; q++) if (T.lab[q]) tm[T.lab[q]].push(q);
      for (let r = 1; r <= T.n; r++) {
        const px = tm[r]; if (!px.every(q => inner[q])) continue;   // touching the edge: may be a vessel running into the clot
        const mo = moments(px, w, sx, sy), eq = 2 * Math.sqrt(px.length * pxA / Math.PI); if (eq < C.spotMinDiam_mm) continue;
        let mx = -Infinity; for (const q of px) mx = Math.max(mx, d.H[q]);
        spots.push({ k, px, maxHU: mx, diam_mm: Math.max(eq, mo.major), cx: mo.cx, cy: mo.cy });
      }
    }
    for (const s of spots) for (const q of s.px) overlay[toGrid(s.k, q)] = 9;
  }

  // ---- assemble
  const pos = (R, x) => ({ slice: R.k, x: Math.round(R.cx + x0), y: Math.round(R.cy + y0) });
  const flagged = f => regions.filter(r => r.flags.includes(f));
  const ids = rs => [...new Set(rs.map(r => r.id))];
  const regionSummary = (id, f) => {
    const rs = regions.filter(r => r.id === id && (!f || r.flags.includes(f))), top = rs.reduce((a, b) => (b.area_mm2 > a.area_mm2 ? b : a));
    return { ...pos(top, 0), slices: slicesOf(id), area_mm2: top.area_mm2, dHU: top.dHU, regionHU: top.regionHU, ringHU: top.ringHU, open: top.open };
  };
  const NC = contrast ? null : undefined;   // NCCT signs are not assessed on a contrast-enhanced scan
  const sign = (present, evidence, detail) => (NC === null ? { auto: null, evidence: [], detail: 'not assessed: the scan appears contrast-enhanced (NCCT signs are defined on non-contrast CT)' } : { auto: present, evidence, detail });
  const by = f => ids(flagged(f)).map(id => regionSummary(id, f));
  const hyp = by('hypodensity'), bh = by('black hole'), sw = by('swirl'), bl = by('blend'), fl = by('fluid level');
  const fmt = (r, what) => `slice ${r.slice + 1}: ${what}`;
  out.signs.hypodensity = sign(hyp.length > 0, hyp, hyp.length ? hyp.map(r => fmt(r, `${r.area_mm2.toFixed(0)} mm², ${r.dHU.toFixed(0)} HU below the surrounding clot`)).join('; ') : 'no encapsulated hypodense region');
  out.signs.blackhole = sign(bh.length > 0, bh, bh.length ? bh.map(r => fmt(r, `${r.dHU.toFixed(0)} HU difference (≥ ${C.blackHole_dHU})`)).join('; ') : hyp.length ? `encapsulated hypodensities present but < ${C.blackHole_dHU} HU difference (max ${Math.max(...hyp.map(r => r.dHU)).toFixed(0)} HU)` : 'no encapsulated hypodense region');
  out.signs.swirl = sign(sw.length > 0, sw, sw.length ? sw.map(r => fmt(r, `${r.regionHU.toFixed(0)} HU region (brain ${brainRef.toFixed(0)} HU) over ${r.slices.length} slices`)).join('; ') : 'no iso/hypodense region within the clot on ≥ 2 contiguous slices');
  out.signs.blend = sign(bl.length > 0, bl, bl.length ? bl.map(r => fmt(r, `${r.area_mm2.toFixed(0)} mm² hypodense part at the edge, ${r.dHU.toFixed(0)} HU below the hyperdense part`)).join('; ') : 'no marginal hypodense part ≥ 18 HU below the hyperdense part');
  out.signs.fluid = sign(fl.length > 0, fl, fl.length ? fl.map(r => fmt(r, 'straight near-horizontal interface, hypodense part anterior (non-dependent)')).join('; ') : 'no straight blood-fluid interface');
  out.signs.heterogeneous = sign(heteroFoci.length >= C.heteroMinFoci, heteroFoci.map(r => pos(r, 0)),
    metalSlices.includes(zMax) ? `largest section (slice ${zMax + 1}) not read: metal artefact` : `${heteroFoci.length} hypodense foci on the largest section (slice ${zMax + 1}); HU coefficient of variation ${(100 * sdHU / meanHU).toFixed(0)}%`);
  if (metalSlices.includes(zMax)) out.signs.heterogeneous.auto = null;
  out.signs.irregular = sign(margins >= C.irregularMinMargins, [{ slice: zMax }],
    `${margins} margin irregularit${margins === 1 ? 'y' : 'ies'} on the largest section (slice ${zMax + 1}); solidity ${solidity.toFixed(2)}`);
  out.signs.island = sign(islandSlices.length > 0, islandSlices.map(s => ({ slice: s.k })),
    islandSlices.length ? islandSlices.map(s => `slice ${s.k + 1}: ${s.separate} separate + ${s.connected} connected`).join('; ')
      : `at most ${Math.max(0, ...perSlice.filter(Boolean).map(s => s.foci.length))} separate and ${Math.max(0, ...perSlice.filter(Boolean).map(s => s.foci.length + s.lobules.filter(l => l.island).length))} total small hematomas on one section`);
  out.signs.satellite = sign(satellites.length > 0, satellites.map(f => ({ slice: f.k, x: Math.round(f.cx + x0), y: Math.round(f.cy + y0) })),
    satellites.length ? satellites.slice(0, 6).map(f => `slice ${f.k + 1}: ${f.diam_mm.toFixed(1)} mm focus, ${f.gap_mm.toFixed(1)} mm from the clot, ${f.meanHU.toFixed(0)} HU`).join('; ') + (satellites.length > 6 ? `; +${satellites.length - 6} more` : '') : 'no separate focus < 10 mm within 1–20 mm');
  if (contrast) {
    out.warnings.push('Spot-sign mimics (Gazzola et al., Stroke 2008): calcification (tumour, choroid plexus) seen on the NCCT, and vascular lesions (micro-AVM, aneurysm, moyamoya pseudoaneurysm) with vessels entering the clot from its edge. Compare every candidate with the NCCT and the CTA source images.');
    const n = spots.length, maxD = n ? Math.max(...spots.map(s => s.diam_mm)) : 0, maxH = n ? Math.max(...spots.map(s => s.maxHU)) : 0;
    const sss = n ? (n >= 3 ? 2 : 1) + (maxD >= 5 ? 1 : 0) + (maxH >= 180 ? 1 : 0) : 0;
    out.signs.spot = { auto: n > 0, evidence: spots.map(s => ({ slice: s.k, x: Math.round(s.cx + x0), y: Math.round(s.cy + y0) })),
      detail: n ? `${n} focus/foci ≥ ${C.spotHU} HU inside the clot, not touching its edge; max ${maxH.toFixed(0)} HU, ${maxD.toFixed(1)} mm; spot sign score ${sss}/4 (approximate on 5 mm slabs)` : `no focus ≥ ${C.spotHU} HU inside the clot`, score: sss };
  } else out.signs.spot = { auto: null, evidence: calcs.map(c => ({ slice: c.slice, x: c.x, y: c.y })),
    detail: 'needs CT angiography (this scan is non-contrast)' + (calcs.length ? `. Calcification ${calcs.some(c => c.inside) ? 'inside' : 'within 5 mm of'} the clot on slice${calcs.length > 1 ? 's' : ''} ${[...new Set(calcs.map(c => c.slice + 1))].join(', ')} (max ${Math.max(...calcs.map(c => c.maxHU))} HU): on a CTA it would mimic a spot sign (Gazzola et al., Stroke 2008)` : '') };
  out.signs.leakage = { auto: null, evidence: [], detail: 'needs CTA plus a delayed (≈5 min) phase: > 10% HU rise in a 10 mm ROI' };
  out.signs.iodine = { auto: null, evidence: [], detail: 'needs dual-energy CT iodine maps' };

  out.hematoma = {
    volume_mL: mainVol, slices: [z0, z1], maxSlice: zMax, maxArea_mm2: sliceData[zMax].area * pxA,
    hemRef, brainRef, metalSlices, catheterSlices, meanHU, sdHU, cv: sdHU / meanHU, solidity, margins, heteroFoci: heteroFoci.length,
    box: { x0, y0, w, h },
  };
  out.regions = regions.map(r => ({ id: r.id, slice: r.k, open: r.open, share: +r.share.toFixed(2), interface_mm: +r.interface_mm.toFixed(1), edge_HUmm: +r.edge_HUmm.toFixed(1), area_mm2: +r.area_mm2.toFixed(1), width_mm: +r.width_mm.toFixed(1), regionHU: +r.regionHU.toFixed(1), ringHU: +r.ringHU.toFixed(1), dHU: +r.dHU.toFixed(1), flags: r.flags, fluid: r.fluid || undefined }));
  out.satellites = satellites.map(f => ({ slice: f.k, diam_mm: +f.diam_mm.toFixed(1), gap_mm: +f.gap_mm.toFixed(1), meanHU: +f.meanHU.toFixed(0), maxHU: +f.maxHU.toFixed(0) }));
  out.islands = islandSlices; out.spots = spots.map(s => ({ slice: s.k, maxHU: s.maxHU, diam_mm: +s.diam_mm.toFixed(1) }));
  return finish(out, C);
}

function finish(out, C) {
  for (const k of ['hypodensity', 'blackhole', 'swirl', 'blend', 'fluid', 'heterogeneous', 'irregular', 'island', 'satellite'])
    if (!out.signs[k]) out.signs[k] = { auto: null, evidence: [], detail: out.status === 'no-ich' ? `no intraparenchymal hematoma ≥ ${C.minHematoma_mL} mL found` : 'not assessed' };
  if (!out.signs.spot) out.signs.spot = { auto: null, evidence: [], detail: 'needs CT angiography' };
  if (!out.signs.leakage) out.signs.leakage = { auto: null, evidence: [], detail: 'needs CTA plus a delayed phase' };
  if (!out.signs.iodine) out.signs.iodine = { auto: null, evidence: [], detail: 'needs dual-energy CT iodine maps' };
  return out;
}

// smallest squared distance (pixels) between two masks
function sqdistMin(a, b, w, h) { const d = sqdist(a, w, h); let m = Infinity; for (let q = 0; q < w * h; q++) if (b[q] && d[q] < m) m = d[q]; return m; }

// straight, near-horizontal interface with hyperdense blood posterior (dependent) to the hypodense region
function fluidLevel(grown, E, cand, w, h, sx, sy, C) {
  const set = new Set(grown), pts = [];
  let xmin = Infinity, xmax = -Infinity, my = 0;
  for (const q of grown) { const x = q % w, y = (q - x) / w; xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); my += y; }
  my /= grown.length;
  // the lowest (most posterior) region pixel in each column that sits on hyperdense clot
  const col = new Map();
  for (const q of grown) { const x = q % w, y = (q - x) / w; if (y + 1 < h) { const t = q + w; if (E[t] && !cand[t] && !set.has(t)) { if (!col.has(x) || col.get(x) < y) col.set(x, y); } } }
  for (const [x, y] of col) pts.push([x * sx, (y + 0.5) * sy]);
  if (pts.length < 3) return { ok: false };
  const n = pts.length, mx = pts.reduce((s, p) => s + p[0], 0) / n, myI = pts.reduce((s, p) => s + p[1], 0) / n;
  let sxx = 0, sxy = 0; for (const [x, y] of pts) { sxx += (x - mx) ** 2; sxy += (x - mx) * (y - myI); }
  const a = sxx ? sxy / sxx : 0, b = myI - a * mx;
  const rms = Math.sqrt(pts.reduce((s, [x, y]) => s + (y - (a * x + b)) ** 2, 0) / n);
  const len = (Math.max(...pts.map(p => p[0])) - Math.min(...pts.map(p => p[0]))) + sx, span = (xmax - xmin + 1) * sx;
  const tilt = Math.abs(Math.atan(a)) * 180 / Math.PI;
  // the whole hypodense region lies above (anterior to) the line, and the line spans much of the clot's width there
  let below = 0; for (const q of grown) { const x = q % w, y = (q - x) / w; if ((y + 0.5) * sy > a * x * sx + b + 1.5 * sy) below++; }
  const row = Math.round(myI / sy); let ex0 = Infinity, ex1 = -Infinity;
  for (let x = 0; x < w; x++) if (row >= 0 && row < h && E[row * w + x]) { ex0 = Math.min(ex0, x); ex1 = Math.max(ex1, x); }
  const clotW = ex1 >= ex0 ? (ex1 - ex0 + 1) * sx : span;
  const ok = tilt <= C.fluidMaxTilt_deg && rms <= C.fluidMaxRms_mm && len >= C.fluidMinLen_mm && len >= 0.6 * span && len >= C.fluidMinShare * clotW
    && my * sy < myI && below <= 0.05 * grown.length;
  return { ok, tilt_deg: +tilt.toFixed(1), rms_mm: +rms.toFixed(2), length_mm: +len.toFixed(1) };
}

// area / convex hull area of a 2D mask (monotone chain hull on pixel corners)
function convexSolidity(M, w, h) {
  const pts = []; let area = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (M[y * w + x]) { area++; pts.push([x, y], [x + 1, y], [x, y + 1], [x + 1, y + 1]); }
  if (area < 3) return 1;
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const p of pts) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
  for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); }
  const hull = lo.slice(0, -1).concat(up.slice(0, -1));
  let A = 0; for (let i = 0; i < hull.length; i++) { const [x1, y1] = hull[i], [x2, y2] = hull[(i + 1) % hull.length]; A += x1 * y2 - x2 * y1; }
  return area / (Math.abs(A) / 2 || 1);
}

// ---------- expansion between a baseline and a follow-up CT ----------
// conventional HE: ICH growth > 6 mL or > 33% (Dowlatshahi et al., Neurology 2011; as in Yu et al. 2026)
// revised HE (rHE): ICH growth >= 6 mL or >= 33%, IVH growth >= 1 mL, or new IVH (Yogendrakumar et al., Stroke 2020;51:1120-1127;
//   used by Ning, Yu et al., Sci Rep 2025;15:32021). "New IVH" from an AI segmentation: < 0.5 mL at baseline, >= 0.5 mL at follow-up.
export const NEW_IVH_mL = 0.5;
export function expansion({ ich0, ich1, ivh0 = 0, ivh1 = 0 }) {
  if ([ich0, ich1].some(v => v == null || isNaN(v))) return null;
  const dICH = ich1 - ich0, rel = ich0 > 0 ? dICH / ich0 : null, dIVH = (ivh1 || 0) - (ivh0 || 0);
  const newIVH = (ivh0 || 0) < NEW_IVH_mL && (ivh1 || 0) >= NEW_IVH_mL;
  const conventional = dICH > 6 || (rel != null && rel > 0.33);
  const ichPart = dICH >= 6 || (rel != null && rel >= 0.33), ivhPart = dIVH >= 1 || newIVH;
  return { dICH, rel, dIVH, newIVH, conventional, revised: ichPart || ivhPart, revisedByIVHOnly: !ichPart && ivhPart,
    thresholds: { ge3: dICH >= 3, ge6: dICH >= 6, ge9: dICH >= 9, ge12_5: dICH >= 12.5 } };
}

// ---------- scores ----------
// BAT (Morotti et al., Stroke 2018): blend 1, any hypodensity 2, onset-to-NCCT < 2.5 h 2; >= 3 = high risk of expansion
export function batScore({ blend, hypodensity, hours }) {
  if (blend == null || hypodensity == null || hours == null || isNaN(hours)) return null;
  const s = (blend ? 1 : 0) + (hypodensity ? 2 : 0) + (hours < 2.5 ? 2 : 0);
  return { score: s, high: s >= 3 };
}
// BRAIN (Wang et al., Stroke 2015): volume <=10 0, 10-20 5, >20 7; recurrent ICH 4; warfarin 6; IVH 2; hours <=1 5, 1-2 4, 2-3 3, 3-4 2, 4-5 1, >5 0
// predicted probability (%) of growth >= 6 mL at 24 h by point total, Figure 1 of the paper (PMID 25503550)
export const BRAIN_PROB = [3.4, 4.2, 5.1, 6.3, 7.7, 9.4, 11.3, 13.7, 16.4, 19.5, 23.1, 27.2, 31.6, 36.4, 41.5, 46.7, 52.1, 57.4, 62.5, 67.4, 71.9, 76.0, 79.7, 83.0, 85.8];
export function brainScore({ volume, recurrent, warfarin, ivh, hours }) {
  if (volume == null || hours == null || isNaN(hours)) return null;
  const v = volume <= 10 ? 0 : volume <= 20 ? 5 : 7;
  const t = hours <= 1 ? 5 : hours <= 2 ? 4 : hours <= 3 ? 3 : hours <= 4 ? 2 : hours <= 5 ? 1 : 0;
  const score = v + (recurrent ? 4 : 0) + (warfarin ? 6 : 0) + (ivh ? 2 : 0) + t;
  return { score, probability_pct: BRAIN_PROB[score], parts: { volume: v, recurrent: recurrent ? 4 : 0, warfarin: warfarin ? 6 : 0, ivh: ivh ? 2 : 0, time: t } };
}

// ---------- DeepMedX sICH_HE region of interest (crop2DROI.ipynb), for their ViT classifier ----------
// The published code upsamples the CT and mask 3.5x in all three axes, takes the slice with the largest mask area and
// its two neighbours (+-1/3.5 of a 5 mm slice), crops each to the bounding box of the largest connected component,
// windows it to 0-90 HU (level 45, width 90) and saves an 8-bit image; the ViT then resizes to 256 and centre-crops 224.
export function deepmedxROI(g, hemMask, { scale = 3.5, wl = 45, ww = 90 } = {}) {
  const { nx, ny, nz, hu } = g, P = nx * ny;
  const L = label3d(hemMask, nx, ny, nz); let best = 0; for (let i = 1; i <= L.n; i++) if (L.sizes[i] > (L.sizes[best] || 0)) best = i;
  if (!best) return null;
  const area = new Float64Array(nz); for (let k = 0; k < nz; k++) for (let p = 0; p < P; p++) if (hemMask[k * P + p]) area[k]++;
  // argmax of the (linearly interpolated) area over the upsampled slice positions t = j / scale
  const NZ = Math.floor(nz * scale); let jm = 0, am = -1;
  for (let j = 0; j < NZ; j++) { const t = Math.min(nz - 1, j / scale), k = Math.floor(t), f = t - k, a = area[k] * (1 - f) + (k + 1 < nz ? area[k + 1] : 0) * f; if (a > am) { am = a; jm = j; } }
  const planes = [];
  for (const [name, j] of [['Max−1 (inferior)', jm - 1], ['Max', jm], ['Max+1 (superior)', jm + 1]]) {
    if (j < 0 || j >= NZ) continue;
    const t = Math.min(nz - 1, j / scale), k = Math.floor(t), f = t - k, k2 = Math.min(nz - 1, k + 1);
    // bounding box of the largest component in this (interpolated) plane
    let bx0 = nx, bx1 = -1, by0 = ny, by1 = -1;
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const p = y * nx + x, v = (L.lab[k * P + p] === best ? 1 - f : 0) + (L.lab[k2 * P + p] === best ? f : 0);
      if (v > 0.5) { if (x < bx0) bx0 = x; if (x > bx1) bx1 = x; if (y < by0) by0 = y; if (y > by1) by1 = y; }
    }
    if (bx1 < 0) continue;
    // crop in the upsampled plane: pixel u covers original x = u / scale (bicubic in plane, linear between slices)
    const U0 = Math.round(bx0 * scale), U1 = Math.round((bx1 + 1) * scale) - 1, V0 = Math.round(by0 * scale), V1 = Math.round((by1 + 1) * scale) - 1;
    const cw = U1 - U0 + 1, ch = V1 - V0 + 1, img = new Uint8ClampedArray(cw * ch);
    const at = (kk, x, y) => hu[kk * P + Math.min(ny - 1, Math.max(0, y)) * nx + Math.min(nx - 1, Math.max(0, x))];
    const cub = (a, b, c, d, s) => b + 0.5 * s * (c - a + s * (2 * a - 5 * b + 4 * c - d + s * (3 * (b - c) + d - a)));
    for (let v = 0; v < ch; v++) for (let u = 0; u < cw; u++) {
      const X = (U0 + u) / scale, Y = (V0 + v) / scale, xi = Math.floor(X), yi = Math.floor(Y), fx = X - xi, fy = Y - yi;
      let val = 0;
      for (const [kk, wz] of [[k, 1 - f], [k2, f]]) {
        if (!wz) continue;
        const rows = [-1, 0, 1, 2].map(dy => cub(at(kk, xi - 1, yi + dy), at(kk, xi, yi + dy), at(kk, xi + 1, yi + dy), at(kk, xi + 2, yi + dy), fx));
        val += wz * cub(rows[0], rows[1], rows[2], rows[3], fy);
      }
      img[v * cw + u] = Math.round((val - (wl - ww / 2)) / ww * 255);
    }
    planes.push({ name, slice: t, w: cw, h: ch, img, vit: vitInput(img, cw, ch) });
  }
  return { planes, maxIndexUpsampled: jm, scale, window: [wl, ww] };
}
// torchvision Resize(256) (shorter side to 256, bilinear) then CenterCrop(224)
function vitInput(img, w, h) {
  const s = 256 / Math.min(w, h), W = Math.max(256, Math.round(w * s)), H = Math.max(256, Math.round(h * s)), out = new Uint8ClampedArray(224 * 224);
  const ox = Math.round((W - 224) / 2), oy = Math.round((H - 224) / 2);
  for (let y = 0; y < 224; y++) for (let x = 0; x < 224; x++) {
    const X = Math.min(w - 1, Math.max(0, (x + ox + 0.5) / s - 0.5)), Y = Math.min(h - 1, Math.max(0, (y + oy + 0.5) / s - 0.5));
    const xi = Math.floor(X), yi = Math.floor(Y), fx = X - xi, fy = Y - yi, x2 = Math.min(w - 1, xi + 1), y2 = Math.min(h - 1, yi + 1);
    out[y * 224 + x] = (img[yi * w + xi] * (1 - fx) + img[yi * w + x2] * fx) * (1 - fy) + (img[y2 * w + xi] * (1 - fx) + img[y2 * w + x2] * fx) * fy;
  }
  return out;
}

// the reference list of CT signs of hematoma expansion (shown on the page, also used for the report)
export const SIGN_INFO = [
  { key: 'spot', name: 'Spot sign', modality: 'CTA', group: 'Contrast extravasation',
    def: '≥ 1 focus of contrast pooling within the hematoma, ≥ 120 HU, discontinuous from adjacent normal or abnormal vessels, any size and shape. Mimics: calcification (check the NCCT) and vascular lesions with vessels entering from the edge (micro-AVM, aneurysm, moyamoya).',
    ref: 'Wada et al., Stroke 2007; Delgado Almandoz et al., Stroke 2009 (spot sign score); Gazzola et al., Stroke 2008;39:1177-1183 (mimics)' },
  { key: 'leakage', name: 'Leakage sign', modality: 'CTA + delayed', group: 'Contrast extravasation',
    def: '> 10% rise in HU within a 10 mm ROI in the hematoma between the CTA phase and a delayed phase about 5 minutes later.',
    ref: 'Orito et al., Stroke 2016;47:958-963' },
  { key: 'iodine', name: 'Iodine sign', modality: 'Dual-energy CT', group: 'Contrast extravasation',
    def: 'Focal iodine (contrast) concentration within the hematoma on dual-energy / spectral iodine maps.',
    ref: 'Fu et al. and others; reviewed by Liu et al., GeroScience 2026' },
  { key: 'blend', name: 'Blend sign', modality: 'NCCT', group: 'Density heterogeneity',
    def: 'A hypodense region next to a hyperdense region of the hematoma, sharply separated, ≥ 18 HU difference, the hypodense part not encapsulated.',
    ref: 'Li et al., Stroke 2015;46:2119-2123' },
  { key: 'blackhole', name: 'Black hole sign', modality: 'NCCT', group: 'Density heterogeneity',
    def: 'A hypodense region encapsulated within the hyperdense hematoma, well defined, ≥ 28 HU difference, not connected with the brain outside.',
    ref: 'Li et al., Stroke 2016;47:1777-1781' },
  { key: 'hypodensity', name: 'Hypodensity sign', modality: 'NCCT', group: 'Density heterogeneity',
    def: 'Any hypodense region strictly encapsulated within the hematoma, of any shape, size and density.',
    ref: 'Boulouis et al., JAMA Neurol 2016;73:961-968' },
  { key: 'swirl', name: 'Swirl sign', modality: 'NCCT', group: 'Density heterogeneity',
    def: 'Rounded, streak-like or irregular region of hypo- or iso-attenuation (relative to brain) within the hyperattenuating hematoma.',
    ref: 'Selariu et al., BMC Neurol 2012;12:109' },
  { key: 'island', name: 'Island sign', modality: 'NCCT', group: 'Morphological instability',
    def: 'On one axial section: ≥ 3 small hematomas all separate from the main one, or ≥ 4 small hematomas some or all of which connect with it.',
    ref: 'Li et al., Stroke 2017;48:3019-3025' },
  { key: 'satellite', name: 'Satellite sign', modality: 'NCCT', group: 'Morphological instability',
    def: 'A small hematoma (< 10 mm) completely separate from the main one on at least one section, 1-20 mm away.',
    ref: 'Shimoda et al., AJNR 2017;38:737-742' },
  { key: 'irregular', name: 'Irregular shape', modality: 'NCCT', group: 'Morphological instability',
    def: '≥ 2 focal margin irregularities, joined or separate from the edge, on the largest section (Barras shape III-V).',
    ref: 'Barras et al., Stroke 2009;40:1325-1331; Morotti et al., Ann Neurol 2019' },
  { key: 'heterogeneous', name: 'Heterogeneous density', modality: 'NCCT', group: 'Density heterogeneity',
    def: '≥ 3 hypodense foci within the hematoma on the largest section (Barras density III-V).',
    ref: 'Barras et al., Stroke 2009; Delcourt et al., Stroke 2016' },
  { key: 'fluid', name: 'Fluid level', modality: 'NCCT', group: 'Density heterogeneity',
    def: 'Hypodense (serum) region above a hyperdense (cellular) region with a straight line between them.',
    ref: 'Morotti et al., Ann Neurol 2019;86:480-492' },
];
