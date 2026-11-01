// HEADS-UP: Hemorrhage Evaluation and Detector System for Underserved Populations (Mayo Clinic).
// Slice-level detection of intracranial hemorrhage and its five subtypes on non-contrast head CT,
// with the 2D CNNs of the RSNA 2019 Intracranial Hemorrhage Detection 1st-place solution
// (SeuTao; Wang X, Shen T, et al. NeuroImage Clin 2021;32:102785), converted to ONNX unchanged.
//
// Preprocessing reproduces the solution's prepare_data.py / dataset.py / predict.py:
//   slice HU -> brain window (DICOM WindowCenter/Width, else 40/80) -> clip -> per-slice min-max -> 0..255 (floor)
//   -> resize to 512 x 512 -> stack [slice above, slice, slice below] as 3 channels
//   -> (256 px models: resize to 256) -> centre crop 80 % -> resize back -> (v/255 - 0.456) / 0.224
// Output per slice: sigmoid probabilities for any, EDH, IPH, IVH, SAH, SDH, and a class activation map
// (per-cell logit contribution) used for approximate localisation only.

export const CLASS_KEYS = ['any', 'EDH', 'IPH', 'IVH', 'SAH', 'SDH'];
export const CLASS_NAMES = ['any intracranial hemorrhage', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'];
export const RSNA_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'];
// label values of the HEADS-UP localisation map (same convention as the BHSD dataset)
export const LABEL_OF_CLASS = [0, 1, 2, 3, 4, 5];   // class index 1..5 -> label 1 EDH, 2 IPH, 3 IVH, 4 SAH, 5 SDH

export const MODEL_SPECS = {
  headsup_dn121: { size: 512, label: 'DenseNet121 · 512 px', short: 'DN121' },
  headsup_dn169: { size: 256, label: 'DenseNet169 · 256 px', short: 'DN169' },
  headsup_sx101: { size: 256, label: 'SE-ResNeXt101 · 256 px', short: 'SX101' },
};

// thresholds on the study-level (maximum slice) probability
export const T_DETECT = 0.5, T_POSSIBLE = 0.2;

// Working grid: the scan at its own in-plane resolution; slices thinner than 4 mm are averaged to 5 mm
// (the RSNA 2019 studies are predominantly 5 mm axial).
export const HEADSUP_GRID = { spacing: sp => [sp[0], sp[1], sp[2] < 4 ? 5 : sp[2]], align: 'extent', interp: 'auto', label: 'HEADS-UP', geometry: true };

// a DICOM brain window, if the series carries one; else the standard 40 / 80
export function brainWindow(w) {
  const c = w && Number(w[0]), ww = w && Number(w[1]);
  if (Number.isFinite(c) && Number.isFinite(ww) && c >= 20 && c <= 60 && ww >= 60 && ww <= 160) return [Math.trunc(c), Math.trunc(ww)];
  return [40, 80];
}

// one slice as the solution's 8-bit PNG (rows = posterior-going y, columns = patient-left-going x)
export function slicePng(hu, nx, ny, k, wc, ww) {
  const P = nx * ny, o = k * P, lo = wc - Math.floor(ww / 2), hi = wc + Math.floor(ww / 2);
  let mi = Infinity, ma = -Infinity;
  const c = new Float64Array(P);
  for (let p = 0; p < P; p++) { let v = hu[o + p]; v = v < lo ? lo : v > hi ? hi : v; c[p] = v; if (v < mi) mi = v; if (v > ma) ma = v; }
  const out = new Uint8Array(P);
  if (ma > mi) { const s = 1 / (ma - mi); for (let p = 0; p < P; p++) out[p] = Math.floor((c[p] - mi) * s * 255); }
  return out;
}

// OpenCV INTER_LINEAR resize of an 8-bit image (half-pixel centres, edge clamp, 11-bit fixed-point weights),
// with OpenCV's switch to INTER_AREA for an exact 2x reduction. ch channels interleaved.
export function resizeU8(src, sw, sh, dw, dh, ch = 1) {
  if (sw === dw && sh === dh) return src.slice();
  const out = new Uint8Array(dw * dh * ch);
  if (sw === 2 * dw && sh === 2 * dh) {   // cv2: INTER_LINEAR with an exact 2x downscale uses INTER_AREA
    for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) for (let c = 0; c < ch; c++) {
      const a = ((2 * y) * sw + 2 * x) * ch + c, b = a + sw * ch;
      out[(y * dw + x) * ch + c] = (src[a] + src[a + ch] + src[b] + src[b + ch] + 2) >> 2;
    }
    return out;
  }
  const taps = (sn, dn) => {
    const s = sn / dn, i0 = new Int32Array(dn), i1 = new Int32Array(dn), w0 = new Int32Array(dn), w1 = new Int32Array(dn);
    for (let d = 0; d < dn; d++) {
      let f = (d + 0.5) * s - 0.5, i = Math.floor(f); f -= i;
      if (i < 0) { i = 0; f = 0; }
      if (i >= sn - 1) { i = sn - 1; f = 0; }
      i0[d] = i; i1[d] = Math.min(i + 1, sn - 1);
      w0[d] = Math.round((1 - f) * 2048); w1[d] = Math.round(f * 2048);
    }
    return { i0, i1, w0, w1 };
  };
  const X = taps(sw, dw), Y = taps(sh, dh), row0 = new Int32Array(dw * ch), row1 = new Int32Array(dw * ch);
  const hrow = (y, buf) => { const b = y * sw * ch; for (let x = 0; x < dw; x++) for (let c = 0; c < ch; c++) buf[x * ch + c] = src[b + X.i0[x] * ch + c] * X.w0[x] + src[b + X.i1[x] * ch + c] * X.w1[x]; };
  for (let y = 0; y < dh; y++) {
    hrow(Y.i0[y], row0); hrow(Y.i1[y], row1);
    const a = Y.w0[y], b = Y.w1[y], o = y * dw * ch;
    // OpenCV's SIMD vertical pass (VResizeLinearVec_32s8u): ((S0 >> 4) * b0 >> 16) + ((S1 >> 4) * b1 >> 16), then (+2) >> 2
    for (let i = 0; i < dw * ch; i++) { const v = (((row0[i] >> 4) * a >> 16) + ((row1[i] >> 4) * b >> 16) + 2) >> 2; out[o + i] = v < 0 ? 0 : v > 255 ? 255 : v; }
  }
  return out;
}

// network input for slices [k0, k1) at size S: Float32 [n, 3, S, S]
export function makeInput(png512, nz, k0, k1, S) {
  const n = k1 - k0, x = new Float32Array(n * 3 * S * S), t = Math.trunc(S * 0.8), s0 = Math.trunc((S - t) / 2);
  for (let j = 0; j < n; j++) {
    const k = k0 + j, chans = [png512[Math.min(k + 1, nz - 1)], png512[k], png512[Math.max(k - 1, 0)]];
    let im = new Uint8Array(512 * 512 * 3);
    for (let p = 0; p < 512 * 512; p++) { im[3 * p] = chans[0][p]; im[3 * p + 1] = chans[1][p]; im[3 * p + 2] = chans[2][p]; }
    if (S !== 512) im = resizeU8(im, 512, 512, S, S, 3);
    const crop = new Uint8Array(t * t * 3);
    for (let y = 0; y < t; y++) crop.set(im.subarray(((y + s0) * S + s0) * 3, ((y + s0) * S + s0 + t) * 3), y * t * 3);
    const r = resizeU8(crop, t, t, S, S, 3), o = j * 3 * S * S;
    for (let p = 0; p < S * S; p++) for (let c = 0; c < 3; c++) x[o + c * S * S + p] = (r[3 * p + c] / 255 - 0.456) / 0.224;
  }
  return x;
}

// Per-voxel localisation from the class activation maps (approximate, NOT a segmentation):
// on slices where a subtype's probability reaches the threshold, voxels inside that subtype's activation hot spot
// (>= half the slice maximum) that are hyperdense (45-100 HU, the range of acute blood) get the subtype's label.
export function localise(g, probs, cam, ch, cw, crop, thr = T_DETECT) {
  const { nx, ny, nz, hu } = g, P = nx * ny, labels = new Uint8Array(P * nz), best = new Float32Array(P);
  for (let k = 0; k < nz; k++) {
    best.fill(0);
    for (let c = 1; c < 6; c++) {
      const pk = probs[k * 6 + c]; if (pk < thr) continue;
      const m = cam.subarray((k * 6 + c) * ch * cw, (k * 6 + c + 1) * ch * cw);
      let mx = 0; for (let i = 0; i < m.length; i++) if (m[i] > mx) mx = m[i];
      if (mx <= 0) continue;
      for (let y = 0; y < ny; y++) {
        const v = (y + 0.5) / ny * 512, cy = (v - crop.s) / crop.t;
        if (cy < 0 || cy >= 1) continue;
        const fy = Math.min(Math.max(cy * ch - 0.5, 0), ch - 1), y0 = Math.floor(fy), y1 = Math.min(y0 + 1, ch - 1), ay = fy - y0;
        for (let x = 0; x < nx; x++) {
          const u = (x + 0.5) / nx * 512, cx = (u - crop.s) / crop.t;
          if (cx < 0 || cx >= 1) continue;
          const p = k * P + y * nx + x, h = hu[p];
          if (h < 45 || h > 100) continue;
          const fx = Math.min(Math.max(cx * cw - 0.5, 0), cw - 1), x0 = Math.floor(fx), x1 = Math.min(x0 + 1, cw - 1), ax = fx - x0;
          const a = (m[y0 * cw + x0] * (1 - ax) + m[y0 * cw + x1] * ax) * (1 - ay) + (m[y1 * cw + x0] * (1 - ax) + m[y1 * cw + x1] * ax) * ay;
          const sc = a / mx;
          if (sc >= 0.5 && sc * pk > best[y * nx + x]) { best[y * nx + x] = sc * pk; labels[p] = LABEL_OF_CLASS[c]; }
        }
      }
    }
  }
  return labels;
}

// study-level summary from the slice probabilities [nz x 6]
export function summarise(probs, nz) {
  return CLASS_KEYS.map((key, c) => {
    let mx = 0, at = 0, n = 0, first = -1, last = -1;
    for (let k = 0; k < nz; k++) { const v = probs[k * 6 + c]; if (v > mx) { mx = v; at = k; } if (v >= T_DETECT) { n++; if (first < 0) first = k; last = k; } }
    return { key, name: CLASS_NAMES[c], max: mx, slice: at, n_slices: n, range: first < 0 ? null : [first, last],
      call: mx >= T_DETECT ? 'detected' : mx >= T_POSSIBLE ? 'possible' : 'not detected' };
  });
}

// Runs the selected networks over every slice. call(name, {x: {data, dims}}) -> [{data, dims}, ...] (prob, cam)
export async function runHeadsUp(g, { call, models, window, log = () => {}, stage = () => {}, batch = 2 }) {
  const { nx, ny, nz, hu } = g;
  const [wc, ww] = brainWindow(window);
  log(`HEADS-UP: ${nz} slices of ${nx}x${ny}, brain window ${wc}/${ww}${window ? ' (from the DICOM header)' : ''}; models: ${models.map(m => MODEL_SPECS[m].short).join(' + ')}`);
  stage('Preparing slices', 0);
  const png = [];
  for (let k = 0; k < nz; k++) png.push(resizeU8(slicePng(hu, nx, ny, k, wc, ww), nx, ny, 512, 512));
  const per = {}, sum = new Float32Array(nz * 6);
  let cam = null, ch = 0, cw = 0, crop = null, done = 0;
  const total = nz * models.length;
  for (const m of models) {
    const S = MODEL_SPECS[m].size, pm = new Float32Array(nz * 6), keepCam = !cam;
    const t = Math.trunc(S * 0.8), s0 = Math.trunc((S - t) / 2);
    for (let k = 0; k < nz; k += batch) {
      const k1 = Math.min(nz, k + batch), x = makeInput(png, nz, k, k1, S);
      const [p, c] = await call(m, { x: { data: x, dims: [k1 - k, 3, S, S] } });
      pm.set(p.data, k * 6);
      if (keepCam) {
        if (!cam) { ch = c.dims[2]; cw = c.dims[3]; cam = new Float32Array(nz * 6 * ch * cw); crop = { s: s0 * 512 / S, t: t * 512 / S }; }
        cam.set(c.data, k * 6 * ch * cw);
      }
      done += k1 - k; stage(`Classifying slices · ${MODEL_SPECS[m].short}`, done / total);
    }
    per[m] = pm; for (let i = 0; i < sum.length; i++) sum[i] += pm[i] / models.length;
  }
  const summary = summarise(sum, nz);
  log('HEADS-UP study level (max slice probability): ' + summary.map(s => `${s.key} ${s.max.toFixed(3)}`).join(', '));
  stage('Localising', 1);
  const labels = localise(g, sum, cam, ch, cw, crop);
  return { labels, probs: sum, perModel: per, cam, camDims: [ch, cw], crop, summary, window: [wc, ww], volumes: {} };
}
