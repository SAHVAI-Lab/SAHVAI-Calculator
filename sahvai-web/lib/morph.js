// Binary morphology on Uint8Array masks (1 = foreground), 2D slices of w*h and 3D volumes.
// Connectivity follows scikit-image defaults where the upstream code relies on them
// (remove_small_objects / remove_small_holes / label: connectivity 1 = 4-neighbour in 2D).

export function label2d(m, w, h, conn8 = false) {
  const lab = new Int32Array(w * h), sizes = [0], stack = new Int32Array(w * h);
  let n = 0;
  for (let p = 0; p < w * h; p++) {
    if (!m[p] || lab[p]) continue;
    n++; let sp = 0, size = 0; stack[sp++] = p; lab[p] = n;
    while (sp) {
      const q = stack[--sp]; size++;
      const x = q % w, y = (q - x) / w;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue; if (!conn8 && dx && dy) continue;
        const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= w || Y >= h) continue;
        const r = Y * w + X; if (m[r] && !lab[r]) { lab[r] = n; stack[sp++] = r; }
      }
    }
    sizes.push(size);
  }
  return { lab, n, sizes };
}

export function removeSmallObjects(m, w, h, minSize, conn8 = false) {
  const { lab, sizes } = label2d(m, w, h, conn8), out = new Uint8Array(w * h);
  for (let p = 0; p < w * h; p++) if (lab[p] && sizes[lab[p]] >= minSize) out[p] = 1;
  return out;
}

export function removeSmallHoles(m, w, h, area, conn8 = false) {
  const inv = new Uint8Array(w * h); for (let p = 0; p < w * h; p++) inv[p] = m[p] ? 0 : 1;
  const keep = removeSmallObjects(inv, w, h, area, conn8), out = new Uint8Array(w * h);
  for (let p = 0; p < w * h; p++) out[p] = keep[p] ? 0 : 1;
  return out;
}

// background reachable from the image border (4-neighbour) through pixels where pass[p] = 1
export function floodFromBorder(pass, w, h) {
  const seen = new Uint8Array(w * h), stack = new Int32Array(w * h); let sp = 0;
  const push = p => { if (pass[p] && !seen[p]) { seen[p] = 1; stack[sp++] = p; } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (sp) {
    const q = stack[--sp], x = q % w, y = (q - x) / w;
    if (x > 0) push(q - 1); if (x < w - 1) push(q + 1); if (y > 0) push(q - w); if (y < h - 1) push(q + w);
  }
  return seen;
}

export function fillHoles(m, w, h) {
  const bg = new Uint8Array(w * h); for (let p = 0; p < w * h; p++) bg[p] = m[p] ? 0 : 1;
  const outside = floodFromBorder(bg, w, h), out = new Uint8Array(w * h);
  for (let p = 0; p < w * h; p++) out[p] = outside[p] ? 0 : 1;
  return out;
}

// squared Euclidean distance to the nearest pixel with m = 1 (Felzenszwalb & Huttenlocher)
function edt1(f, n, d, v, z) {
  let k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s;
    for (;;) {
      s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      if (s <= z[k]) { k--; if (k < 0) { k = 0; break; } } else break;
    }
    k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; const dq = q - v[k]; d[q] = dq * dq + f[v[k]]; }
}
export function sqdist(m, w, h) {
  const INF = 1e20, g = new Float64Array(w * h), n = Math.max(w, h);
  const f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = m[y * w + x] ? 0 : INF;
    edt1(f, h, d, v, z); for (let y = 0; y < h; y++) g[y * w + x] = d[y];
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = g[y * w + x];
    edt1(f, w, d, v, z); for (let x = 0; x < w; x++) g[y * w + x] = d[x];
  }
  return g;
}
// dilation / erosion by a disk of radius r (x^2 + y^2 <= r^2, like skimage.morphology.disk)
export function dilate(m, w, h, r) {
  const g = sqdist(m, w, h), out = new Uint8Array(w * h), r2 = r * r + 1e-9;
  for (let p = 0; p < w * h; p++) out[p] = g[p] <= r2 ? 1 : 0;
  return out;
}
export function erode(m, w, h, r) {
  const inv = new Uint8Array(w * h); for (let p = 0; p < w * h; p++) inv[p] = m[p] ? 0 : 1;
  const d = dilate(inv, w, h, r), out = new Uint8Array(w * h);
  for (let p = 0; p < w * h; p++) out[p] = d[p] ? 0 : 1;
  return out;
}
export const closing = (m, w, h, r) => erode(dilate(m, w, h, r), w, h, r);

// 3D 6-neighbour components; returns a mask of the largest one
export function largest3d(m, nx, ny, nz) {
  const N = nx * ny * nz, lab = new Int32Array(N), stack = new Int32Array(N), plane = nx * ny;
  let n = 0, best = 0, bestSize = 0;
  for (let p = 0; p < N; p++) {
    if (!m[p] || lab[p]) continue;
    n++; let sp = 0, size = 0; stack[sp++] = p; lab[p] = n;
    while (sp) {
      const q = stack[--sp]; size++;
      const z = Math.floor(q / plane), r = q - z * plane, y = Math.floor(r / nx), x = r - y * nx;
      const nb = [x > 0 ? q - 1 : -1, x < nx - 1 ? q + 1 : -1, y > 0 ? q - nx : -1, y < ny - 1 ? q + nx : -1, z > 0 ? q - plane : -1, z < nz - 1 ? q + plane : -1];
      for (const t of nb) if (t >= 0 && m[t] && !lab[t]) { lab[t] = n; stack[sp++] = t; }
    }
    if (size > bestSize) { bestSize = size; best = n; }
  }
  const out = new Uint8Array(N);
  for (let p = 0; p < N; p++) if (lab[p] === best && best) out[p] = 1;
  return out;
}
