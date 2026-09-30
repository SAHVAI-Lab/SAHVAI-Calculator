// Runs the Hybrid 2D/3D UNet split into ONNX parts (see tools/parts.py).
// Full-resolution stages go slab by slab (n slices + a 1-slice halo each side) and each 3D
// InstanceNorm gets whole-volume statistics from per-slice sums returned by the previous
// part. The half-resolution-and-below middle (pm) runs on the whole 18-slice window.
// call(name, feeds) -> Promise<[{data, dims}]> in the model's output order; feeds are {name: {data, dims}}.

export const PARTS = ['p1', 'p2', 'p3', 'p4', 'p5', 'pm', 'p6', 'p7', 'p8'];

export async function runUnet(x, D, Hh, Ww, call, { slab = 4, progress = () => {} } = {}) {
  const HW = Hh * Ww, hw = (Hh / 2) * (Ww / 2), N = D * HW;
  const slabs = []; for (let s = 0; s < D; s += slab) slabs.push([s, Math.min(s + slab, D)]);
  const total = slabs.length * 8 + 1; let done = 0;
  const tick = name => progress(++done / total, name);

  // full array (D, C, plane) -> halo slab (e-s+2, C, plane), zeros outside the volume
  const halo = (a, C, plane, s, e) => {
    const n = e - s + 2, out = new Float32Array(n * C * plane), sl = C * plane;
    for (let k = s - 1; k <= e; k++) if (k >= 0 && k < D) out.set(a.subarray(k * sl, (k + 1) * sl), (k - s + 1) * sl);
    return { data: out, dims: [n, C, Hh * Ww === plane ? Hh : Hh / 2, Hh * Ww === plane ? Ww : Ww / 2] };
  };
  const valid = (s, e) => { const v = new Float32Array(e - s + 2); for (let k = s - 1; k <= e; k++) v[k - s + 1] = k >= 0 && k < D ? 1 : 0; return { data: v, dims: [e - s + 2, 1, 1, 1] }; };
  const stats = (S, Q, C) => {
    const m = new Float32Array(C), i = new Float32Array(C);
    for (let c = 0; c < C; c++) { const mean = S[c] / N, v = Q[c] / N - mean * mean; m[c] = mean; i[c] = 1 / Math.sqrt(v + 1e-5); }
    return { m: { data: m, dims: [1, C, 1, 1] }, i: { data: i, dims: [1, C, 1, 1] } };
  };
  // run a part over all slabs; outputs listed in `keep` (index -> [C, plane]) are gathered into full arrays,
  // outputs 's'/'q' (per-slice channel sums) are accumulated
  async function loop(name, feedsFor, keep, sumIdx) {
    const full = keep.map(([C, plane]) => new Float32Array(D * C * plane));
    let S = null, Q = null, C = 0;
    for (const [s, e] of slabs) {
      const out = await call(name, feedsFor(s, e));
      keep.forEach(([Ck, plane], k) => full[k].set(out[k].data, s * Ck * plane));
      if (sumIdx !== undefined) {
        const so = out[sumIdx], qo = out[sumIdx + 1]; C = so.dims[1];
        if (!S) { S = new Float64Array(C); Q = new Float64Array(C); }
        for (let r = 0; r < so.dims[0]; r++) for (let c = 0; c < C; c++) { S[c] += so.data[r * C + c]; Q[c] += qo.data[r * C + c]; }
      }
      tick(name);
    }
    return { full, st: S ? stats(S, Q, C) : null };
  }

  let r = await loop('p1', (s, e) => ({ x: halo(x, 5, HW, s, e) }), [[32, HW], [32, HW]], 2);
  let [a2, r1] = r.full;
  r = await loop('p2', (s, e) => ({ r: halo(r1, 32, HW, s, e), v: valid(s, e), ...r.st }), [[32, HW]], 1);
  r1 = null;
  let [r2] = r.full;
  r = await loop('p3', (s, e) => ({ r: halo(r2, 32, HW, s, e), a2: halo(a2, 32, HW, s, e), v: valid(s, e), ...r.st }), [[64, hw], [64, HW]], 2);
  r2 = null; a2 = null;
  let [b2, r3] = r.full;
  r = await loop('p4', (s, e) => ({ r: halo(r3, 64, HW, s, e), v: valid(s, e), ...r.st }), [[64, HW]], 1);
  r3 = null;
  let [r4] = r.full;
  r = await loop('p5', (s, e) => ({ r: halo(r4, 64, HW, s, e), v: valid(s, e), ...r.st }), [[64, hw]]);
  r4 = null;
  let [b3] = r.full;
  const hOut = await call('pm', { b2: { data: b2, dims: [D, 64, Hh / 2, Ww / 2] }, b3: { data: b3, dims: [D, 64, Hh / 2, Ww / 2] } });
  let h = hOut[0].data; tick('pm');
  r = await loop('p6', (s, e) => ({ b2: halo(b2, 64, hw, s, e), b3: halo(b3, 64, hw, s, e), h: halo(h, 64, hw, s, e), v: valid(s, e) }), [[32, HW]], 1);
  b2 = b3 = h = null;
  let [r5] = r.full;
  r = await loop('p7', (s, e) => ({ r: halo(r5, 32, HW, s, e), v: valid(s, e), ...r.st }), [[32, HW]], 1);
  r5 = null;
  const [r6] = r.full;
  // last part: logits -> labels per slab (argmax), no need to keep 4-channel logits
  const labels = new Uint8Array(D * HW), st = r.st;
  for (const [s, e] of slabs) {
    const out = await call('p8', { r: { data: r6.subarray(s * 32 * HW, e * 32 * HW), dims: [e - s, 32, Hh, Ww] }, ...st });
    const L = out[0].data;
    for (let d = 0; d < e - s; d++) for (let i = 0; i < HW; i++) {
      let best = 0, bv = L[(d * 4) * HW + i];
      for (let c = 1; c < 4; c++) { const v = L[(d * 4 + c) * HW + i]; if (v > bv) { bv = v; best = c; } }
      labels[(s + d) * HW + i] = best;
    }
    tick('p8');
  }
  return labels;
}
