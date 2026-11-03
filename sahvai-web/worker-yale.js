// Yale hematoma-expansion model (Tran et al., npj Digit Med 2024; github.com/anhtrnyaleedu/HE) off the main thread.
// In:  {job: 'run', id, grid: {nx, ny, nz, sp, hu}, device, passes, mask (optional SAHVAI hematoma mask on the grid)}
// Out: {type: 'log'|'progress'|'result'|'error'|'missing', id, ...}
import * as ort from './vendor/ort/ort.webgpu.min.mjs';
import { YALE, prepBrain, segment, clsInput, gridToLAS, maskToGrid } from './lib/yalehe.js';

const base = new URL('./', import.meta.url).href;
ort.env.wasm.wasmPaths = base + 'vendor/ort/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 16) : 1;
ort.env.logLevel = 'error';
const VERSION = 'y1', CACHE = 'sahvai-yale-' + VERSION, FILES = { yale_seg: 9475016, yale_cls: 24023452 };
const post = (type, x = {}) => self.postMessage({ type, ...x });

async function fetchModel(name, onBytes) {
  const url = new URL(`models/yale/${name}.onnx`, base).href;
  const cache = self.caches ? await caches.open(CACHE).catch(() => null) : null;
  const hit = cache && await cache.match(url);
  if (hit) { const b = new Uint8Array(await hit.arrayBuffer()); onBytes(b.length); return b; }
  const r = await fetch(url);
  if (r.status === 404) { const e = new Error('model files not installed on this site'); e.missing = true; throw e; }
  if (!r.ok) throw new Error(`could not download ${name}.onnx (${r.status})`);
  const rd = r.body.getReader(), parts = []; for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); onBytes(value.length); }
  const blob = new Blob(parts); if (cache) await cache.put(url, new Response(blob)).catch(() => {});
  return new Uint8Array(await blob.arrayBuffer());
}
let sess = null;
async function sessions(device, id) {
  if (sess && sess.device === device) return sess;
  const total = Object.values(FILES).reduce((a, b) => a + b, 0); let got = 0;
  const s = { device };
  for (const n of Object.keys(FILES)) {
    const bytes = await fetchModel(n, b => { got += b; post('progress', { id, stage: `Loading the Yale model · ${(got / 1e6).toFixed(1)} of ${(total / 1e6).toFixed(1)} MB`, frac: Math.min(1, got / total) }); });
    s[n] = await ort.InferenceSession.create(bytes, { executionProviders: [device], graphOptimizationLevel: 'all', logSeverityLevel: 3 });
  }
  return (sess = s);
}

self.onmessage = async ({ data }) => {
  if (data.job !== 'run') return;
  const id = data.id, t0 = performance.now();
  try {
    let device = data.device === 'wasm' ? 'wasm' : (self.navigator.gpu && await self.navigator.gpu.requestAdapter().catch(() => null)) ? 'webgpu' : 'wasm';
    const S = await sessions(device, id).catch(async e => { if (e.missing || device === 'wasm') throw e; post('log', { id, msg: `WebGPU failed (${e.message}); using the CPU` }); device = 'wasm'; return sessions('wasm', id); });
    post('progress', { id, stage: 'Yale model: preparing the brain window', frac: 0 });
    const las = gridToLAS(data.grid), P = prepBrain(las.hu, las.dims, las.sp);
    let mask, segSource;
    if (data.mask) {   // the SAHVAI hematoma mask instead of the Yale SegResNet (faster; not the published pipeline)
      const m = gridToLAS({ ...data.grid, hu: Float32Array.from(data.mask) });
      mask = padMask(m.hu, m.dims, P.preDims); segSource = 'SAHVAI IPH';
    } else {
      mask = await segment(P.brain, async win => {
        const out = await S.yale_seg.run({ x: new ort.Tensor('float32', win, [1, 1, ...YALE.roi]) });
        const a = out.logit.data.slice(); out.logit.dispose?.(); return a;
      }, f => post('progress', { id, stage: `Yale model: segmenting the hematoma (SegResNet, ${device === 'webgpu' ? 'WebGPU' : 'CPU'})`, frac: f }));
      segSource = 'Yale SegResNet';
    }
    let nm = 0; for (const v of mask) nm += v;
    const segVol = nm * P.spacing[0] * P.spacing[1] * P.spacing[2] / 1000;
    post('progress', { id, stage: 'Yale model: classifier input', frac: 0 });
    const x = clsInput(P.brain, mask, P.spacing), xt = new ort.Tensor('float32', x, [1, 2, ...YALE.cls]);
    const run = async mc => { const o = await S.yale_cls.run({ x: xt, mc: new ort.Tensor('float32', new Float32Array([mc]), [1]) }); const v = o.logit.data[0]; o.logit.dispose?.(); return v; };
    const det = await run(0), passes = Math.max(0, data.passes | 0), mcs = [];
    for (let i = 0; i < passes; i++) { post('progress', { id, stage: `Yale model: Monte-Carlo dropout pass ${i + 1} of ${passes}`, frac: i / passes }); mcs.push(await run(1)); }
    const sig = v => 1 / (1 + Math.exp(-v)), ps = mcs.map(sig);
    const mean = mcs.length ? mcs.reduce((a, b) => a + b, 0) / mcs.length : det;
    const pMean = ps.length ? ps.reduce((a, b) => a + b, 0) / ps.length : sig(det);
    const sd = ps.length > 1 ? Math.sqrt(ps.reduce((a, b) => a + (b - pMean) ** 2, 0) / ps.length) : 0;
    const H = p => (p <= 0 || p >= 1) ? 0 : -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
    post('result', { id, device, seconds: (performance.now() - t0) / 1000, segSource, segVolume: segVol, nonEmpty: nm >= 50,
      det: { logit: det, p: sig(det) }, mc: { passes, logits: mcs, p: sig(mean), pMean, sd, entropy: H(sig(mean)) },
      maskGrid: maskToGrid(mask, P.preDims, data.grid), spacing: P.spacing });
  } catch (e) {
    console.error(e);
    post(e.missing ? 'missing' : 'error', { id, msg: e.message || String(e) });
  }
};

// SAHVAI mask (LAS) -> the 512 x 512 x 48 grid: nearest-neighbour z resampling to the prepBrain slice count, then pad/crop
function padMask(v, d, pre) {
  const [X, Y, Z] = d, nz = pre[2], out = new Float32Array(X * Y * nz), sc = nz > 1 ? (Z - 1) / (nz - 1) : 0;
  for (let z = 0; z < nz; z++) { const zs = Math.round(z * sc); out.set(v.subarray(zs * X * Y, (zs + 1) * X * Y), z * X * Y); }
  const [SX, SY, SZ] = YALE.seg, res = new Uint8Array(SX * SY * SZ);
  const off = [X, Y, nz].map((n, a) => n < YALE.seg[a] ? Math.floor((YALE.seg[a] - n) / 2) : -(Math.floor(n / 2) - Math.floor(YALE.seg[a] / 2)));
  for (let z = 0; z < SZ; z++) { const zs = z - off[2]; if (zs < 0 || zs >= nz) continue;
    for (let y = 0; y < SY; y++) { const ys = y - off[1]; if (ys < 0 || ys >= Y) continue;
      for (let x = 0; x < SX; x++) { const xs = x - off[0]; if (xs < 0 || xs >= X) continue; res[x + SX * (y + SY * z)] = out[xs + X * (ys + Y * zs)] > 0 ? 1 : 0; } } }
  return res;
}
