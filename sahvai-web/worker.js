// SAHVAI in the browser: runs the whole pipeline off the main thread.
// In:  {preload, device} | {files, device, angle} | {demo: url, device, angle}
// Out: {type: 'log'|'progress'|'ready'|'result'|'error', ...}
import * as ort from './vendor/ort/ort.webgpu.min.mjs';
import { openInput } from './lib/source.js';
import { pickSeries, seriesVolume } from './lib/series.js';
import { niftiVolume } from './lib/nifti.js';
import { nrrdVolume } from './lib/nrrd.js';
import { toGrid } from './lib/grid.js';
import { analyse } from './lib/pipeline.js';
import { PARTS } from './lib/unet.js';

const base = new URL('./', import.meta.url).href;
ort.env.wasm.wasmPaths = base + 'vendor/ort/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 16) : 1;
ort.env.logLevel = 'error';

const MODEL_VERSION = 'v2';   // bump when the ONNX files change so the browser cache is refreshed
const CACHE = 'sahvai-models-' + MODEL_VERSION;
const SIZES = { p1: 33368, p2: 57861, p3: 374005, p4: 223749, p5: 3285, pm: 25650461, p6: 107413, p7: 57861, p8: 1023 };
const post = (type, x = {}) => self.postMessage({ type, ...x });
const log = msg => post('log', { msg });
const stage = (name, frac) => post('progress', { stage: name, frac });
const MB = x => (x / 1e6).toFixed(1) + ' MB';

async function fetchModel(name, onBytes) {
  const url = new URL(`models/${name}.onnx`, base).href;
  if (self.caches) for (const k of await caches.keys()) if (k.startsWith('sahvai-models-') && k !== CACHE) await caches.delete(k);
  const cache = self.caches ? await caches.open(CACHE).catch(() => null) : null;
  const hit = cache && await cache.match(url);
  if (hit) { const b = new Uint8Array(await hit.arrayBuffer()); onBytes(b.length); return { bytes: b, cached: true }; }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`could not download the model file ${name}.onnx (${r.status})`);
  const rd = r.body.getReader(), parts = []; let n = 0;
  for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); n += value.length; onBytes(value.length); }
  const blob = new Blob(parts);
  if (cache) await cache.put(url, new Response(blob)).catch(() => {});
  return { bytes: new Uint8Array(await blob.arrayBuffer()), cached: false };
}

async function loadModels(device) {
  const t0 = performance.now(), total = Object.values(SIZES).reduce((a, b) => a + b, 0);
  let got = 0, fresh = false;
  const sess = {};
  for (const p of PARTS) {
    const { bytes, cached } = await fetchModel(p, b => { got += b; stage(`Loading the model · ${MB(got)} of ${MB(total)}`, Math.min(1, got / total)); });
    fresh ||= !cached;
    sess[p] = await ort.InferenceSession.create(bytes, { executionProviders: [device], graphOptimizationLevel: 'all', enableMemPattern: false, logSeverityLevel: 3 });
  }
  log(`model (${PARTS.length} parts, ${MB(total)}) ${fresh ? 'downloaded' : 'loaded from the browser cache'} and ready on ${device === 'webgpu' ? 'WebGPU' : `CPU (${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads > 1 ? 's' : ''})`} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return { device, sess };
}

let models = null;
function getModels(device) {
  if (models && models.device === device) return models.p;
  const old = models;
  const p = (async () => {
    if (old) await old.p.then(m => Promise.all(Object.values(m.sess).map(s => s.release())), () => {});
    return loadModels(device);
  })();
  models = { device, p };
  p.catch(() => { if (models && models.p === p) models = null; });
  return p;
}

async function pickDevice(want) {
  if (want === 'wasm') return 'wasm';
  const gpu = self.navigator.gpu && await self.navigator.gpu.requestAdapter().catch(() => null);
  if (gpu) return 'webgpu';
  if (want === 'webgpu') throw new Error('WebGPU is not available in this browser');
  log('WebGPU not available; using the CPU');
  return 'wasm';
}

function caller(M) {
  return async (name, feeds) => {
    const f = {};
    for (const [k, v] of Object.entries(feeds)) f[k] = new ort.Tensor('float32', v.data, v.dims);
    const s = M.sess[name], out = await s.run(f);
    const res = s.outputNames.map(n => ({ data: out[n].data, dims: out[n].dims }));
    for (const n of s.outputNames) out[n].dispose?.();
    return res;
  };
}

let latest = 0;
self.onmessage = async ({ data }) => {
  const t0 = performance.now(), me = ++latest;
  if (data.preload) {
    try {
      const device = await pickDevice(data.device);
      if (me !== latest) return;
      await getModels(device);
      post('ready');
    } catch (e) { log(`model preload failed (${e.message}); will retry when a scan is loaded`); post('ready'); }
    return;
  }
  try {
    let device = await pickDevice(data.device);
    const modelsReady = getModels(device).catch(e => e);
    stage('Reading the scan', 0);
    let vol, info = {};
    if (data.demo) {
      const r = await fetch(new URL(data.demo, base));
      if (!r.ok) throw new Error('could not download the demo case');
      const blob = await r.blob();
      info.source = 'BrainHemoAI demo case (head.nrrd)';
      vol = await nrrdVolume(blob, log);
    } else {
      const inp = await openInput(data.files, log);
      if (inp.kind === 'nifti') { vol = await niftiVolume(inp.file, log); info.source = inp.name; }
      else if (inp.kind === 'nrrd') { vol = await nrrdVolume(inp.file, log); info.source = inp.name; }
      else {
        const pick = await pickSeries(inp.entries, log);
        info = { source: 'DICOM', series_description: pick.best.desc || '', body_part: pick.best.body_part || '', series_selection: pick.why, series_in_upload: pick.list };
        vol = await seriesVolume(pick.best.files, log);
      }
    }
    const g = await toGrid(vol, log, f => stage('Decoding slices', f));
    info.native_size = g.native.dims; info.native_spacing_mm = g.native.spacing.map(s => +s.toFixed(3));
    info.grid_size = [g.nx, g.ny, g.nz]; info.grid_spacing_mm = g.sp.map(s => +s.toFixed(3));

    stage('Loading the model', 0);
    let M = await modelsReady;
    if (M instanceof Error) {
      if (device !== 'webgpu' || data.device === 'webgpu') throw M;
      log(`WebGPU session failed (${M.message}); falling back to the CPU`);
      device = 'wasm'; M = await getModels(device);
    }
    const opts = dev => ({ call: caller(M), log, angle: data.angle, slab: dev === 'webgpu' ? 6 : 4,
      stage: (s, f) => stage(`${s}${s.startsWith('Segmenting') ? ` · ${dev === 'webgpu' ? 'WebGPU' : 'CPU'}` : ''}`, f) });
    let res;
    try { res = await analyse(g, opts(M.device)); }
    catch (e) {
      if (M.device !== 'webgpu' || data.device === 'webgpu') throw e;
      log(`WebGPU run failed (${e.message}); retrying on the CPU`);
      M = await getModels('wasm');
      res = await analyse(g, opts('wasm'));
    }
    const seconds = (performance.now() - t0) / 1000;
    log(`total ${seconds.toFixed(1)}s`);
    const grid = { nx: g.nx, ny: g.ny, nz: g.nz, sp: g.sp, affine: g.affine, hu: g.hu };
    self.postMessage({ type: 'result', grid, labels: res.labels, mask: res.mask, volumes: res.volumes, angle: res.angle, slices: res.slices, windows: res.windows,
      warnings: res.warnings, info, device: M.device, seconds }, [g.hu.buffer, res.labels.buffer, res.mask.buffer]);
  } catch (e) {
    console.error(e);
    post('error', { msg: e && e.message ? e.message : String(e) });
  }
};
