// SAHVAI Web model pages (CLAIM, BLAST-CT, RHUH dataset): runs the model pipelines off the main thread. One job at a time; sessions are kept per model.
// In:  {job: 'preload', model, device}
//      {job: 'run', id, model: 'sahvai'|'claim'|'blast', device, files | demo (url), gt (File | url), gtScheme, opts}
// Out: {type: 'log'|'progress'|'ready'|'result'|'error', id, model, ...}
import * as ort from './vendor/ort/ort.webgpu.min.mjs';
import { openInput } from './lib/source.js';
import { pickSeries, seriesVolume } from './lib/series.js';
import { niftiVolume } from './lib/nifti.js';
import { nrrdVolume } from './lib/nrrd.js';
import { toGrid } from './lib/grid.js';
import { analyse } from './lib/pipeline.js';
import { PARTS } from './lib/unet.js';
import { runBlast, BLAST_GRID } from './lib/blast.js';
import { runClaim, CLAIM_GRID, CLAIM_INPUT_GRID } from './lib/claim.js';
import { detectStripped, airBackground, brainMaskFor } from './lib/stripped.js';

const base = new URL('./', import.meta.url).href;
ort.env.wasm.wasmPaths = base + 'vendor/ort/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 16) : 1;
ort.env.logLevel = 'error';

// Model files. Bump a version when its ONNX files change so the browser cache is refreshed.
// chunks: the file is stored as name.onnx.0, name.onnx.1, ... (GitHub Pages limits file size) and joined here.
const BLAST_SIZE = 2142772;
export const MODELS = {
  sahvai: { label: 'SAHVAI (BrainHemoAI Hybrid 2D/3D UNet)', version: 'v2', dir: 'models/',
    files: Object.fromEntries(Object.entries({ p1: 33368, p2: 57861, p3: 374005, p4: 223749, p5: 3285, pm: 25650461, p6: 107413, p7: 57861, p8: 1023 }).map(([k, s]) => [k, { size: s }])) },
  blast: { label: 'BLAST-CT (DeepMedic)', version: 'b1', dir: 'models/blast/',
    files: Object.fromEntries(Array.from({ length: 12 }, (_, i) => [`blast_m${i + 1}`, { size: BLAST_SIZE }])) },
  claim: { label: 'CLAIM nnU-Net 2D', version: 'c1', dir: 'models/claim/', files: { claim2d: { size: 92701875, chunks: 2 } } },
};
const BLOOD = { sahvai: [1, 2, 3], blast: [1, 2, 4], claim: [1, 2, 4, 6] };

let cur = { id: 0, model: '' };
const post = (type, x = {}) => self.postMessage({ type, id: cur.id, model: cur.model, ...x });
const log = msg => post('log', { msg });
const stage = (name, frac) => post('progress', { stage: name, frac });
const MB = x => (x / 1e6).toFixed(1) + ' MB';

async function fetchBytes(url, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`could not download ${url.split('/').pop()} (${r.status})`);
  const rd = r.body.getReader(), parts = []; let n = 0;
  for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); n += value.length; onBytes(value.length); }
  return new Blob(parts);
}

async function fetchModel(m, name, onBytes) {
  const M = MODELS[m], f = M.files[name];
  // SAHVAI shares its cache with the SAHVAI page (worker.js); other models keep one cache per version
  const cacheName = m === 'sahvai' ? `sahvai-models-${M.version}` : `sahvai-models-${m}-${M.version}`;
  const url = new URL(`${M.dir}${name}.onnx`, base).href;
  if (self.caches && m !== 'sahvai') for (const k of await caches.keys()) if (k.startsWith(`sahvai-models-${m}-`) && k !== cacheName) await caches.delete(k);
  const cache = self.caches ? await caches.open(cacheName).catch(() => null) : null;
  const hit = cache && await cache.match(url);
  if (hit) { const b = new Uint8Array(await hit.arrayBuffer()); onBytes(b.length); return { bytes: b, cached: true }; }
  let blob;
  if (f.chunks) {
    const parts = [];
    for (let i = 0; i < f.chunks; i++) parts.push(await fetchBytes(`${url}.${i}`, onBytes));
    blob = new Blob(parts);
  } else blob = await fetchBytes(url, onBytes);
  if (cache) await cache.put(url, new Response(blob)).catch(() => {});
  return { bytes: new Uint8Array(await blob.arrayBuffer()), cached: false };
}

const sessions = new Map();   // `${device}|${name}` -> Promise<InferenceSession>
let sessDevice = null;
async function setDevice(device) {
  if (sessDevice === device) return;
  const old = [...sessions.values()]; sessions.clear(); sessDevice = device;
  for (const p of old) await p.then(s => s.release(), () => {});
}
async function getSessions(m, names, device) {
  await setDevice(device);
  const M = MODELS[m], todo = names.filter(n => !sessions.has(`${device}|${n}`));
  const total = todo.reduce((a, n) => a + M.files[n].size, 0), t0 = performance.now();
  let got = 0, fresh = false;
  for (const n of todo) {
    const p = (async () => {
      const { bytes, cached } = await fetchModel(m, n, b => { got += b; stage(`Loading ${M.label} · ${MB(got)} of ${MB(total)}`, Math.min(1, got / total)); });
      fresh ||= !cached;
      return ort.InferenceSession.create(bytes, { executionProviders: [device], graphOptimizationLevel: 'all', enableMemPattern: false, logSeverityLevel: 3 });
    })();
    sessions.set(`${device}|${n}`, p);
    p.catch(() => sessions.delete(`${device}|${n}`));
    await p;
  }
  if (todo.length) log(`${M.label}: ${todo.length} model file${todo.length > 1 ? 's' : ''} (${MB(total)}) ${fresh ? 'downloaded' : 'loaded from the browser cache'} and ready on ${device === 'webgpu' ? 'WebGPU' : `CPU (${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads > 1 ? 's' : ''})`} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  const out = {};
  for (const n of names) out[n] = await sessions.get(`${device}|${n}`);
  return out;
}

async function pickDevice(want) {
  if (want === 'wasm') return 'wasm';
  const gpu = self.navigator.gpu && await self.navigator.gpu.requestAdapter().catch(() => null);
  if (gpu) return 'webgpu';
  if (want === 'webgpu') throw new Error('WebGPU is not available in this browser');
  log('WebGPU not available; using the CPU');
  return 'wasm';
}

function caller(S) {
  return async (name, feeds) => {
    const f = {};
    for (const [k, v] of Object.entries(feeds)) f[k] = new ort.Tensor('float32', v.data, v.dims);
    const s = S[name], out = await s.run(f);
    const res = s.outputNames.map(n => ({ data: out[n].data, dims: out[n].dims }));
    for (const n of s.outputNames) out[n].dispose?.();
    return res;
  };
}

const fileOf = async (x, name) => {
  if (typeof x !== 'string') return x;
  const r = await fetch(new URL(x, base)); if (!r.ok) throw new Error(`could not download ${name || x}`);
  return new File([await r.blob()], x.split('/').pop());
};

async function readVolume(data) {
  let vol, info = {};
  if (data.demo) {
    const f = await fileOf(data.demo, 'the demo case');
    info.source = data.demoName || f.name;
    vol = /\.nrrd$/i.test(f.name) ? await nrrdVolume(f, log) : await niftiVolume(f, log);
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
  return { vol, info };
}

// expert / reference label map on the model's grid (nearest neighbour)
async function readGT(src, gridOpts, g) {
  const f = await fileOf(src, 'the reference labels');
  const v = /\.nrrd$/i.test(f.name) ? await nrrdVolume(f, () => {}) : await niftiVolume(f, () => {});
  const lg = await toGrid(v, () => {}, () => {}, { ...gridOpts, interp: 'nearest', label: undefined });
  if (lg.nx !== g.nx || lg.ny !== g.ny || lg.nz !== g.nz) throw new Error(`the reference labels (${lg.nx}x${lg.ny}x${lg.nz} on the model grid) do not match the scan (${g.nx}x${g.ny}x${g.nz}); they must share the scan's geometry`);
  const out = new Uint8Array(lg.hu.length); let mx = 0;
  for (let i = 0; i < out.length; i++) { const x = Math.max(0, Math.round(lg.hu[i])); out[i] = x > 255 ? 255 : x; if (out[i] > mx) mx = out[i]; }
  return { labels: out, max: mx, name: f.name };
}

function dice(a, b) { let i = 0, s = 0; for (let k = 0; k < a.length; k++) { if (a[k] && b[k]) i++; s += a[k] + b[k]; } return s ? 2 * i / s : null; }
// reference label schemes: a model's own labels, 'binary' (any non-zero = blood) or 'bhsd'
// (BHSD dataset: 1 EDH, 2 IPH, 3 IVH, 4 SAH, 5 SDH), which is mapped onto CLAIM's labels (EDH -> 7, reference only)
const BHSD_TO_CLAIM = [0, 7, 4, 2, 1, 6];
function compare(model, labels, gt, scheme, sp) {
  const vox = sp[0] * sp[1] * sp[2] / 1000, B = new Set(BLOOD[model]);
  let mapped = null;
  if (scheme === 'bhsd') {
    mapped = 'bhsd';
    const m = new Uint8Array(gt.length); for (let i = 0; i < gt.length; i++) m[i] = BHSD_TO_CLAIM[gt[i]] ?? 0;
    gt = m; scheme = model === 'claim' ? 'claim' : 'binary';
  }
  const refBlood = scheme === 'binary' ? new Set(Array.from({ length: 255 }, (_, i) => i + 1)) : new Set([...BLOOD[scheme], ...(scheme === 'claim' ? [7] : [])]);
  const pa = new Uint8Array(labels.length), pb = new Uint8Array(labels.length); let na = 0, nb = 0;
  for (let i = 0; i < labels.length; i++) { if (B.has(labels[i])) { pa[i] = 1; na++; } if (refBlood.has(gt[i])) { pb[i] = 1; nb++; } }
  const res = { scheme, mapped, blood: { dice: dice(pa, pb), model_mL: na * vox, reference_mL: nb * vox }, classes: {} };
  if (scheme === model) {
    const n1 = new Float64Array(256), n2 = new Float64Array(256), both = new Float64Array(256);
    for (let i = 0; i < labels.length; i++) { const p = labels[i], q = gt[i]; n1[p]++; n2[q]++; if (p === q) both[p]++; }
    for (let c = 1; c < 256; c++) {
      if (!n2[c] && !(n1[c] && B.has(c))) continue;   // classes in the reference, plus blood classes the model found
      res.classes[c] = { dice: n1[c] + n2[c] ? 2 * both[c] / (n1[c] + n2[c]) : null, model_mL: n1[c] * vox, reference_mL: n2[c] * vox };
    }
  }
  return res;
}

async function runModel(model, g, S, dev, data) {
  const o = data.opts || {}, call = caller(S);
  const st = (s, f) => stage(`${s}${s.startsWith('Segmenting') ? ` · ${dev === 'webgpu' ? 'WebGPU' : 'CPU'}` : ''}`, f);
  if (model === 'sahvai') return analyse(g, { call, log, angle: o.angle ?? null, mask: data._stripped ? await brainMaskFor(data._vol) : null, slab: dev === 'webgpu' ? 6 : 4, stage: st });
  if (model === 'blast') return runBlast(g, { call, models: Object.keys(S), log, stage: st });
  if (model === 'claim') {
    const gi = await toGrid(data._vol, () => {}, f => stage('Preparing the network input', f), CLAIM_INPUT_GRID);
    return runClaim(g, { call, input: gi.hu, mode: o.mode || 'exact', mirror: !!o.mirror, log, stage: st });
  }
}

const GRID = { sahvai: {}, blast: BLAST_GRID, claim: CLAIM_GRID };
function modelFiles(model, opts = {}) {
  if (model === 'blast') return Object.keys(MODELS.blast.files).slice(0, opts.ensemble ? 12 : 1);
  return Object.keys(MODELS[model].files);
}

// jobs run one after another (a run that arrives during a preload waits for it)
let chain = Promise.resolve();
self.onmessage = ({ data }) => { chain = chain.then(() => handle(data)); };
async function handle(data) {
  const t0 = performance.now();
  cur = { id: data.id || 0, model: data.model };
  const model = data.model;
  if (data.job === 'preload') {
    try { const device = await pickDevice(data.device); await getSessions(model, modelFiles(model, data.opts), device); post('ready'); }
    catch (e) { log(`${MODELS[model].label}: preload failed (${e.message}); will retry when a scan is loaded`); post('ready', { failed: true }); }
    return;
  }
  try {
    let device = await pickDevice(data.device);
    const names = modelFiles(model, data.opts);
    const ready = getSessions(model, names, device).catch(e => e);
    stage('Reading the scan', 0);
    let { vol, info } = await readVolume(data);
    const stripped = await detectStripped(vol);
    if (stripped) { vol = airBackground(vol); info.skull_stripped = true; log('skull-stripped volume (background 0, no air or bone): background set to -1000 HU' + (model === 'sahvai' ? '; brain mask taken from the non-background voxels' : '')); }
    data._vol = vol; data._stripped = stripped;
    const g = await toGrid(vol, log, f => stage('Decoding slices', f), GRID[model]);
    info.native_size = g.native.dims; info.native_spacing_mm = g.native.spacing.map(s => +s.toFixed(3));
    info.grid_size = [g.nx, g.ny, g.nz]; info.grid_spacing_mm = g.sp.map(s => +s.toFixed(3));
    // a check on the intensities: these models expect Hounsfield units
    let lo = Infinity, hi = -Infinity; for (let i = 0; i < g.hu.length; i += 7) { const v = g.hu[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    info.intensity_range = [+lo.toFixed(1), +hi.toFixed(1)];
    const warnHU = hi - lo < 300 ? `The intensities span only ${lo.toFixed(1)} to ${hi.toFixed(1)}; the models expect Hounsfield units (air about -1000, bone above 300), so this volume looks intensity-normalised and results are unreliable.` : null;
    let gt = null;
    if (data.gt) { stage('Reading the reference labels', 0); gt = await readGT(data.gt, GRID[model], g); log(`reference labels ${gt.name}: values 0-${gt.max}`); }

    stage(`Loading ${MODELS[model].label}`, 0);
    let S = await ready;
    if (S instanceof Error) {
      if (device !== 'webgpu' || data.device === 'webgpu') throw S;
      log(`WebGPU session failed (${S.message}); falling back to the CPU`);
      device = 'wasm'; S = await getSessions(model, names, device);
    }
    let res;
    try { res = await runModel(model, g, S, device, data); }
    catch (e) {
      if (device !== 'webgpu' || data.device === 'webgpu') throw e;
      log(`WebGPU run failed (${e.message}); retrying on the CPU`);
      device = 'wasm'; S = await getSessions(model, names, device);
      res = await runModel(model, g, S, device, data);
    }
    const warnings = [...(res.warnings || [])]; if (warnHU) warnings.unshift(warnHU);
    if (stripped) warnings.unshift('This scan is skull-stripped (brain only, background 0), as in the RHUH dataset. The models were trained on whole-head CT, so the background was set to air (-1000 HU)' + (model === 'sahvai' ? ' and the brain mask was taken from the scan' : '') + '; treat the result as approximate.');
    const scheme = data.gtScheme || (gt && gt.max <= 1 ? 'binary' : model);
    const comparison = gt ? compare(model, res.labels, gt.labels, scheme, g.sp) : null;
    if (comparison) log(`vs reference: any-blood Dice ${comparison.blood.dice == null ? 'n/a' : comparison.blood.dice.toFixed(3)} (model ${comparison.blood.model_mL.toFixed(1)} mL, reference ${comparison.blood.reference_mL.toFixed(1)} mL)`);
    const seconds = (performance.now() - t0) / 1000;
    log(`total ${seconds.toFixed(1)}s`);
    const grid = { nx: g.nx, ny: g.ny, nz: g.nz, sp: g.sp, affine: g.affine, hu: g.hu };
    const transfer = [g.hu.buffer, res.labels.buffer];
    if (res.mask) transfer.push(res.mask.buffer);
    if (gt) transfer.push(gt.labels.buffer);
    self.postMessage({ type: 'result', id: cur.id, model, grid, labels: res.labels, mask: res.mask || null, gt: gt ? gt.labels : null, gtName: gt ? gt.name : null,
      comparison, volumes: res.volumes, angle: res.angle, slices: res.slices, windows: res.windows, mode: res.mode, mirror: res.mirror, nModels: model === 'blast' ? names.length : undefined,
      tiles: res.tiles, passes: res.passes, warnings, info, device, seconds }, transfer);
  } catch (e) {
    console.error(e);
    post('error', { msg: e && e.message ? e.message : String(e) });
  }
}
