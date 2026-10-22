// SAHVAI Web: four panels (SAHVAI, CLAIM, BLAST-CT, RHUH dataset) sharing one worker, one compute setting and one log.
import { filesFromDataTransfer } from './lib/source.js';
import { niftiGz, toInt16 } from './lib/export.js';
import { Viewer, save } from './lib/viewer.js';

const $ = id => document.getElementById(id);
const logEl = $('log'), t0 = performance.now();
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const mL = v => v == null ? '–' : v < 10 ? v.toFixed(2) : v.toFixed(1);
const r3 = x => x == null ? null : +x.toFixed(3);

function log(msg) {
  const t = ((performance.now() - t0) / 1000).toFixed(2).padStart(7);
  logEl.textContent += `[${t}s] ${msg}\n`; logEl.scrollTop = logEl.scrollHeight;
  console.log('[sahvai]', msg);
}

(async () => {
  const bits = [];
  const gpu = navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null);
  bits.push(gpu ? 'WebGPU available' : 'no WebGPU');
  bits.push(crossOriginIsolated ? `CPU: ${Math.min(navigator.hardwareConcurrency || 4, 16)} threads` : 'CPU: 1 thread (open this page in its own tab for more)');
  $('caps').textContent = bits.join(' · ');
})();
const device = () => document.querySelector('input[name=device]:checked').value;

// ---------- class tables ----------
const C = (key, name, color) => ({ key, name, color });
const CLASSES = {
  sahvai: { 1: C('SAH', 'subarachnoid', [255, 176, 0]), 2: C('IPH', 'intraparenchymal', [255, 45, 120]), 3: C('IVH', 'intraventricular', [30, 200, 255]) },
  claim: { 1: C('SAH', 'subarachnoid', [255, 176, 0]), 2: C('IVH', 'intraventricular', [30, 200, 255]), 3: C('Ventricles', 'CSF in the ventricles', [120, 140, 255]),
    4: C('ICH', 'intracerebral', [255, 45, 120]), 5: C('Aneurysm', 'aneurysm', [190, 90, 255]), 6: C('SDH', 'subdural', [80, 220, 120]) },
  blast: { 1: C('IPH', 'intraparenchymal', [255, 45, 120]), 2: C('EAH', 'extra-axial (SAH, SDH, EDH)', [255, 176, 0]), 3: C('Oedema', 'perilesional oedema', [80, 220, 120]),
    4: C('IVH', 'intraventricular', [30, 200, 255]) },
};
const BLOOD = { sahvai: [1, 2, 3], claim: [1, 2, 4, 6], blast: [1, 2, 4] };
const MODEL_NAME = {
  sahvai: 'Hybrid 2D/3D UNet (Hu et al., NeuroImage 2023;279:120321), BrainHemoAI weights, MIT License',
  claim: 'nnU-Net v2 2D multiclass aSAH model (Kiewitz et al., Front Neurol 2024;15:1490216), CLAIM Charité weights, CC BY-NC 4.0',
  blast: 'BLAST-CT DeepMedic (Monteiro et al., Lancet Digit Health 2020;2:e314-22), biomedia-mira weights, Apache-2.0',
};

function volumesFor(model, r) {
  const v = r.volumes, cl = CLASSES[model];
  if (model === 'sahvai') return [...Object.values(cl).map(c => [c.key, v[c.key], c.name, c.color]), ['Total blood', v.total, 'SAH + IPH + IVH', null, 'total']];
  const cards = Object.entries(cl).map(([, c]) => [c.key, v[c.key], c.name, c.color]);
  const tot = BLOOD[model].reduce((a, i) => a + (v[cl[i].key] || 0), 0);
  cards.push(['Total blood', tot, BLOOD[model].map(i => cl[i].key).join(' + '), null, 'total']);
  return cards;
}

// ---------- worker ----------
let worker = null, jobId = 0, active = null, busy = false;
const panels = {};
function startWorker() {
  if (worker) return;
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => {
    if (data.type === 'log') { log(data.msg); return; }
    const p = active && data.id === active.job ? active.panel : null;
    if (data.type === 'ready') { const q = panels[data.model]; if (q && !busy) q.setStage(data.failed ? 'Ready. The model will be downloaded when you run a scan.' : 'Model ready. Drop a scan or try the demo case.', data.failed ? 0 : 1); return; }
    if (!p) return;
    if (data.type === 'progress') p.setStage(data.stage, data.frac);
    else if (data.type === 'error') { log(`ERROR: ${data.msg}`); p.setStage(`Error: ${data.msg}`, 0); finish(); }
    else if (data.type === 'result') { p.setStage(`Done in ${data.seconds.toFixed(0)}s`, 1); p.show(data); finish(); }
  };
  worker.onerror = e => { log(`worker error: ${e.message}`); active?.panel.setStage(`Error: ${e.message}`, 0); worker = null; finish(); };
}
function submit(panel, msg) {
  if (busy) { panel.setStage('Another analysis is running; wait for it to finish.', 0); panel.status.hidden = false; return false; }
  startWorker();
  busy = true; active = { panel, job: ++jobId, start: performance.now() };
  document.body.classList.add('busy');
  log(`---- ${panel.title}: ${msg.label}`);
  worker.postMessage({ job: 'run', id: active.job, device: device(), ...msg, label: undefined });
  return true;
}
function finish() { busy = false; document.body.classList.remove('busy'); active?.panel.done(); active = null; }
setInterval(() => { if (busy && active) active.panel.showPct(); }, 1000);

// ---------- one model panel ----------
class Panel {
  constructor(cfg) {
    Object.assign(this, cfg);
    this.root = $(`panel-${cfg.id}`);
    const q = s => this.root.querySelector(s);
    this.q = q; this.status = q('.status'); this.results = q('.results'); this.cur = { name: '', frac: 0 };
    this.viewer = new Viewer(q('.viewerhost'), CLASSES[cfg.model]);
    const drop = q('.drop');
    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', async e => { e.preventDefault(); drop.classList.remove('over'); this.runFiles(await filesFromDataTransfer(e.dataTransfer)); });
    for (const sel of ['.f-files', '.f-folder']) { const el = q(sel); if (el) el.onchange = e => { this.runFiles(e.target.files); e.target.value = ''; }; }
    const gtIn = q('.f-gt');
    if (gtIn) gtIn.onchange = e => { this.gtFile = e.target.files[0] || null; q('.gtname').textContent = this.gtFile ? this.gtFile.name : 'none'; };
    this.root.querySelectorAll('[data-demo]').forEach(b => b.onclick = () => this.runDemo(this.demos[+b.dataset.demo]));
  }
  opts() { return this.getOpts ? this.getOpts(this.root) : {}; }
  runFiles(files) {
    files = [...files]; if (!files.length) return;
    const label = `${files.length} file(s): ${files.slice(0, 3).map(f => f.webkitRelativePath || f.name).join(', ')}${files.length > 3 ? ', …' : ''}`;
    if (submit(this, { model: this.model, files, gt: this.gtFile || null, opts: this.opts(), label })) this.begin();
  }
  runDemo(d) {
    if (submit(this, { model: this.model, demo: d.url, demoName: d.name, gt: d.gt || null, gtScheme: d.gtScheme, opts: this.opts(), label: `demo case: ${d.name}` })) this.begin();
  }
  begin() { this.status.hidden = false; this.results.hidden = true; this.q('.drop').classList.add('busy'); this.setStage('Starting', 0); }
  done() { this.q('.drop').classList.remove('busy'); }
  setStage(name, frac) {
    this.cur = { name, frac }; this.status.hidden = false;
    this.q('.stage-name').textContent = name; this.q('.bar i').style.width = `${Math.round(frac * 100)}%`; this.showPct();
  }
  showPct() {
    const pct = this.cur.frac > 0 ? `${Math.round(this.cur.frac * 100)}%` : '';
    const mine = busy && active && active.panel === this;
    this.q('.pct').textContent = mine ? [pct, `${((performance.now() - active.start) / 1000).toFixed(0)}s`].filter(Boolean).join(' · ') : pct;
  }
  show(r) {
    this.R = r;
    const card = ([k, val, sub, c, cls = '']) => `<div class="vol ${cls}" style="--c:${c ? `rgb(${c.join(',')})` : 'var(--mayo-blue)'}"><div class="k">${esc(k)}</div><div class="v">${mL(val)} <small>mL</small></div><div class="s">${esc(sub)}</div></div>`;
    this.q('.vols').innerHTML = volumesFor(this.model, r).map(card).join('');
    const w = r.warnings || [];
    this.q('.warnbox').hidden = !w.length; this.q('.warnbox').innerHTML = w.map(esc).join('<br>');
    this.q('.cmp').hidden = !r.comparison; if (r.comparison) this.q('.cmp').innerHTML = comparisonHTML(this.model, r.comparison, r.gtName);
    const g = r.grid, i = r.info || {};
    this.q('.details').innerHTML = [
      i.series_description !== undefined ? `Series <b>${esc(i.series_description || '?')}</b>${i.body_part ? ` [${esc(i.body_part)}]` : ''} (${esc(i.series_selection)})` : `Source <b>${esc(i.source || '?')}</b>`,
      `native ${i.native_size.join('×')} at ${i.native_spacing_mm.join(' × ')} mm → model grid ${i.grid_size.join('×')} at ${i.grid_spacing_mm.join(' × ')} mm`,
      ...this.detail(r), `${r.device === 'webgpu' ? 'WebGPU' : 'CPU'}, ${r.seconds.toFixed(0)} s`, '<b>research and education use only, not for clinical decisions</b>',
    ].join(' · ');
    this.results.hidden = false;
    this.viewer.show(g, r.labels, { mask: r.mask, gt: r.gt, range: r.slices || null });
    const dl = this.viewer.dl, pre = this.prefix;
    dl.innerHTML = `<button class="btn small" data-k="lab">Label map (.nii.gz)</button><button class="btn small ghost" data-k="ct">CT used (.nii.gz)</button>
      <button class="btn small ghost" data-k="json">Report (.json)</button><button class="btn small ghost" data-k="png">This view (.png)</button>`;
    const legend = Object.entries(CLASSES[this.model]).map(([k, c]) => `${k}=${c.key}`).join(' ');
    dl.querySelector('[data-k=lab]').onclick = async () => save(await niftiGz(r.labels, g, { datatype: 2, description: `${pre} labels: ${legend}`.slice(0, 79) }), `${pre}-labels.nii.gz`);
    dl.querySelector('[data-k=ct]').onclick = async () => save(await niftiGz(toInt16(g.hu), g, { datatype: 4, description: `${pre} model grid (HU)` }), `${pre}-ct.nii.gz`);
    dl.querySelector('[data-k=png]').onclick = () => this.viewer.png(`${pre}-slice-${this.viewer.z + 1}.png`);
    dl.querySelector('[data-k=json]').onclick = () => save(new Blob([JSON.stringify(this.report(r), null, 2)], { type: 'application/json' }), `${pre}-report.json`);
    this.root.querySelector('.results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  detail(r) { return this.details ? this.details(r) : []; }
  report(r) {
    const g = r.grid, vox = g.sp[0] * g.sp[1] * g.sp[2] / 1000, P = g.nx * g.ny, cl = CLASSES[this.model];
    const per = [];
    for (let k = 0; k < g.nz; k++) {
      const c = {}; for (let p = 0; p < P; p++) { const l = r.labels[k * P + p]; if (l) c[l] = (c[l] || 0) + 1; }
      per.push({ slice: k + 1, ...Object.fromEntries(Object.entries(cl).map(([id, x]) => [x.key, r3((c[id] || 0) * vox)])) });
    }
    return {
      disclaimer: 'Research and education use only. Not a medical device and not for diagnosis or treatment decisions.',
      model: MODEL_NAME[this.model], app: 'SAHVAI Web (sahvai-lab.github.io/SAHVAI-Calculator/sahvai-web/), ONNX Runtime Web',
      created: new Date().toISOString(), ...r.info, labels: Object.fromEntries(Object.entries(cl).map(([k, c]) => [k, c.key])),
      volumes_mL: Object.fromEntries(Object.entries(r.volumes).filter(([k, v]) => typeof v === 'number' && k !== 'voxel_mL').map(([k, v]) => [k, r3(v)])), voxel_mL: vox,
      ...(this.extra ? this.extra(r) : {}), comparison_with_reference: r.comparison ? { file: r.gtName, ...r.comparison } : undefined,
      per_slice_mL: per, warnings: r.warnings, device: r.device, seconds: r3(r.seconds),
    };
  }
}

function comparisonHTML(model, c, name) {
  const cl = CLASSES[model], d = x => x == null ? '–' : x.toFixed(3);
  const rows = [`<tr><th>Any blood</th><td>${d(c.blood.dice)}</td><td>${mL(c.blood.model_mL)}</td><td>${mL(c.blood.reference_mL)}</td></tr>`];
  for (const [id, x] of Object.entries(c.classes)) rows.push(`<tr><th>${esc(cl[id] ? cl[id].key : +id === 7 ? 'EDH (reference only)' : `label ${id}`)}</th><td>${d(x.dice)}</td><td>${mL(x.model_mL)}</td><td>${mL(x.reference_mL)}</td></tr>`);
  return `<h3>Agreement with the reference labels <small>${esc(name || '')}${c.scheme === 'binary' ? ' · binary mask: compared as any blood' : ''}${c.mapped === 'bhsd' ? ' · BHSD labels mapped to CLAIM classes' : ''}</small></h3>
    <table><thead><tr><th></th><th>Dice</th><th>Model mL</th><th>Reference mL</th></tr></thead><tbody>${rows.join('')}</tbody></table>
    <p class="small">The reference outline is drawn in white on the image. Dice is computed on the model grid.</p>`;
}

// ---------- the panels ----------
panels.sahvai = new Panel({
  id: 'sahvai', model: 'sahvai', title: 'SAHVAI', prefix: 'sahvai',
  demos: [{ url: 'demo/brainhemoai-demo-head.nrrd', name: 'BrainHemoAI head.nrrd (aneurysmal SAH)' }],
  getOpts: q => ({ angle: q.querySelector('input[name=align]:checked').value === 'manual' ? (Number(q.querySelector('.angle').value) || 0) : null }),
  details: r => [`slices ${r.slices[0] + 1}–${r.slices[1] + 1} of ${r.grid.nz} analysed (${((r.slices[1] - r.slices[0] + 1) * r.grid.sp[2]).toFixed(0)} mm) in ${r.windows.length} window${r.windows.length > 1 ? 's' : ''} of 18`, `rotation ${r.angle}°`],
  extra: r => ({ rotation_deg: r.angle, slices_analysed: [r.slices[0] + 1, r.slices[1] + 1], windows: r.windows.map(k => k + 1) }),
});
document.querySelectorAll('#panel-sahvai input[name=align]').forEach(el => el.addEventListener('change', () => {
  $('panel-sahvai').querySelector('.angle').disabled = document.querySelector('#panel-sahvai input[name=align]:checked').value !== 'manual';
}));

panels.claim = new Panel({
  id: 'claim', model: 'claim', title: 'CLAIM', prefix: 'claim',
  demos: [1, 2, 3].map(k => ({ url: `demo/claim/SAH_00${k}_0000.nii.gz`, gt: `demo/claim/SAH_00${k}_gt.nii.gz`, gtScheme: 'bhsd', name: `BHSD case SAH_00${k} (CLAIM demo)` })),
  getOpts: q => ({ mode: q.querySelector('input[name=claim-mode]:checked').value, mirror: q.querySelector('.mirror').checked }),
  details: r => [`${r.mode === 'exact' ? 'nnU-Net sliding window over the whole slice' : 'nnU-Net sliding window over the head only'}${r.mirror ? ', mirroring ×4' : ''} · ${r.passes} network passes`],
  extra: r => ({ inference: { mode: r.mode, mirroring: r.mirror, passes: r.passes } }),
});

panels.blast = new Panel({
  id: 'blast', model: 'blast', title: 'BLAST-CT', prefix: 'blast-ct',
  demos: [{ url: 'demo/blast/scan_0_image.nii.gz', gt: 'demo/blast/scan_0_segmentation.nii.gz', name: 'BLAST-CT example scan_0 (traumatic brain injury)' }],
  getOpts: q => ({ ensemble: q.querySelector('input[name=blast-ens]:checked').value === '12' }),
  details: r => [`${r.tiles} tiles of 60³ mm, ${r.nModels === 12 ? 'ensemble of 12 models' : 'single model (blast-ct default)'}`],
  extra: r => ({ inference: { models: r.nModels, tiles: r.tiles } }),
});

// ---------- the RHUH dataset panel: a case from Cepeda & Garcia-Garcia (Zenodo 8228847) + its expert mask ----------
class DatasetPanel extends Panel {
  constructor() {
    super({ id: 'rhuh', model: 'sahvai', title: 'RHUH dataset', prefix: 'rhuh-sahvai', demos: [] });
    const q = this.q;
    this.ct = null; this.mask = null;
    const pick = (sel, key, out) => { q(sel).onchange = e => { this[key] = e.target.files[0] || null; q(out).textContent = this[key] ? this[key].name : 'none'; e.target.value = ''; }; };
    pick('.f-ct', 'ct', '.ctname'); pick('.f-mask', 'mask', '.maskname');
    q('.run').onclick = () => this.go();
    q('select.model').onchange = () => this.setModel(q('select.model').value);
    this.model = null; this.setModel('sahvai');
  }
  setModel(model) {
    if (model === this.model) return;
    this.model = model; this.prefix = `rhuh-${panels[model].prefix}`;
    this.details = panels[model].details; this.extra = panels[model].extra;
    this.viewer = new Viewer(this.q('.viewerhost'), CLASSES[model]); this.results.hidden = true;
  }
  runFiles(files) {   // a dropped case: the file named like a mask/segmentation is the expert label, the other the CT
    const q = this.q;
    files = [...files];
    if (files.some(f => /\.rar$/i.test(f.name))) { this.setStage('RAR archives cannot be opened in the browser: extract the archive and drop the two NIfTI files of one case (CT and mask).', 0); return; }
    for (const f of files.filter(f => /\.(nii|nii\.gz|nrrd)$/i.test(f.name))) {
      if (/mask|seg|label|gt|roi|bleed|hemorr|blood/i.test(f.name)) { this.mask = f; q('.maskname').textContent = f.name; }
      else { this.ct = f; q('.ctname').textContent = f.name; }
    }
    if (this.ct) this.setStage(`Case ready: ${this.ct.name}${this.mask ? ` with expert mask ${this.mask.name}` : ' (no expert mask chosen)'}. Pick a model and press Run.`, 0);
  }
  go() {
    if (!this.ct) { this.setStage('Choose or drop the case CT first.', 0); return; }
    this.setModel(this.q('select.model').value);
    const opts = this.model === 'claim' ? { mode: 'fast', mirror: false } : this.model === 'blast' ? { ensemble: false } : { angle: null };
    if (submit(this, { model: this.model, files: [this.ct], gt: this.mask, gtScheme: 'binary', opts, label: `RHUH case ${this.ct.name}${this.mask ? ` + ${this.mask.name}` : ''} with ${this.model}` })) this.begin();
  }
}
panels.rhuh = new DatasetPanel();

// When embedded as a tab of the SAHVAI Calculator, tell the page how tall this content is.
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());
if (window.parent !== window) {
  document.body.classList.add('embedded');
  const report = () => window.parent.postMessage({ type: 'sahvai-web-height', height: document.documentElement.scrollHeight }, location.origin);
  new ResizeObserver(report).observe(document.body); report();
}

// CLAIM's converted weights are added separately; until they are, the panel says so and stays disabled
fetch('models/claim/claim2d.onnx.0', { method: 'HEAD' }).then(r => r.ok, () => false).then(ok => {
  const p = $('panel-claim');
  p.querySelector('.pending').hidden = ok; p.querySelector('.drop').classList.toggle('disabled', !ok);
  const o = $('rhuh-model').querySelector('option[value=claim]'); o.disabled = !ok; if (!ok) o.textContent += ' (weights not installed yet)';
});

// warm up the SAHVAI model (the smallest download) so the first panel is ready
startWorker();
panels.sahvai.setStage('Loading the model', 0);
worker.postMessage({ job: 'preload', model: 'sahvai', device: device() });
document.querySelectorAll('input[name=device]').forEach(el => el.addEventListener('change', () => { if (!busy) worker.postMessage({ job: 'preload', model: 'sahvai', device: device() }); }));
