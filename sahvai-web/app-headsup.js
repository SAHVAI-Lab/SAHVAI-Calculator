// HEADS-UP page (headsup.html): ICH detection + 5-subtype classification, then hand-off of the scan to a
// SAHVAI Web segmentation tool. Runs in worker-models.js (model 'headsup').
import { filesFromDataTransfer } from './lib/source.js';
import { niftiGz, toInt16 } from './lib/export.js';
import { Viewer, save } from './lib/viewer.js';
import { CLASS_KEYS, CLASS_NAMES, T_DETECT, T_POSSIBLE, MODEL_SPECS } from './lib/headsup.js';
import { putHandoff, openTarget, TARGETS, ROUTES } from './lib/handoff.js';

const $ = id => document.getElementById(id);
const root = $('panel-headsup'), q = s => root.querySelector(s);
const logEl = $('log'), t0 = performance.now();
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = v => `${(v * 100).toFixed(v >= 0.995 || v < 0.005 ? 0 : v < 0.1 ? 1 : 0)}%`;
const r4 = x => +x.toFixed(4);
function log(msg) {
  const t = ((performance.now() - t0) / 1000).toFixed(2).padStart(7);
  logEl.textContent += `[${t}s] ${msg}\n`; logEl.scrollTop = logEl.scrollHeight; console.log('[headsup]', msg);
}

(async () => {
  const gpu = navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null);
  $('caps').textContent = [gpu ? 'WebGPU available' : 'no WebGPU',
    crossOriginIsolated ? `CPU: ${Math.min(navigator.hardwareConcurrency || 4, 16)} threads` : 'CPU: 1 thread (open this page in its own tab for more)'].join(' · ');
})();
// phones and tablets (iPadOS reports itself as a Mac with touch): lighter settings, and the engine is released after each run
const LITE = /iPhone|iPad|iPod|Android|Mobile/i.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
// On phones and tablets Auto means the CPU: WebGPU ran the model but then crashed iPhone Safari (out of memory, Nov 2026),
// while the CPU completes the same scan in about 30 s with identical results. WebGPU can still be chosen explicitly.
const device = () => { const v = document.querySelector('input[name=device]:checked').value; return LITE && v === 'auto' ? 'wasm' : v; };
if (LITE) {
  const auto = document.querySelector('input[name=device][value=auto]');
  if (auto) auto.parentElement.lastChild.textContent = ' Auto (CPU on this device)';
  const gpu = document.querySelector('input[name=device][value=webgpu]');
  if (gpu) gpu.parentElement.title = 'WebGPU has crashed iPhone Safari with this model (out of memory); the CPU is the safe choice on phones';
}
const ensemble = () => document.querySelector('input[name=hu-ens]:checked').value === '3';

// subtypes: label value in the localisation map, colour
const SUB = [
  null,
  { key: 'EDH', name: 'epidural', color: [190, 90, 255] },
  { key: 'IPH', name: 'intraparenchymal', color: [255, 45, 120] },
  { key: 'IVH', name: 'intraventricular', color: [30, 200, 255] },
  { key: 'SAH', name: 'subarachnoid', color: [255, 176, 0] },
  { key: 'SDH', name: 'subdural', color: [80, 220, 120] },
];
const VCLASSES = Object.fromEntries(SUB.map((c, i) => [i, c]).filter(([, c]) => c));

const DEMOS = [
  { url: 'demo/claim/SAH_002_0000.nii.gz', name: 'BHSD case SAH_002 (aneurysmal SAH with IVH and ICH)', gt: 'demo/claim/SAH_002_gt.nii.gz' },
  { url: 'demo/deepbleed/jhu_mistie_01_ct.nii.gz', name: 'MISTIE/CLEAR ICH case 01 (JHU, CC BY-NC 4.0)' },
  { url: 'demo/blast/scan_0_image.nii.gz', name: 'BLAST-CT scan_0 (traumatic brain injury)' },
  { url: 'demo/brainhemoai-demo-head.nrrd', name: 'BrainHemoAI head.nrrd (aneurysmal SAH)' },
];

// ---------- viewer: the shared axial viewer, with slice probabilities instead of volumes ----------
class HUViewer extends Viewer {
  render() {
    super.render();
    if (!this.P) return;
    const k = this.z, p = this.P, nz = this.R.g.nz;
    this.info.textContent = [`slice ${k + 1}/${nz}`, `any ${pct(p[k * 6])}`,
      ...[1, 2, 3, 4, 5].filter(c => p[k * 6 + c] >= T_POSSIBLE).map(c => `${CLASS_KEYS[c]} ${pct(p[k * 6 + c])}`)].join(' · ');
  }
  chart() {   // per-slice probability of each type (replaces the volume-per-slice chart)
    const R = this.R, P = this.P; if (!R || !P) return;
    const cv = this.chartCv, dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (!W) return;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext('2d'); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, W, H);
    const n = R.g.nz, bw = W / n, top = 6, bot = H - 14, y = v => bot - v * (bot - top);
    ctx.fillStyle = '#fdecef'; ctx.fillRect(0, y(1), W, y(T_DETECT) - y(1));
    ctx.strokeStyle = '#d9a3ae'; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(0, y(T_DETECT)); ctx.lineTo(W, y(T_DETECT)); ctx.stroke(); ctx.setLineDash([]);
    const line = (c, col, w) => { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); for (let k = 0; k < n; k++) { const X = (k + 0.5) * bw, Y = y(P[k * 6 + c]); k ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); } ctx.stroke(); };
    line(0, '#00205b', 2.5);
    for (let c = 1; c < 6; c++) line(c, `rgb(${SUB[c].color.join(',')})`, 1.8);
    ctx.lineWidth = 2; ctx.strokeStyle = '#003da5'; ctx.strokeRect(this.z * bw + 0.5, 1, Math.max(2, bw - 1), H - 2);
    ctx.fillStyle = '#5f6673'; ctx.font = '10px system-ui'; ctx.textAlign = 'left';
    ctx.fillText('inferior', 3, H - 3); ctx.textAlign = 'right'; ctx.fillText('superior', W - 3, H - 3); ctx.fillText('p = 0.5', W - 3, y(T_DETECT) - 3);
  }
}
const viewer = new HUViewer(q('.viewerhost'), VCLASSES);
// relabel the shared viewer's controls for a detector
(() => {
  const ov = q('.viewerhost .overlay'); if (ov) ov.parentElement.lastChild.textContent = ' Localisation overlay (approximate)';
  const tip = q('.viewerhost .tip'); if (tip) tip.textContent = 'Scroll over the image or use ↑/↓ to change slice. The chart shows each type\'s probability per slice (dark line: any hemorrhage; shaded: above 0.5); click it to jump.';
  const ch = q('.viewerhost .chart'); if (ch) { ch.height = 150; ch.style.height = '150px'; ch.setAttribute('aria-label', 'Hemorrhage probability per slice'); }
  const lg = q('.viewerhost .legend'); if (lg) lg.insertAdjacentHTML('afterbegin', '<span><i style="background:#00205b"></i>any</span>');
})();

// ---------- worker ----------
let worker = null, busy = false, jobId = 0, cur = { name: '', frac: 0 }, started = 0, lastInput = null, R = null;
function startWorker() {
  if (worker) return;
  worker = new Worker(new URL('./worker-models.js?v=h3', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => {
    if (data.type === 'log') { log(data.msg); return; }
    if (data.type === 'ready') { if (!busy) setStage(data.failed ? 'Ready. The model downloads when you run a scan.' : 'Model ready. Drop a scan or try a demo case.', data.failed ? 0 : 1); return; }
    if (data.id !== jobId) return;
    if (data.type === 'progress') setStage(data.stage, data.frac);
    else if (data.type === 'error') {
      log(`ERROR: ${data.msg}`);
      // a failed engine start (e.g. out of memory) is sticky inside a worker: start a fresh one next time
      stopWorker();
      setStage(/initWasm|backend|memory|allocation/i.test(data.msg)
        ? 'The browser could not start the AI engine, usually for lack of memory. Close other tabs and try again; on a phone, the DenseNet121 model and the CPU setting use the least memory.'
        : `Error: ${data.msg}`, 0);
      finish();
    }
    else if (data.type === 'result') {
      setStage(`Done in ${data.seconds.toFixed(0)}s`, 1);
      if (LITE) stopWorker();   // free the engine's memory before the results are drawn
      show(data); finish();
    }
  };
  worker.onerror = e => { log(`worker error: ${e.message}`); setStage(`Error: ${e.message}`, 0); stopWorker(); finish(); };
}
function stopWorker() { if (worker) { worker.terminate(); worker = null; } }
function setStage(name, frac) {
  cur = { name, frac }; q('.status').hidden = false;
  q('.stage-name').textContent = name; q('.bar i').style.width = `${Math.round(frac * 100)}%`; showPct();
}
function showPct() {
  const p = cur.frac > 0 ? `${Math.round(cur.frac * 100)}%` : '';
  q('.pct').textContent = busy ? [p, `${((performance.now() - started) / 1000).toFixed(0)}s`].filter(Boolean).join(' · ') : p;
}
setInterval(() => { if (busy) showPct(); }, 1000);
function finish() { busy = false; document.body.classList.remove('busy'); q('.drop').classList.remove('busy'); }

function submit(input, label) {
  if (busy) { setStage('An analysis is running; wait for it to finish.', 0); return; }
  startWorker();
  busy = true; started = performance.now(); jobId++; lastInput = { ...input, label };
  document.body.classList.add('busy'); q('.drop').classList.add('busy'); q('.results').hidden = true; setStage('Starting', 0);
  log(`---- HEADS-UP: ${label}`);
  worker.postMessage({ job: 'run', id: jobId, model: 'headsup', device: device(), lite: LITE, opts: { ensemble: ensemble() }, ...input });
}
const runFiles = files => {
  files = [...files]; if (!files.length) return;
  submit({ files }, `${files.length} file(s): ${files.slice(0, 3).map(f => f.webkitRelativePath || f.name).join(', ')}${files.length > 3 ? ', …' : ''}`);
};
const runDemo = d => submit({ demo: d.url, demoName: d.name }, `demo case: ${d.name}`);

const drop = q('.drop');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', async e => { e.preventDefault(); drop.classList.remove('over'); runFiles(await filesFromDataTransfer(e.dataTransfer)); });
for (const sel of ['.f-files', '.f-folder']) q(sel).onchange = e => { runFiles(e.target.files); e.target.value = ''; };
root.querySelectorAll('[data-demo]').forEach(b => b.onclick = () => runDemo(DEMOS[+b.dataset.demo]));
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());

// ---------- results ----------
function recommend(summary) {
  const score = {}, why = {};
  for (const s of summary.slice(1)) {
    if (s.call === 'not detected') continue;
    (ROUTES[s.key] || []).forEach((t, i) => {
      score[t] = (score[t] || 0) + (s.call === 'detected' ? 2 : 1) * (3 - i) * s.max;
      (why[t] ||= []).push(s.key);
    });
  }
  // the lead tool by the dominant pattern: SAH -> SAHVAI (separate SAH, IPH, IVH volumes); ICH without SAH -> DeepBleed (MISTIE);
  // extra-axial blood alone (SDH / EDH) -> BLAST-CT (trained on traumatic brain injury)
  const det = k => summary.find(s => s.key === k).call === 'detected';
  const lead = det('SAH') ? 'web' : det('IPH') ? 'deepbleed' : det('IVH') ? 'web' : det('SDH') || det('EDH') ? 'blast' : null;
  if (lead) score[lead] = (score[lead] || 0) + 100;
  return Object.keys(TARGETS).sort((a, b) => (score[b] || 0) - (score[a] || 0)).map(k => ({ key: k, score: score[k] || 0, why: why[k] || [] }));
}

function show(r) {
  R = r; window.HEADSUP_LAST = r;   // (also used by the automated tests)
  const H = r.headsup, S = H.summary, any = S[0], g = r.grid;
  const call = any.call, cls = call === 'detected' ? 'pos' : call === 'possible' ? 'maybe' : 'neg';
  const found = S.slice(1).filter(s => s.call !== 'not detected');
  q('.verdict').className = `verdict ${cls}`;
  q('.verdict').innerHTML = `<div class="vt">${call === 'detected' ? 'Intracranial hemorrhage detected' : call === 'possible' ? 'Possible intracranial hemorrhage: review' : 'No intracranial hemorrhage detected'}</div>
    <div class="vs">any hemorrhage ${pct(any.max)} (highest slice probability)${any.range ? ` · slices ${any.range[0] + 1}–${any.range[1] + 1} of ${g.nz} above ${T_DETECT}` : ''}${found.length ? ` · ${found.map(s => `<b>${s.key}</b>${s.call === 'possible' ? '?' : ''}`).join(', ')}` : ''}</div>`;
  q('.subtypes').innerHTML = S.slice(1).map((s, i) => {
    const c = SUB[i + 1];
    return `<div class="vol sub ${s.call.replace(' ', '-')}" style="--c:rgb(${c.color.join(',')})" data-slice="${s.slice}" title="Jump to the slice where ${s.key} is most likely">
      <div class="k">${s.key}</div><div class="v">${pct(s.max)}</div>
      <div class="s">${esc(c.name)} · <b>${s.call}</b>${s.range ? ` · slices ${s.range[0] + 1}–${s.range[1] + 1}` : ''}</div></div>`;
  }).join('');
  q('.subtypes').querySelectorAll('[data-slice]').forEach(el => el.onclick = () => viewer.setSlice(+el.dataset.slice));
  const w = r.warnings || [];
  if (g.nz < 12) w.push(`Only ${g.nz} slices: the networks were trained on whole-head axial studies, so a partial scan may be misread.`);
  q('.warnbox').hidden = !w.length; q('.warnbox').innerHTML = w.map(esc).join('<br>');

  // step 2: segmentation tools, ranked by the types found
  const rec = recommend(S), top = rec.filter(t => t.score > 0);
  q('.next').innerHTML = `<h3>Step 2 · send this scan to a segmentation tool</h3>
    <p class="small">${top.length ? `Suggested for <b>${esc(found.map(s => s.key).join(', '))}</b>, best match first.` : 'HEADS-UP found nothing above its thresholds; you can still segment the scan with any tool.'}
    The tool opens with the same scan (${esc(lastInput.label.replace(/^demo case: /, ''))}) and runs it; nothing is uploaded.</p>
    <div class="targets">${rec.map((t, i) => `<button class="target${t.score > 0 ? ' rec' : ''}${i === 0 && t.score > 0 ? ' first' : ''}" data-t="${t.key}">
      <b>${esc(TARGETS[t.key].name)}</b><span>${esc(TARGETS[t.key].does)}</span>${t.why.length ? `<em>for ${esc(t.why.join(', '))}</em>` : ''}</button>`).join('')}</div>`;
  q('.next').querySelectorAll('[data-t]').forEach(b => b.onclick = () => handOff(b.dataset.t));

  viewer.P = H.probs;
  viewer.show(g, r.labels, {});
  let best = any.slice; viewer.setSlice(best);
  const dl = viewer.dl;
  dl.innerHTML = `<button class="btn small" data-k="json">Report (.json)</button><button class="btn small ghost" data-k="lab">Localisation labels (.nii.gz)</button>
    <button class="btn small ghost" data-k="ct">CT analysed (.nii.gz)</button><button class="btn small ghost" data-k="png">This view (.png)</button>`;
  dl.querySelector('[data-k=json]').onclick = () => save(new Blob([JSON.stringify(report(r), null, 2)], { type: 'application/json' }), 'heads-up-report.json');
  dl.querySelector('[data-k=lab]').onclick = async () => save(await niftiGz(r.labels, g, { datatype: 2, description: 'HEADS-UP approx. localisation 1EDH 2IPH 3IVH 4SAH 5SDH' }), 'heads-up-labels.nii.gz');
  dl.querySelector('[data-k=ct]').onclick = async () => save(await niftiGz(toInt16(g.hu), g, { datatype: 4, description: 'HEADS-UP analysed grid (HU)' }), 'heads-up-ct.nii.gz');
  dl.querySelector('[data-k=png]').onclick = () => viewer.png(`heads-up-slice-${viewer.z + 1}.png`);
  const i = r.info || {};
  q('.details').innerHTML = [
    i.series_description !== undefined ? `Series <b>${esc(i.series_description || '?')}</b> (${esc(i.series_selection)})` : `Source <b>${esc(i.source || '?')}</b>`,
    `native ${i.native_size.join('×')} at ${i.native_spacing_mm.join(' × ')} mm → ${g.nz} slices of ${g.sp[2].toFixed(1)} mm`,
    `brain window ${H.window.join('/')} HU${i.dicom_window ? ' (DICOM)' : ''}`,
    `${H.models.map(m => MODEL_SPECS[m].label).join(' + ')}`,
    `thresholds: detected ≥ ${T_DETECT}, possible ≥ ${T_POSSIBLE} (provisional)`,
    `${r.device === 'webgpu' ? 'WebGPU' : 'CPU'}, ${r.seconds.toFixed(0)} s`, '<b>research and education use only, not for clinical decisions</b>',
  ].join(' · ');
  q('.results').hidden = false;
  q('.results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function report(r) {
  const H = r.headsup, nz = r.grid.nz;
  return {
    disclaimer: 'Research and education use only. Not a medical device and not for diagnosis, triage or treatment decisions.',
    tool: 'HEADS-UP (Hemorrhage Evaluation and Detector System for Underserved Populations), SAHVAI Web',
    models: H.models.map(m => `${MODEL_SPECS[m].label}, RSNA 2019 1st-place solution (Wang et al., NeuroImage Clin 2021;32:102785), fold 0`),
    created: new Date().toISOString(), input: lastInput ? lastInput.label : null, ...r.info,
    brain_window_HU: H.window, thresholds: { detected: T_DETECT, possible: T_POSSIBLE, note: 'provisional; not calibrated on an external cohort' },
    study: Object.fromEntries(H.summary.map(s => [s.key, { probability: r4(s.max), call: s.call, peak_slice: s.slice + 1, slices_above_threshold: s.n_slices, range: s.range && s.range.map(k => k + 1) }])),
    per_slice: Array.from({ length: nz }, (_, k) => ({ slice: k + 1, ...Object.fromEntries(CLASS_KEYS.map((c, j) => [c, r4(H.probs[k * 6 + j])])) })),
    per_model: H.models.length > 1 ? Object.fromEntries(H.models.map(m => [m, CLASS_KEYS.map((c, j) => { let mx = 0; for (let k = 0; k < nz; k++) mx = Math.max(mx, H.perModel[m][k * 6 + j]); return r4(mx); })])) : undefined,
    localisation_labels: { 1: 'EDH', 2: 'IPH', 3: 'IVH', 4: 'SAH', 5: 'SDH', note: 'approximate: hyperdense voxels in class-activation hot spots; not a segmentation' },
    classes: Object.fromEntries(CLASS_KEYS.map((c, j) => [c, CLASS_NAMES[j]])),
    warnings: r.warnings, device: r.device, seconds: r4(r.seconds),
  };
}

// ---------- step 2: hand the scan to a segmentation tool ----------
async function handOff(key) {
  if (!lastInput || !R) return;
  const entry = { from: 'HEADS-UP', label: lastInput.label.replace(/^demo case: /, ''), summary: R.headsup.summary.map(s => ({ key: s.key, max: s.max, call: s.call, range: s.range })) };
  if (lastInput.files) entry.files = lastInput.files; else { entry.demo = lastInput.demo; entry.demoName = lastInput.demoName; }
  try {
    const id = await putHandoff(entry);
    log(`hand-off ${id} to ${TARGETS[key].name}`);
    openTarget(key, id);
  } catch (e) {
    log(`hand-off failed: ${e.message}`);
    setStage(`Could not pass the scan to ${TARGETS[key].name} (${e.message}). Open that tool and drop the same files there.`, 0);
  }
}

// ---------- embedded as a tab of the SAHVAI Calculator ----------
if (window.parent !== window) {
  document.body.classList.add('embedded');
  const rep = () => window.parent.postMessage({ type: 'sahvai-web-height', page: 'headsup', height: Math.ceil(document.body.getBoundingClientRect().height) + 24 }, location.origin);
  new ResizeObserver(rep).observe(document.body); rep();
}

// warm up the phone model (14 MB) on computers; on phones it loads at the first run, so no memory is held while idle
if (!LITE) {
  startWorker();
  setStage('Loading the model', 0);
  worker.postMessage({ job: 'preload', model: 'headsup', device: device(), opts: { ensemble: false } });
} else setStage('Ready. The 14 MB model downloads at the first run, then is cached.', 0);
document.querySelectorAll('input[name=hu-ens]').forEach(el => el.addEventListener('change', () => {
  if (!busy && ensemble()) setStage('The ensemble (134 MB) downloads on the next run and is then cached.', 0);
}));
