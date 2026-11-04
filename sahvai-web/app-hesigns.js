// ICH expansion signs page: SAHVAI segmentation (worker.js) -> lib/hesigns.js -> signs, scores, DeepMedX patches.
import { filesFromDataTransfer } from './lib/source.js';
import { niftiGz, toInt16 } from './lib/export.js';
import { receivedHandoff, handoffBanner, handoffFiles } from './lib/handoff.js';
import { Viewer, save } from './lib/viewer.js';
import { mountResearch, publicInfo } from './lib/research.js';
import { analyseSigns, deepmedxROI, batScore, brainScore, expansion, NEW_IVH_mL, OVERLAY, SIGN_INFO, CRITERIA } from './lib/hesigns.js';

const $ = id => document.getElementById(id), q = s => document.querySelector(s);
const logEl = $('log'), t0 = performance.now();
let worker = null, busy = false, runStart = 0, cur = { frac: 0 }, R = null, lastLabel = '', role = 'baseline', FU = null, yaleWorker = null, yaleId = 0;
const override = {};
const DEMOS = [
  { url: 'demo/claim/SAH_002_0000.nii.gz', name: 'BHSD case SAH_002 (ICH + IVH + SAH)' },
  { url: 'demo/deepbleed/jhu_mistie_01_ct.nii.gz', name: 'MISTIE/CLEAR ICH case 01 (JHU, CC BY-NC 4.0)' },
];
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const r1 = x => +x.toFixed(1);

function log(msg) {
  const t = ((performance.now() - t0) / 1000).toFixed(2).padStart(7);
  logEl.textContent += `[${t}s] ${msg}\n`; logEl.scrollTop = logEl.scrollHeight; console.log('[hesigns]', msg);
}
(async () => {
  const gpu = navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null);
  $('caps').textContent = [gpu ? 'WebGPU available' : 'no WebGPU', crossOriginIsolated ? `CPU: ${Math.min(navigator.hardwareConcurrency || 4, 16)} threads` : 'CPU: 1 thread (open this page in its own tab for more)'].join(' · ');
})();

const device = () => q('input[name=device]:checked').value;
const angle = () => q('input[name=align]:checked').value === 'manual' ? (Number($('angle').value) || 0) : null;
const includeIVH = () => q('input[name=hem]:checked').value === 'ivh';
document.querySelectorAll('input[name=align]').forEach(el => el.addEventListener('change', () => { $('angle').disabled = angle() === null; }));

function startWorker() {
  if (worker) return;
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = onMessage;
  worker.onerror = e => { log(`worker error: ${e.message}`); setStage(`Error: ${e.message}`, 0); done(); };
}
startWorker();
q('.status').hidden = false; setStage('Loading the segmentation model', 0);
worker.postMessage({ preload: true, device: device() });

function setStage(name, frac) { cur = { frac }; q('.stage-name').textContent = name; q('.bar i').style.width = `${Math.round(frac * 100)}%`; showPct(); }
function showPct() { const pct = cur.frac > 0 ? `${Math.round(cur.frac * 100)}%` : ''; q('.pct').textContent = busy ? [pct, `${((performance.now() - runStart) / 1000).toFixed(0)}s`].filter(Boolean).join(' · ') : pct; }
setInterval(() => { if (busy) showPct(); }, 1000);

function begin(label) {
  busy = true; runStart = performance.now(); lastLabel = label;
  q('.drop').classList.add('busy'); q('.status').hidden = false; if (role !== 'followup') q('.results').hidden = true;
  log('---- ' + label); setStage('Starting', 0); startWorker();
}
function done() { busy = false; q('.drop').classList.remove('busy'); }
function run(files, label, asRole = 'baseline') {
  files = [...files]; if (!files.length || busy) return;
  if (asRole === 'followup' && !R) { setStage('Load the baseline CT first, then the follow-up.', 0); return; }
  role = asRole;
  if (asRole === 'followup') q('.fudrop').classList.add('busy');
  begin(label || `${files.length} file(s): ${files.slice(0, 3).map(f => f.webkitRelativePath || f.name).join(', ')}${files.length > 3 ? ', …' : ''}`);
  worker.postMessage({ files, device: device(), angle: angle(), wantRaw: true });
}
async function runDemo(d) {
  if (busy) return;
  begin(`demo case: ${d.name}`); setStage('Downloading the demo case', 0);
  try { const r = await fetch(d.url); if (!r.ok) throw new Error(`could not download ${d.url}`); const f = new File([await r.blob()], d.url.split('/').pop()); busy = false; run([f], `demo case: ${d.name}`); }
  catch (e) { setStage(`Error: ${e.message}`, 0); done(); }
}
q('.demos').querySelectorAll('[data-demo]').forEach(b => b.onclick = () => runDemo(DEMOS[+b.dataset.demo]));

function onMessage({ data }) {
  if (data.type === 'log') log(data.msg);
  else if (data.type === 'ready') { if (!busy) setStage('Model ready. Drop a scan or try a demo case.', 1); }
  else if (data.type === 'progress') setStage(data.stage, data.frac);
  else if (data.type === 'error') { log(`ERROR: ${data.msg}`); setStage(`Error: ${data.msg}`, 0); role = 'baseline'; q('.fudrop').classList.remove('busy'); done(); }
  else if (data.type === 'result' && role === 'followup') {
    try { followup(data); setStage(`Follow-up done in ${((performance.now() - runStart) / 1000).toFixed(0)}s`, 1); }
    catch (e) { console.error(e); setStage(`Error: ${e.message}`, 0); }
    role = 'baseline'; q('.fudrop').classList.remove('busy'); done();
  }
  else if (data.type === 'result') {
    setStage('Reading the expansion signs', 0.99);
    setTimeout(() => {
      try { analyse(data); setStage(`Done in ${((performance.now() - runStart) / 1000).toFixed(0)}s`, 1); }
      catch (e) { console.error(e); log(`ERROR in sign analysis: ${e.message}`); setStage(`Error: ${e.message}`, 0); }
      done();
    }, 30);
  }
}

// ---------- analysis and display ----------
const viewer = new Viewer(q('.viewerhost'), OVERLAY);
function analyse(res) {
  const ta = performance.now(), g = res.grid;
  const S = analyseSigns(g, res.labels, res.raw, res.mask, { includeIVH: includeIVH() });
  const hemMask = new Uint8Array(res.labels.length);
  for (let i = 0; i < hemMask.length; i++) hemMask[i] = res.labels[i] === 2 || (includeIVH() && res.labels[i] === 3) ? 1 : 0;
  const roi = S.status === 'ok' && !S.contrast.enhanced ? deepmedxROI(g, hemMask) : null;
  log(`signs read in ${((performance.now() - ta) / 1000).toFixed(1)}s: ${Object.entries(S.signs).filter(([, s]) => s.auto).map(([k]) => k).join(', ') || 'none'}`);
  for (const k of Object.keys(override)) delete override[k];
  R = { res, g, S, roi, yale: null }; FU = null;
  show();
  runYale();
}

// ---------- Yale expansion model (worker-yale.js) ----------
function runYale() {
  const box = q('.yale');
  if (!$('yale').checked) { box.innerHTML = '<p class="small">Not run (tick <i>Yale HE model: run</i> above and load the scan again).</p>'; return; }
  if (R.S.contrast.enhanced) { box.innerHTML = '<p class="small">Not run: the model was trained on non-contrast CT.</p>'; return; }
  if (!yaleWorker) { yaleWorker = new Worker(new URL('./worker-yale.js', import.meta.url), { type: 'module' }); yaleWorker.onmessage = onYale; }
  const id = ++yaleId, g = R.g, useSahvai = $('yaleMask').value === 'sahvai';
  let mask = null; if (useSahvai) { mask = new Uint8Array(R.res.labels.length); for (let i = 0; i < mask.length; i++) mask[i] = R.res.labels[i] === 2 ? 1 : 0; }
  box.innerHTML = `<div class="scard"><p class="small">Running the Yale model (${useSahvai ? 'SAHVAI hematoma mask' : 'its own SegResNet segmentation'}, ${+$('yalePasses').value || 0} Monte-Carlo passes)…</p><div class="bar"><i class="ybar"></i></div><p class="small ystage"></p></div>`;
  yaleWorker.postMessage({ job: 'run', id, device: device(), passes: +$('yalePasses').value || 0, mask,
    grid: { nx: g.nx, ny: g.ny, nz: g.nz, sp: g.sp, hu: Float32Array.from(g.hu) } });
}
function onYale({ data }) {
  if (data.id !== yaleId || !R) return;
  const box = q('.yale');
  if (data.type === 'log') log('Yale: ' + data.msg);
  else if (data.type === 'progress') { const b = box.querySelector('.ybar'), st = box.querySelector('.ystage'); if (b) b.style.width = `${Math.round(data.frac * 100)}%`; if (st) st.textContent = data.stage; }
  else if (data.type === 'missing') box.innerHTML = `<div class="scard"><p class="small"><b>The Yale model files are not installed on this copy of the site.</b> The weights are public on GitHub but carry no licence, so they are hosted only once the authors agree. Local installs: put <code>yale_seg.onnx</code> and <code>yale_cls.onnx</code> in <code>models/yale/</code> (see <code>tools/yale_export.py</code>).</p></div>`;
  else if (data.type === 'error') { log('Yale ERROR: ' + data.msg); box.innerHTML = `<div class="scard"><p class="small">Yale model error: ${esc(data.msg)}</p></div>`; }
  // the 3D networks hold several hundred MB of working memory: free it as soon as the model has answered
  if (['result', 'error', 'missing'].includes(data.type)) { yaleWorker.terminate(); yaleWorker = null; }
  if (data.type === 'result') { R.yale = data; log(`Yale model: p(HE ≥ 6 mL) ${data.mc.p.toFixed(3)} (deterministic ${data.det.p.toFixed(3)}), ${data.segSource} ${data.segVolume.toFixed(1)} mL, ${data.seconds.toFixed(0)} s`); renderYale(); }
}
function renderYale() {
  const y = R.yale, box = q('.yale'); if (!y) return;
  const pct = v => (100 * v).toFixed(0) + '%', mc = y.mc.passes > 0;
  const ps = y.mc.logits.map(v => 1 / (1 + Math.exp(-v))).sort((a, b) => a - b), lo = ps.length ? ps[Math.floor(0.1 * (ps.length - 1))] : null, hi = ps.length ? ps[Math.ceil(0.9 * (ps.length - 1))] : null;
  box.innerHTML = `<div class="scores"><div class="scard"><h4>Predicted probability of expansion ≥ 6 mL <small>Yale 3D DenseNet121</small></h4>
      <p class="big">${pct(mc ? y.mc.p : y.det.p)}</p>
      <table><tr><td>${mc ? `Monte-Carlo mean (${y.mc.passes} passes)` : 'Deterministic'}</td><td>${(mc ? y.mc.p : y.det.p).toFixed(3)}</td></tr>
      ${mc ? `<tr><td>Deterministic (dropout off)</td><td>${y.det.p.toFixed(3)}</td></tr><tr><td>Spread of passes (10th–90th pct)</td><td>${lo.toFixed(2)}–${hi.toFixed(2)}</td></tr><tr><td>SD across passes · entropy</td><td>${y.mc.sd.toFixed(3)} · ${y.mc.entropy.toFixed(2)} bits</td></tr>` : ''}
      <tr><td>Hematoma mask (${esc(y.segSource)})</td><td>${y.segVolume.toFixed(1)} mL</td></tr></table>
      <p class="small">Trained on ATACH-2 and Yale patients (n = 634; supratentorial ICH within 4.5 h). In its test cohort, after excluding the most uncertain 5% of predictions, AUC 0.81, sensitivity 0.62 and specificity 0.81 for expansion ≥ 6 mL. No decision threshold was published; treat the probability as a research score. A wide spread or high entropy means low confidence.</p></div>
    <div class="scard"><h4>About this run</h4><p class="small">${y.device === 'webgpu' ? 'WebGPU' : 'CPU'}, ${y.seconds.toFixed(0)} s. The model sees the brain window (0–80 HU) skull-stripped and padded to 512 × 512 × 48, its hematoma mask resampled to 1 mm, the slices with blood ± 16 mm, and the CT within 40 mm of the clot. ${y.segSource === 'SAHVAI IPH' ? 'The SAHVAI IPH mask replaced the Yale SegResNet: faster, but not the published pipeline.' : 'Its SegResNet mask is shown as the white outline in the viewer.'}</p>
      <p class="small">Code and weights: <a href="https://github.com/anhtrnyaleedu/HE" target="_blank" rel="noopener">anhtrnyaleedu/HE</a> (Tran et al., <i>npj Digit Med</i> 2024).</p></div></div>`;
  if (y.segSource !== 'SAHVAI IPH') { const z = viewer.z; viewer.show(R.g, R.S.overlay, { mask: R.res.mask, range: R.S.hematoma?.slices || null, gt: y.maskGrid, gtName: 'Yale SegResNet' }); viewer.setSlice(z); }
}

// ---------- follow-up CT: conventional and revised expansion ----------
function vols(labels, sp) {
  const vox = sp[0] * sp[1] * sp[2] / 1000; let ich = 0, ivh = 0; for (const l of labels) { if (l === 2) ich++; else if (l === 3) ivh++; }
  return { ich: ich * vox, ivh: ivh * vox };
}
function followup(res) {
  const v = vols(res.labels, res.grid.sp);
  FU = { res, v, date: res.info && res.info.series_date };
  renderFU();
}
function renderFU() {
  if (!R) return;
  const b = vols(R.res.labels, R.g.sp), box = q('.futable');
  const val = (id, d) => { const el = $(id); return el && el.value !== '' ? +el.value : d; };
  const ich0 = val('fu-ich0', +b.ich.toFixed(1)), ivh0 = val('fu-ivh0', +b.ivh.toFixed(1)), ich1 = val('fu-ich1', FU ? +FU.v.ich.toFixed(1) : null), ivh1 = val('fu-ivh1', FU ? +FU.v.ivh.toFixed(1) : null);
  const E = ich1 != null ? expansion({ ich0, ich1, ivh0, ivh1 }) : null;
  const yes = (v, extra = '') => v ? `<b class="yes">Yes</b>${extra}` : '<b class="no">No</b>';
  box.innerHTML = `<table class="futab"><tr><th></th><th>Baseline</th><th>Follow-up</th><th>Change</th></tr>
    <tr><td>ICH (all IPH) mL</td><td><input id="fu-ich0" type="number" step="0.1" value="${ich0}"></td><td><input id="fu-ich1" type="number" step="0.1" value="${ich1 ?? ''}" placeholder="drop CT or type"></td><td>${E ? `${E.dICH >= 0 ? '+' : ''}${E.dICH.toFixed(1)} mL${E.rel != null ? ` (${E.rel >= 0 ? '+' : ''}${(100 * E.rel).toFixed(0)}%)` : ''}` : ''}</td></tr>
    <tr><td>IVH mL</td><td><input id="fu-ivh0" type="number" step="0.1" value="${ivh0}"></td><td><input id="fu-ivh1" type="number" step="0.1" value="${ivh1 ?? ''}"></td><td>${E ? `${E.dIVH >= 0 ? '+' : ''}${E.dIVH.toFixed(1)} mL${E.newIVH ? ' · new IVH' : ''}` : ''}</td></tr></table>
    ${E ? `<div class="scores"><div class="scard"><h4>Conventional HE <small>&gt; 6 mL or &gt; 33%</small></h4><p class="big">${yes(E.conventional)}</p></div>
      <div class="scard"><h4>Revised HE (rHE) <small>ICH ≥ 6 mL / ≥ 33%, IVH ≥ 1 mL, or new IVH</small></h4><p class="big">${yes(E.revised, E.revisedByIVHOnly ? ' <small>(by IVH growth only)</small>' : '')}</p></div>
      <div class="scard"><h4>Other absolute thresholds</h4><table><tr><td>≥ 3 mL</td><td>${E.thresholds.ge3 ? 'yes' : 'no'}</td></tr><tr><td>≥ 6 mL</td><td>${E.thresholds.ge6 ? 'yes' : 'no'}</td></tr><tr><td>≥ 9 mL</td><td>${E.thresholds.ge9 ? 'yes' : 'no'}</td></tr><tr><td>≥ 12.5 mL</td><td>${E.thresholds.ge12_5 ? 'yes' : 'no'}</td></tr></table></div></div>
      <p class="small">Volumes are SAHVAI segmentations (all IPH; IVH) unless typed. New IVH here: &lt; ${NEW_IVH_mL} mL at baseline and ≥ ${NEW_IVH_mL} mL at follow-up, because tiny AI-labelled specks are not reliable. Check both segmentations, especially after surgery or a drain, which change the volumes for other reasons.${FU ? ` Follow-up source: ${esc(FU.res.info.series_description || FU.res.info.source || '?')}.` : ''}</p>` : ''}`;
  box.querySelectorAll('input').forEach(el => el.onchange = renderFU);
}
const fud = q('.fudrop');
q('.fu-files').onchange = e => { run(e.target.files, 'follow-up CT', 'followup'); e.target.value = ''; };
q('.fu-folder').onchange = e => { run(e.target.files, 'follow-up CT', 'followup'); e.target.value = ''; };
fud.addEventListener('dragover', e => { e.preventDefault(); e.stopPropagation(); fud.classList.add('over'); });
fud.addEventListener('dragleave', () => fud.classList.remove('over'));
fud.addEventListener('drop', async e => { e.preventDefault(); e.stopPropagation(); fud.classList.remove('over'); run(await filesFromDataTransfer(e.dataTransfer), 'follow-up CT', 'followup'); });
window.HESIGNS_FOLLOWUP = (files) => run(files, 'follow-up CT', 'followup');   // (used by the automated tests)

const SIGN_ORDER = ['blend', 'blackhole', 'hypodensity', 'swirl', 'island', 'satellite', 'irregular', 'heterogeneous', 'fluid', 'spot', 'leakage', 'iodine'];
const INFO = Object.fromEntries(SIGN_INFO.map(s => [s.key, s]));
const eff = k => override[k] ?? R.S.signs[k].auto;
function ivhYes() { const v = $('ivh').value; return v === 'yes' ? true : v === 'no' ? false : R.S.volumes.IVH >= 0.5; }
const hoursVal = () => { const v = $('hours').value; return v === '' ? null : Number(v); };

function show() {
  const { S, g, res } = R, H = S.hematoma || {};
  const ncct = SIGN_ORDER.filter(k => INFO[k].modality === 'NCCT'), nPos = ncct.filter(k => S.signs[k].auto).length;
  const v = q('.verdict');
  if (S.status === 'no-ich') { v.className = 'verdict'; v.innerHTML = `<div class="vt">No intraparenchymal hematoma ≥ ${CRITERIA.minHematoma_mL} mL found</div><div class="vs">The expansion signs describe a spontaneous ICH; nothing to read here. Check the segmentation below.</div>`; }
  else if (S.contrast.enhanced) { v.className = `verdict ${S.signs.spot.auto ? 'pos' : ''}`; v.innerHTML = `<div class="vt">Contrast-enhanced scan: ${S.signs.spot.auto ? `${S.spots.length} spot-sign candidate${S.spots.length > 1 ? 's' : ''}` : 'no spot-sign candidate'}</div><div class="vs">${esc(S.signs.spot.detail)}. NCCT signs are not read on a contrast scan.</div>`; }
  else { v.className = `verdict ${nPos ? (nPos >= 3 ? 'pos' : 'maybe') : ''}`; v.innerHTML = `<div class="vt">${nPos ? `${nPos} of ${ncct.length} NCCT expansion signs found` : 'No NCCT expansion sign found'}</div><div class="vs">hematoma ${H.volume_mL.toFixed(1)} mL, largest on slice ${H.maxSlice + 1}${nPos ? ` · ${ncct.filter(k => S.signs[k].auto).map(k => `<b>${esc(INFO[k].name)}</b>`).join(', ')}` : ''} · candidates for a reader, not diagnoses</div>`; }
  const card = (k, val, unit, sub, c, cls = '') => `<div class="vol ${cls}" style="--c:${c}"><div class="k">${k}</div><div class="v">${val}${unit ? ` <small>${unit}</small>` : ''}</div><div class="s">${sub}</div></div>`;
  q('.summary').innerHTML = card('Hematoma', S.volumes.main.toFixed(1), 'mL', 'largest IPH component', 'rgb(255,45,120)') +
    card('All IPH', S.volumes.IPH.toFixed(1), 'mL', `${S.volumes.iphComponents} component${S.volumes.iphComponents === 1 ? '' : 's'}`, 'rgb(255,176,0)') +
    card('IVH', S.volumes.IVH.toFixed(1), 'mL', S.volumes.IVH >= 0.5 ? 'extension present' : 'none', 'rgb(30,200,255)') +
    card('Density', H.hemRef ? H.hemRef.toFixed(0) : '–', 'HU', H.brainRef ? `clot (75th pct) · brain ${H.brainRef.toFixed(0)} HU` : '', 'var(--mayo-blue)') +
    `<div class="vol total"><div class="k">BAT score</div><div class="v bat">–</div><div class="s bats"></div></div>`;
  const w = [...(res.warnings || []), ...S.warnings];
  if (S.contrast.enhanced) w.push(`Contrast detected (${S.contrast.vessel_mL.toFixed(1)} mL of 130–600 HU in the brain): spot-sign candidates only. They are experimental and need checking on the thin CTA source images (spot window W200/L110).`);
  q('.warnbox').hidden = !w.length; q('.warnbox').innerHTML = w.map(esc).join('<br>');

  viewer.show(g, S.overlay, { mask: res.mask, range: H.slices || null });
  if (H.maxSlice != null) viewer.setSlice(H.maxSlice);
  viewer.dl.innerHTML = `<button class="btn small" data-k="json">Report (.json)</button><button class="btn small ghost" data-k="lab">Sign map (.nii.gz)</button>
    <button class="btn small ghost" data-k="seg">Segmentation (.nii.gz)</button><button class="btn small ghost" data-k="ct">CT analysed (.nii.gz)</button><button class="btn small ghost" data-k="png">This view (.png)</button>`;
  viewer.dl.querySelector('[data-k=json]').onclick = () => save(new Blob([JSON.stringify(report(), null, 2)], { type: 'application/json' }), 'ich-expansion-signs.json');
  viewer.dl.querySelector('[data-k=lab]').onclick = async () => save(await niftiGz(S.overlay, g, { datatype: 2, description: 'HE signs: 1hem 2blood 3IVH 4hypo 5blackhole 6blend 7fluid 8sat/island 9spot' }), 'ich-expansion-sign-map.nii.gz');
  viewer.dl.querySelector('[data-k=seg]').onclick = async () => save(await niftiGz(res.labels, g, { datatype: 2, description: 'SAHVAI labels: 1=SAH 2=IPH 3=IVH' }), 'sahvai-labels.nii.gz');
  viewer.dl.querySelector('[data-k=ct]').onclick = async () => save(await niftiGz(toInt16(g.hu), g, { datatype: 4, description: 'working grid (HU)' }), 'ich-ct.nii.gz');
  viewer.dl.querySelector('[data-k=png]').onclick = () => viewer.png(`ich-signs-slice-${viewer.z + 1}.png`);

  renderSigns(); renderScores(); renderROI(); renderFU();
  const i = res.info || {};
  q('.details').innerHTML = [
    i.series_description !== undefined ? `Series <b>${esc(i.series_description || '?')}</b> (${esc(i.series_selection)})` : `Source <b>${esc(i.source || '?')}</b>`,
    `native ${i.native_size.join('×')} at ${i.native_spacing_mm.join(' × ')} mm → grid ${i.grid_size.join('×')} at ${i.grid_spacing_mm.join(' × ')} mm`,
    `SAHVAI: SAH ${res.volumes.SAH.toFixed(1)}, IPH ${res.volumes.IPH.toFixed(1)}, IVH ${res.volumes.IVH.toFixed(1)} mL`,
    `rotation ${res.angle}°`, `${res.device === 'webgpu' ? 'WebGPU' : 'CPU'}, ${res.seconds.toFixed(0)} s`, '<b>research and education use only</b>',
  ].join(' · ');
  q('.results').hidden = false;
  // research export (lib/research.js): volumes, signs (reader overrides included) and scores, read when the button is pressed
  q('.results').querySelectorAll('.rx').forEach(e => e.remove());
  mountResearch(q('.results'), () => {
    if (!R) return null;
    const rp = report(), sc = rp.scores || {}, used = SIGN_ORDER.filter(k => eff(k) === true);
    return { tool: 'hesigns', info: R.res.info, compute: R.res.device, seconds: R.res.seconds, warnings: [...(R.res.warnings || []), ...(R.S.warnings || [])], inputFormat: /^demo case/.test(lastLabel) ? 'demo' : undefined,
      measurements: { vol_sah_ml: R.res.volumes.SAH, vol_iph_ml: R.res.volumes.IPH, vol_ivh_ml: R.res.volumes.IVH, vol_total_blood_ml: R.res.volumes.total ?? (R.res.volumes.SAH + R.res.volumes.IPH + R.res.volumes.IVH),
        he_hematoma_ml: R.S.volumes ? R.S.volumes.main : null, he_signs_present: used.join(';') || null, he_bat_score: sc.BAT ? sc.BAT.score : null, he_brain_score: sc.BRAIN ? sc.BRAIN.score : null, he_yale_p: rp.yale_model ? rp.yale_model.p : null },
      provenance: { hematoma_definition: rp.hematoma_definition, status: rp.status, signs: Object.fromEntries(Object.entries(rp.signs).map(([k, s]) => [k, { auto: s.auto, reader: s.reader, used: s.used }])), scores: rp.scores, yale_model: rp.yale_model },
      labelMap: () => niftiGz(R.res.labels, R.g, { datatype: 2, description: 'SAHVAI labels: 1=SAH 2=IPH 3=IVH' }) };
  }, { label: 'ICH expansion signs' });
  q('.results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function status(k) {
  const s = R.S.signs[k], o = override[k];
  if (o != null) return o ? ['pos', 'Present (reader)'] : ['neg', 'Absent (reader)'];
  if (s.auto === true) return ['pos', 'Detected'];
  if (s.auto === false) return ['neg', 'Not detected'];
  return ['na', INFO[k].modality === 'NCCT' ? 'Not assessed' : `Needs ${INFO[k].modality}`];
}
function renderSigns() {
  const box = q('.signs');
  box.innerHTML = SIGN_ORDER.map(k => {
    const I = INFO[k], s = R.S.signs[k], [cls, txt] = status(k), ev = s.evidence.filter(e => e.slice != null);
    const sl = [...new Set(ev.map(e => e.slice))].slice(0, 8);
    return `<div class="sign ${cls}" data-k="${k}">
      <div class="sh">${schematic(k, 44)}<div><b>${esc(I.name)}</b><span class="mod">${esc(I.modality)}</span><div class="st">${txt}</div></div></div>
      <p class="det">${esc(s.detail)}</p>
      ${sl.length ? `<p class="jumps">${sl.map(z => `<button type="button" class="jz" data-z="${z}">slice ${z + 1}</button>`).join('')}</p>` : ''}
      <label class="rd">Reader <select data-k="${k}"><option value="">${s.auto == null ? '—' : 'agree'}</option><option value="1"${override[k] === true ? ' selected' : ''}>present</option><option value="0"${override[k] === false ? ' selected' : ''}>absent</option></select></label>
    </div>`;
  }).join('');
  box.querySelectorAll('.jz').forEach(b => b.onclick = e => { e.stopPropagation(); viewer.setSlice(+b.dataset.z); q('.viewerhost').scrollIntoView({ behavior: 'smooth', block: 'center' }); });
  box.querySelectorAll('.sign').forEach(c => c.onclick = e => { if (e.target.closest('select,button')) return; const z = R.S.signs[c.dataset.k].evidence.find(x => x.slice != null); if (z) { viewer.setSlice(z.slice); q('.viewerhost').scrollIntoView({ behavior: 'smooth', block: 'center' }); } });
  box.querySelectorAll('select').forEach(sel => sel.onchange = () => { const k = sel.dataset.k; if (sel.value === '') delete override[k]; else override[k] = sel.value === '1'; renderSigns(); renderScores(); });
}

function renderScores() {
  if (!R) return;
  const { S } = R, h = hoursVal(), vol = S.volumes.main, ok = S.status === 'ok';
  const bat = ok ? batScore({ blend: !!eff('blend'), hypodensity: !!eff('hypodensity'), hours: h }) : null;
  const imgBat = ok ? (eff('blend') ? 1 : 0) + (eff('hypodensity') ? 2 : 0) : null;
  const br = ok ? brainScore({ volume: vol, recurrent: $('recurrent').checked, warfarin: $('warfarin').checked, ivh: ivhYes(), hours: h }) : null;
  const uhg = ok && h > 0 ? vol / h : null;
  const batEl = q('.bat'), batS = q('.bats');
  if (batEl) { batEl.innerHTML = bat ? `${bat.score} <small>/ 5</small>` : imgBat != null ? `${imgBat} <small>+ time</small>` : '–'; batS.textContent = bat ? (bat.high ? '≥ 3: high risk of expansion' : '< 3: lower risk') : ok ? 'enter onset-to-CT time for the full score' : ''; }
  q('.scores').innerHTML = !ok ? '<p class="small">No hematoma to score.</p>' : `
    <div class="scard"><h4>BAT score <small>Morotti et al., Stroke 2018</small></h4>
      <table><tr><td>Blend sign</td><td>${eff('blend') ? 1 : 0}</td></tr><tr><td>Any intrahematoma hypodensity</td><td>${eff('hypodensity') ? 2 : 0}</td></tr>
      <tr><td>Onset to NCCT &lt; 2.5 h</td><td>${h == null ? '?' : h < 2.5 ? 2 : 0}</td></tr><tr class="tot"><td>Total (0–5)</td><td>${bat ? bat.score : '?'}</td></tr></table>
      <p class="small">≥ 3 predicts expansion (&gt; 6 mL or &gt; 33%) with high specificity (0.89–0.93 in published cohorts) and low sensitivity (≈ 0.4–0.5).</p></div>
    <div class="scard"><h4>BRAIN score <small>Wang et al., Stroke 2015 (INTERACT)</small></h4>
      <table><tr><td>Baseline volume ${vol.toFixed(1)} mL</td><td>${vol <= 10 ? 0 : vol <= 20 ? 5 : 7}</td></tr><tr><td>Recurrent ICH</td><td>${$('recurrent').checked ? 4 : 0}</td></tr>
      <tr><td>Warfarin at onset</td><td>${$('warfarin').checked ? 6 : 0}</td></tr><tr><td>IVH extension${$('ivh').value === 'auto' ? ` (scan: ${S.volumes.IVH.toFixed(1)} mL)` : ''}</td><td>${ivhYes() ? 2 : 0}</td></tr>
      <tr><td>Hours onset to CT</td><td>${br ? br.parts.time : '?'}</td></tr><tr class="tot"><td>Total (0–24)</td><td>${br ? br.score : '?'}</td></tr></table>
      <p class="small">Predicted probability of ≥ 6 mL growth at 24 h rose from 3.4% at 0 points to 85.8% at 24 points (C-statistic 0.73). The volume here is the AI segmentation, not ABC/2.</p></div>
    <div class="scard"><h4>Ultra-early hematoma growth <small>volume ÷ onset-to-CT time</small></h4>
      <p class="big">${uhg != null ? `${uhg.toFixed(1)} <small>mL/h</small>` : '<small>enter onset-to-CT time</small>'}</p>
      <p class="small">A time-adjusted marker of early bleeding rate (reviewed by Liu et al. 2026); it assumes zero volume at onset and depends on an accurate onset time.</p></div>
    ${S.contrast.enhanced ? `<div class="scard"><h4>Spot sign score <small>Delgado Almandoz et al., Stroke 2009</small></h4><p class="big">${S.signs.spot.score ?? 0} <small>/ 4</small></p><p class="small">Number of spots (1–2: 1, ≥ 3: 2), maximum axial dimension ≥ 5 mm (1), maximum attenuation ≥ 180 HU (1). Approximate: measured on 5 mm slabs.</p></div>` : ''}`;
}
['hours', 'warfarin', 'recurrent', 'ivh'].forEach(id => $(id).addEventListener('input', renderScores));

function grayCanvas(img, w, h, maxH) {
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d'), id = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) { id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = img[i]; id.data[4 * i + 3] = 255; }
  ctx.putImageData(id, 0, 0);
  if (maxH) { cv.style.height = `${maxH}px`; cv.style.width = 'auto'; }
  return cv;
}
function renderROI() {
  const box = q('.dmx'), roi = R.roi;
  box.innerHTML = `<p class="small">The DeepMedX pipeline (Yu et al., <i>npj Digit Med</i> 2026; 2,020 patients, 5 centres) segments the hematoma with U-Mamba (Dice 0.98), crops the slice of largest hematoma area and its two neighbours, and classifies each crop with a Vision Transformer trained on real plus Diffusion-UKAN synthetic expansion images (HE:NHE 1:2); the patient-level probability is the mean of the three. External AUCs 0.793 and 0.781 for expansion &gt; 6 mL or &gt; 33%.
    Here the SAHVAI segmentation stands in for U-Mamba and the crops are made exactly as in their <code>crop2DROI.ipynb</code> (3.5× upsampling, bounding box of the largest component, window 45/90 HU) and their ViT transform (resize 256, centre crop 224).
    <b>The trained ViT weights are not public</b>, so no DeepMedX expansion probability is computed: these are the inputs it would receive.</p>
    <div class="rois"></div>`;
  if (!roi) { box.querySelector('.rois').innerHTML = `<p class="small">${R.S.contrast.enhanced ? 'Not made for a contrast-enhanced scan (the pipeline uses NCCT).' : 'No hematoma to crop.'}</p>`; return; }
  const host = box.querySelector('.rois');
  for (const p of roi.planes) {
    const fig = document.createElement('figure');
    fig.append(grayCanvas(p.img, p.w, p.h, 150), grayCanvas(p.vit, 224, 224, 150));
    const cap = document.createElement('figcaption');
    cap.innerHTML = `<b>${esc(p.name)}</b> · grid slice ${(p.slice + 1).toFixed(2)} · crop ${p.w}×${p.h} px → ViT input 224×224 <button type="button" class="btn small ghost">PNG</button>`;
    cap.querySelector('button').onclick = () => fig.querySelector('canvas').toBlob(b => save(b, `deepmedx-roi-${p.name.split(' ')[0].replace('−', 'minus').replace('+', 'plus')}.png`));
    fig.append(cap); host.append(fig);
  }
}

function report() {
  const { S, res, roi } = R, h = hoursVal();
  return {
    disclaimer: 'Research and education use only. Not a medical device. Signs are rule-based candidates on an AI segmentation; a reader must confirm them.',
    tool: 'ICH expansion signs, SAHVAI Web', created: new Date().toISOString(), input: lastLabel, ...publicInfo(res.info),
    segmentation: { model: 'SAHVAI (BrainHemoAI Hybrid 2D/3D UNet)', volumes_mL: { SAH: r1(res.volumes.SAH), IPH: r1(res.volumes.IPH), IVH: r1(res.volumes.IVH) } },
    hematoma_definition: includeIVH() ? 'largest 26-connected IPH+IVH component' : 'largest 26-connected IPH component',
    status: S.status, contrast: S.contrast, hematoma: S.hematoma && { ...S.hematoma, box: undefined, maxSlice: S.hematoma.maxSlice + 1 },
    signs: Object.fromEntries(SIGN_ORDER.map(k => [k, { name: INFO[k].name, modality: INFO[k].modality, auto: S.signs[k].auto, reader: override[k] ?? null, used: eff(k) ?? null, detail: S.signs[k].detail, slices: [...new Set(S.signs[k].evidence.map(e => e.slice).filter(z => z != null))].map(z => z + 1), definition: INFO[k].def, reference: INFO[k].ref }])),
    clinical: { onset_to_ct_h: h, warfarin: $('warfarin').checked, recurrent_ich: $('recurrent').checked, ivh: ivhYes() },
    scores: S.status === 'ok' ? { BAT: batScore({ blend: !!eff('blend'), hypodensity: !!eff('hypodensity'), hours: h }), BRAIN: brainScore({ volume: S.volumes.main, recurrent: $('recurrent').checked, warfarin: $('warfarin').checked, ivh: ivhYes(), hours: h }), uHG_mL_per_h: h > 0 ? r1(S.volumes.main / h) : null, spot_sign_score: S.signs.spot.score ?? null } : null,
    yale_model: R.yale ? { model: 'Tran et al., npj Digit Med 2024 (github.com/anhtrnyaleedu/HE)', target: 'HE >= 6 mL', p: R.yale.mc.passes ? R.yale.mc.p : R.yale.det.p, p_deterministic: R.yale.det.p, mc_passes: R.yale.mc.passes, mc_sd: R.yale.mc.sd, entropy_bits: R.yale.mc.entropy, mask: R.yale.segSource, mask_mL: R.yale.segVolume, device: R.yale.device, seconds: R.yale.seconds } : null,
    follow_up: (() => { const g = id => { const el = $(id); return el && el.value !== '' ? +el.value : null; }; const E = g('fu-ich1') != null ? expansion({ ich0: g('fu-ich0'), ich1: g('fu-ich1'), ivh0: g('fu-ivh0') || 0, ivh1: g('fu-ivh1') || 0 }) : null;
      return E ? { ich_mL: [g('fu-ich0'), g('fu-ich1')], ivh_mL: [g('fu-ivh0'), g('fu-ivh1')], ...E, definitions: { conventional: 'ICH growth > 6 mL or > 33% (Dowlatshahi 2011)', revised: 'ICH growth >= 6 mL or >= 33%, IVH growth >= 1 mL, or new IVH (Yogendrakumar, Stroke 2020)' } } : null; })(),
    deepmedx_roi: roi ? { scale: roi.scale, window_HU: roi.window, planes: roi.planes.map(p => ({ name: p.name, grid_slice: +(p.slice + 1).toFixed(2), crop_px: [p.w, p.h] })), classifier: 'ViT weights not public; not run' } : null,
    regions: S.regions, satellites: S.satellites, islands: (S.islands || []).map(s => ({ ...s, k: undefined, slice: s.k + 1 })), spots: S.spots, criteria: S.criteria, warnings: S.warnings,
  };
}

// ---------- schematics (drawn for this page; anterior at the top) ----------
function schematic(k, size = 90) {
  const clot = '#e8e8e8', dim = '#9a9a9a', bg = '#1d1f24', brain = '#5c5f66';
  const base = `<rect width="100" height="100" rx="12" fill="${bg}"/><ellipse cx="50" cy="52" rx="44" ry="44" fill="${brain}"/>`;
  const e = (fill, extra = '') => `<ellipse cx="50" cy="52" rx="24" ry="18" fill="${fill}" ${extra}/>`;
  const body = {
    spot: `${e('#b8b8b8')}<path d="M8 40 Q30 30 30 52 M92 64 Q72 70 74 52" stroke="#fff" stroke-width="2" fill="none"/><circle cx="54" cy="50" r="3.6" fill="#fff"/><circle cx="54" cy="50" r="7" fill="none" stroke="#ff5252" stroke-width="1.4"/>`,
    leakage: `${e('#b8b8b8')}<circle cx="52" cy="50" r="9" fill="#fff" opacity=".85"/><circle cx="52" cy="50" r="2.5" fill="#fff"/><circle cx="52" cy="50" r="11" fill="none" stroke="#ffd54f" stroke-width="1.4" stroke-dasharray="3 2"/>`,
    iodine: `${e('#9a9a9a')}<circle cx="56" cy="50" r="8" fill="#ff9800" opacity=".85"/><circle cx="44" cy="56" r="4" fill="#ffc107" opacity=".8"/>`,
    blend: `<clipPath id="cb${size}"><ellipse cx="50" cy="52" rx="24" ry="18"/></clipPath><g clip-path="url(#cb${size})"><rect x="20" y="30" width="60" height="44" fill="${clot}"/><path d="M58 30 Q49 52 60 74 L80 74 L80 30Z" fill="${dim}"/></g>`,
    blackhole: `${e(clot)}<ellipse cx="54" cy="50" rx="8" ry="6" fill="#4a4c52"/>`,
    hypodensity: `${e(clot)}<path d="M40 48 q6 -6 12 0 q4 5 -2 9 q-7 3 -10 -9z" fill="${dim}"/>`,
    swirl: `${e(clot)}<path d="M36 50 C44 40 60 42 62 52 C63 60 52 62 48 56 C45 51 52 48 55 52" stroke="${dim}" stroke-width="5" fill="none" stroke-linecap="round"/>`,
    island: `${e(clot)}<circle cx="50" cy="26" r="4" fill="${clot}"/><circle cx="20" cy="44" r="3.5" fill="${clot}"/><circle cx="80" cy="60" r="4" fill="${clot}"/><circle cx="70" cy="37" r="4.5" fill="${clot}"/>`,
    satellite: `${e(clot)}<circle cx="82" cy="40" r="4" fill="${clot}"/><path d="M75 44 L79 42" stroke="#ffd54f" stroke-width="1" stroke-dasharray="1.5 1.5"/>`,
    irregular: `<path d="M30 52 C24 40 36 30 46 36 C48 26 60 28 60 36 C70 32 78 42 72 50 C80 58 70 70 60 66 C56 76 42 74 40 66 C30 70 22 60 30 52Z" fill="${clot}"/>`,
    heterogeneous: `${e(clot)}<circle cx="42" cy="48" r="3.5" fill="${dim}"/><circle cx="58" cy="46" r="3" fill="${dim}"/><circle cx="52" cy="60" r="3.8" fill="${dim}"/><circle cx="64" cy="57" r="2.6" fill="${dim}"/>`,
    fluid: `<clipPath id="cf${size}"><ellipse cx="50" cy="52" rx="24" ry="18"/></clipPath><g clip-path="url(#cf${size})"><rect x="20" y="30" width="60" height="22" fill="${dim}"/><rect x="20" y="52" width="60" height="22" fill="${clot}"/></g><line x1="26" y1="52" x2="74" y2="52" stroke="#ffd54f" stroke-width=".8"/>`,
  }[k] || '';
  return `<svg class="schem" viewBox="0 0 100 100" width="${size}" height="${size}" aria-hidden="true">${base}${body}</svg>`;
}
q('.refgrid').innerHTML = SIGN_INFO.map(s => `<div class="refcard">${schematic(s.key, 96)}<div><b>${esc(s.name)}</b> <span class="mod">${esc(s.modality)}</span> <span class="grp">${esc(s.group)}</span>
  <p>${esc(s.def)}</p><p class="small">${esc(s.ref)}</p></div></div>`).join('');

// ---------- input ----------
q('.f-files').onchange = e => { run(e.target.files); e.target.value = ''; };
q('.f-folder').onchange = e => { run(e.target.files); e.target.value = ''; };
const drop = q('.drop');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', async e => { e.preventDefault(); drop.classList.remove('over'); run(await filesFromDataTransfer(e.dataTransfer)); });
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());

if (window.parent !== window) {
  document.body.classList.add('embedded');
  const rep = () => window.parent.postMessage({ type: 'sahvai-web-height', height: document.documentElement.scrollHeight }, location.origin);
  new ResizeObserver(rep).observe(document.body); rep();
}
receivedHandoff().then(async h => {
  if (!h) return;
  q('.drop').before(handoffBanner(h, 'ICH expansion signs'));
  log(`hand-off from HEADS-UP: ${h.label || ''}`);
  try { run(await handoffFiles(h), `from HEADS-UP: ${h.label || 'scan'}`); } catch (e) { setStage(`Could not open the scan from HEADS-UP: ${e.message}`, 0); }
});
window.HESIGNS = () => R;   // (used by the automated tests)
