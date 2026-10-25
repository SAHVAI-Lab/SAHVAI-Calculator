import { filesFromDataTransfer } from './lib/source.js';
import { niftiGz, toInt16 } from './lib/export.js';

const $ = id => document.getElementById(id);
const logEl = $('log'), t0 = performance.now();
let worker = null, busy = false, runStart = 0, cur = { name: '', frac: 0 }, R = null, z = 0;
const COLORS = { 1: [255, 176, 0], 2: [255, 45, 120], 3: [30, 200, 255] };
const NAMES = { 1: 'SAH', 2: 'IPH', 3: 'IVH' };

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
const angle = () => document.querySelector('input[name=align]:checked').value === 'manual' ? (Number($('angle').value) || 0) : null;
document.querySelectorAll('input[name=align]').forEach(el => el.addEventListener('change', () => { $('angle').disabled = angle() === null; }));

function startWorker() {
  if (worker) return;
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = onMessage;
  worker.onerror = e => { log(`worker error: ${e.message}`); setStage(`Error: ${e.message}`, 0); done(); };
}
startWorker();
$('status').hidden = false; setStage('Loading the model', 0);
worker.postMessage({ preload: true, device: device() });
document.querySelectorAll('input[name=device]').forEach(el => el.addEventListener('change', () => { if (!busy) worker.postMessage({ preload: true, device: device() }); }));

function begin(msg) {
  busy = true; runStart = performance.now();
  $('drop').classList.add('busy'); $('status').hidden = false; $('results').hidden = true;
  log('---- new scan'); setStage('Starting', 0); if (msg) log(msg);
  startWorker();
}
function run(files) {
  files = [...files];
  if (!files.length || busy) return;
  begin(`${files.length} file(s): ${files.slice(0, 3).map(f => f.webkitRelativePath || f.name).join(', ')}${files.length > 3 ? ', …' : ''}`);
  worker.postMessage({ files, device: device(), angle: angle() });
}
$('demo').onclick = () => { if (busy) return; begin('demo case: BrainHemoAI head.nrrd (aneurysmal SAH)'); worker.postMessage({ demo: 'demo/brainhemoai-demo-head.nrrd', device: device(), angle: angle() }); };
function done() { busy = false; $('drop').classList.remove('busy'); }

function setStage(name, frac) { cur = { name, frac }; $('stage').textContent = name; $('bar').style.width = `${Math.round(frac * 100)}%`; showPct(); }
function showPct() {
  const pct = cur.frac > 0 ? `${Math.round(cur.frac * 100)}%` : '';
  $('pct').textContent = busy ? [pct, `${((performance.now() - runStart) / 1000).toFixed(0)}s`].filter(Boolean).join(' · ') : pct;
}
setInterval(() => { if (busy) showPct(); }, 1000);

function onMessage({ data }) {
  if (data.type === 'log') log(data.msg);
  else if (data.type === 'ready') { if (!busy) setStage('Model ready. Drop a scan or try the demo case.', 1); }
  else if (data.type === 'progress') setStage(data.stage, data.frac);
  else if (data.type === 'error') { log(`ERROR: ${data.msg}`); setStage(`Error: ${data.msg}`, 0); done(); }
  else if (data.type === 'result') { setStage(`Done in ${data.seconds.toFixed(0)}s`, 1); show(data); done(); }
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const mL = v => v < 10 ? v.toFixed(2) : v.toFixed(1);

function show(r) {
  const g = r.grid, P = g.nx * g.ny, vox = g.sp[0] * g.sp[1] * g.sp[2] / 1000;
  // per-slice volumes and the brain outline
  const per = Array.from({ length: g.nz }, () => [0, 0, 0, 0]);
  for (let k = 0; k < g.nz; k++) for (let p = 0; p < P; p++) per[k][r.labels[k * P + p]]++;
  const edge = new Uint8Array(P * g.nz);
  for (let k = 0; k < g.nz; k++) for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) {
    const p = k * P + y * g.nx + x; if (!r.mask[p]) continue;
    if (x === 0 || y === 0 || x === g.nx - 1 || y === g.ny - 1 || !r.mask[p - 1] || !r.mask[p + 1] || !r.mask[p - g.nx] || !r.mask[p + g.nx]) edge[p] = 1;
  }
  R = { ...r, per, edge, vox };
  const v = r.volumes;
  const card = (k, val, sub, c, cls = '') => `<div class="vol ${cls}" style="--c:${c}"><div class="k">${k}</div><div class="v">${mL(val)} <small>mL</small></div><div class="s">${sub}</div></div>`;
  $('vols').innerHTML = card('SAH', v.SAH, 'subarachnoid', 'var(--sah)') + card('IPH', v.IPH, 'intraparenchymal', 'var(--iph)') +
    card('IVH', v.IVH, 'intraventricular', 'var(--ivh)') + card('Total blood', v.total, 'SAH + IPH + IVH', 'var(--mayo-blue)', 'total');
  const w = r.warnings || [];
  $('warn').hidden = !w.length; $('warn').innerHTML = w.map(esc).join('<br>');
  const i = r.info || {};
  const mm = ((r.slices[1] - r.slices[0] + 1) * g.sp[2]).toFixed(0);
  $('details').innerHTML = [
    i.series_description !== undefined ? `Series <b>${esc(i.series_description || '?')}</b>${i.body_part ? ` [${esc(i.body_part)}]` : ''} (${esc(i.series_selection)})` : `Source <b>${esc(i.source || '?')}</b>`,
    `native ${i.native_size.join('×')} at ${i.native_spacing_mm.join(' × ')} mm → working grid ${i.grid_size.join('×')} at ${i.grid_spacing_mm.join(' × ')} mm`,
    `slices ${r.slices[0] + 1}–${r.slices[1] + 1} of ${g.nz} analysed (${mm} mm) in ${r.windows.length} window${r.windows.length > 1 ? 's' : ''} of 18`,
    `rotation ${r.angle}°`, `${r.device === 'webgpu' ? 'WebGPU' : 'CPU'}, ${r.seconds.toFixed(0)} s`,
    '<b>research use only, not for clinical decisions</b>',
  ].join(' · ');
  $('results').hidden = false;
  const cv = $('view'); cv.width = g.nx; cv.height = g.ny; cv.style.aspectRatio = `${g.nx * g.sp[0]} / ${g.ny * g.sp[1]}`;
  $('slice').max = g.nz - 1;
  let best = Math.floor(g.nz / 2), bv = 0;
  per.forEach((c, k) => { const b = c[1] + c[2] + c[3]; if (b > bv) { bv = b; best = k; } });
  setSlice(best);
  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setSlice(k) {
  if (!R) return;
  z = Math.max(0, Math.min(R.grid.nz - 1, k)); $('slice').value = z; render(); chart();
}

function render() {
  if (!R) return;
  const g = R.grid, P = g.nx * g.ny, o = z * P, cv = $('view'), ctx = cv.getContext('2d');
  const [c, w] = document.querySelector('input[name=win]:checked').value.split(',').map(Number), lo = c - w / 2;
  const ov = $('overlay').checked, a = $('opacity').value / 100, outl = $('outline').checked;
  const img = ctx.createImageData(g.nx, g.ny), d = img.data;
  for (let p = 0; p < P; p++) {
    let v = (g.hu[o + p] - lo) / w * 255; v = v < 0 ? 0 : v > 255 ? 255 : v;
    let rr = v, gg = v, bb = v;
    const l = R.labels[o + p];
    if (ov && l) { const col = COLORS[l]; rr = rr * (1 - a) + col[0] * a; gg = gg * (1 - a) + col[1] * a; bb = bb * (1 - a) + col[2] * a; }
    if (outl && R.edge[o + p]) { rr = 60; gg = 230; bb = 90; }
    const q = p * 4; d[q] = rr; d[q + 1] = gg; d[q + 2] = bb; d[q + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const s = R.per[z], inR = z >= R.slices[0] && z <= R.slices[1];
  $('sliceinfo').textContent = [`slice ${z + 1}/${g.nz}${inR ? '' : ' (not analysed)'}`,
    ...[1, 2, 3].filter(k => s[k]).map(k => `${NAMES[k]} ${(s[k] * R.vox).toFixed(2)} mL`)].join(' · ');
}

function chart() {
  if (!R) return;
  const cv = $('chart'), dpr = window.devicePixelRatio || 1, Wc = cv.clientWidth, Hc = cv.clientHeight;
  cv.width = Wc * dpr; cv.height = Hc * dpr;
  const ctx = cv.getContext('2d'); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, Wc, Hc);
  const n = R.grid.nz, bw = Wc / n, max = Math.max(1, ...R.per.map(c => c[1] + c[2] + c[3]));
  ctx.fillStyle = '#eef2f9'; ctx.fillRect(R.slices[0] * bw, 0, (R.slices[1] - R.slices[0] + 1) * bw, Hc);
  R.per.forEach((c, k) => {
    let y = Hc - 14;
    for (const t of [1, 2, 3]) {
      const h = c[t] / max * (Hc - 22); if (!h) continue;
      ctx.fillStyle = `rgb(${COLORS[t].join(',')})`; ctx.fillRect(k * bw + 1, y - h, Math.max(1, bw - 2), h); y -= h;
    }
  });
  ctx.strokeStyle = '#003da5'; ctx.lineWidth = 2; ctx.strokeRect(z * bw + 0.5, 1, bw - 1, Hc - 2);
  ctx.fillStyle = '#5f6673'; ctx.font = '10px system-ui'; ctx.fillText('inferior', 3, Hc - 3); ctx.textAlign = 'right'; ctx.fillText('superior', Wc - 3, Hc - 3);
}
$('chart').addEventListener('click', e => { if (!R) return; const r = e.target.getBoundingClientRect(); setSlice(Math.floor((e.clientX - r.left) / r.width * R.grid.nz)); });
$('slice').addEventListener('input', e => setSlice(+e.target.value));
document.querySelectorAll('input[name=win], #overlay, #opacity, #outline').forEach(el => el.addEventListener('input', render));
$('canvaswrap').addEventListener('wheel', e => { if (!R) return; e.preventDefault(); setSlice(z + (e.deltaY > 0 ? -1 : 1)); }, { passive: false });
window.addEventListener('keydown', e => { if (!R || e.target.tagName === 'INPUT' && e.target.type !== 'range') return; if (e.key === 'ArrowUp') { setSlice(z + 1); e.preventDefault(); } if (e.key === 'ArrowDown') { setSlice(z - 1); e.preventDefault(); } });
window.addEventListener('resize', chart);

function save(blob, name) { const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: name }); a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
$('dlLabels').onclick = async () => R && save(await niftiGz(R.labels, R.grid, { datatype: 2, description: 'SAHVAI labels: 1=SAH 2=IPH 3=IVH' }), 'sahvai-labels.nii.gz');
$('dlCT').onclick = async () => R && save(await niftiGz(toInt16(R.grid.hu), R.grid, { datatype: 4, description: 'SAHVAI working grid (HU)' }), 'sahvai-ct.nii.gz');
$('dlPNG').onclick = () => R && $('view').toBlob(b => save(b, `sahvai-slice-${z + 1}.png`));
$('dlJSON').onclick = () => {
  if (!R) return;
  const r2 = x => +x.toFixed(3);
  const report = {
    disclaimer: 'Research use only. Not a medical device and not for diagnosis or treatment decisions.',
    model: 'Hybrid 2D/3D UNet (Hu et al., NeuroImage 2023;279:120321), BrainHemoAI weights, run with ONNX Runtime Web',
    created: new Date().toISOString(), ...R.info,
    volumes_mL: { SAH: r2(R.volumes.SAH), IPH: r2(R.volumes.IPH), IVH: r2(R.volumes.IVH), total: r2(R.volumes.total) },
    voxel_mL: R.vox, rotation_deg: R.angle, slices_analysed: [R.slices[0] + 1, R.slices[1] + 1], windows: R.windows.map(k => k + 1),
    per_slice_mL: R.per.map((c, k) => ({ slice: k + 1, SAH: r2(c[1] * R.vox), IPH: r2(c[2] * R.vox), IVH: r2(c[3] * R.vox) })),
    warnings: R.warnings, device: R.device, seconds: r2(R.seconds),
  };
  save(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }), 'sahvai-report.json');
};

$('files').onchange = e => { run(e.target.files); e.target.value = ''; };
$('folder').onchange = e => { run(e.target.files); e.target.value = ''; };
const drop = $('drop');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', async e => { e.preventDefault(); drop.classList.remove('over'); run(await filesFromDataTransfer(e.dataTransfer)); });
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());

// When embedded as a tab of the SAHVAI Calculator, tell the page how tall this content is.
if (window.parent !== window) {
  document.body.classList.add('embedded');
  const report = () => window.parent.postMessage({ type: 'sahvai-web-height', height: document.documentElement.scrollHeight }, location.origin);
  new ResizeObserver(report).observe(document.body); report();
}
