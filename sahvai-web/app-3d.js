// SAHVAI 3D-4D: 3D hemorrhage map and blood volumes over serial scans.
// Segmentation is the BrainHemoAI pipeline of SAHVAI Web (worker.js, multi-scan mode); this page adds
// surfaces (lib/mesh.js + three.js), orthogonal reformats and the volume-over-time chart.
import { filesFromDataTransfer } from './lib/source.js';
import * as THREE from './vendor/three/three-sahvai.min.js';
import { buildSurfaces } from './lib/mesh.js';

const $ = id => document.getElementById(id);
const logEl = $('log'), t0 = performance.now();
const COLORS = { 1: [255, 176, 0], 2: [255, 45, 120], 3: [30, 200, 255] };
const NAMES = { 1: 'SAH', 2: 'IPH', 3: 'IVH' };
const SERIES = [
  { k: 'SAH', label: 'SAH (SAHV)', color: '#e09a00' },
  { k: 'IPH', label: 'IPH', color: '#ff2d78' },
  { k: 'IVH', label: 'IVH', color: '#12a9dc' },
  { k: 'total', label: 'Total blood', color: '#003da5' },
];
const DAY = 86400000;
let worker = null, busy = false, runStart = 0, stageNow = { name: '', frac: 0 };
let scans = [], cur = -1, batchBase = 0, nextId = 1;
const shown = { SAH: true, IPH: true, IVH: true, total: true };

function log(msg) {
  const t = ((performance.now() - t0) / 1000).toFixed(2).padStart(7);
  logEl.textContent += `[${t}s] ${msg}\n`; logEl.scrollTop = logEl.scrollHeight;
  console.log('[sahvai-3d]', msg);
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const mL = v => v == null || isNaN(v) ? '–' : v < 10 ? v.toFixed(2) : v.toFixed(1);

(async () => {
  const bits = [];
  const gpu = navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null);
  bits.push(gpu ? 'WebGPU available' : 'no WebGPU');
  bits.push(crossOriginIsolated ? `CPU: ${Math.min(navigator.hardwareConcurrency || 4, 16)} threads` : 'CPU: 1 thread (open this page in its own tab for more)');
  $('caps').textContent = bits.join(' · ');
})();

// ---------------------------------------------------------------- worker
const device = () => document.querySelector('input[name=device]:checked').value;
const addMode = () => document.querySelector('input[name=addmode]:checked').value;
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
  if (addMode() === 'replace') clearAll();
  $('drop').classList.add('busy'); $('status').hidden = false;
  log('---- new upload'); setStage('Starting', 0); if (msg) log(msg);
  startWorker();
}
function run(files) {
  files = [...files];
  if (!files.length || busy) return;
  begin(`${files.length} file(s): ${files.slice(0, 3).map(f => f.webkitRelativePath || f.name).join(', ')}${files.length > 3 ? ', …' : ''}`);
  worker.postMessage({ multi: true, files, device: device() });
}
$('demo').onclick = () => {
  if (busy) return;
  begin('demo case: BrainHemoAI head.nrrd (aneurysmal SAH)');
  worker.postMessage({ multi: true, demo: 'demo/brainhemoai-demo-head.nrrd', device: device() });
};
function done() { busy = false; $('drop').classList.remove('busy'); showPct(); }

function setStage(name, frac) { stageNow = { name, frac }; $('stage').textContent = name; $('bar').style.width = `${Math.round(frac * 100)}%`; showPct(); }
function showPct() {
  const pct = stageNow.frac > 0 ? `${Math.round(stageNow.frac * 100)}%` : '';
  $('pct').textContent = busy ? [pct, `${((performance.now() - runStart) / 1000).toFixed(0)}s`].filter(Boolean).join(' · ') : pct;
}
setInterval(() => { if (busy) showPct(); }, 1000);

function onMessage({ data }) {
  if (data.type === 'log') log(data.msg);
  else if (data.type === 'ready') { if (!busy) setStage(scans.length ? 'Model ready. Drop more scans to add them to the timeline.' : 'Model ready. Drop a scan or try the demo case.', 1); }
  else if (data.type === 'progress') setStage(data.stage, data.frac);
  else if (data.type === 'error') { log(`ERROR: ${data.msg}`); setStage(`Error: ${data.msg}`, 0); done(); }
  else if (data.type === 'scans') {
    batchBase = scans.length;
    for (const s of data.list) scans.push({ id: nextId++, name: s.name, kind: s.kind, date: s.date, dateFrom: s.date ? (s.kind === 'dicom' ? 'DICOM header' : 'file name') : null, patientKey: s.patientKey, status: 'pending' });
    log(`${data.list.length} scan(s) found${data.list.length > 1 ? ', analysed oldest first' : ''}`);
    $('results').hidden = false; refreshAll();
  }
  else if (data.type === 'result') {
    const s = scans[batchBase + data.index]; if (!s) return;
    Object.assign(s, { status: 'done', res: data, surf: null });
    log(`scan ${data.index + 1}: SAH ${data.volumes.SAH.toFixed(1)} mL, IPH ${data.volumes.IPH.toFixed(1)} mL, IVH ${data.volumes.IVH.toFixed(1)} mL in ${data.seconds.toFixed(0)}s`);
    if (cur < 0 || scans[cur]?.status !== 'done' || data.count === 1) select(batchBase + data.index);
    else refreshAll();
  }
  else if (data.type === 'scanerror') {
    const s = scans[batchBase + data.index]; if (!s) return;
    s.status = 'error'; s.err = data.msg; log(`scan ${data.index + 1} failed: ${data.msg}`); refreshAll();
  }
  else if (data.type === 'alldone') {
    const ok = scans.filter(s => s.status === 'done').length;
    setStage(`Done in ${data.seconds.toFixed(0)}s · ${ok} scan${ok === 1 ? '' : 's'} on the timeline`, 1); done(); prebuild();
  }
}

// ---------------------------------------------------------------- timeline order and dates
const time = s => s.date ? new Date(s.date).getTime() : NaN;
function order() {
  return scans.map((s, i) => i).sort((a, b) => {
    const ta = time(scans[a]), tb = time(scans[b]);
    if (!isNaN(ta) && !isNaN(tb)) return ta - tb || a - b;
    if (!isNaN(ta)) return -1; if (!isNaN(tb)) return 1; return a - b;
  });
}
const fmtDate = s => {
  if (!s.date) return null;
  const d = new Date(s.date);
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) + ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
};
function dayOf(s) {
  const done = order().map(i => scans[i]).filter(x => x.status === 'done' && x.date);
  if (!done.length || !s.date) return null;
  return (time(s) - time(done[0])) / DAY;
}
const dayLbl = d => d == null ? '' : `Day ${Math.abs(d - Math.round(d)) < 0.05 ? Math.round(d) : d.toFixed(1)}`;

function clearAll() {
  for (const s of scans) disposeSurf(s);
  scans = []; cur = -1; batchBase = 0;
  $('results').hidden = true; refreshAll();
}
$('clear').onclick = () => { if (!busy) { clearAll(); setStage('Cleared. Drop a scan or try the demo case.', 0); } };

function refreshAll() {
  $('clear').hidden = !scans.length;
  renderStrip(); renderVols(); renderTimeline(); renderPlay();
  const keys = new Set(scans.map(s => s.patientKey).filter(Boolean));
  const pw = $('patwarn');
  pw.hidden = keys.size <= 1;
  pw.textContent = 'These scans carry different patient IDs. The timeline assumes one patient; remove scans that belong to someone else.';
}

function renderStrip() {
  const el = $('strip'), o = order();
  el.hidden = scans.length < 2;
  el.innerHTML = o.map((i, n) => {
    const s = scans[i], v = s.res?.volumes;
    const cls = ['chip', i === cur ? 'on' : '', s.status === 'pending' ? 'pending' : '', s.status === 'error' ? 'error' : ''].join(' ');
    const sub = s.status === 'done' ? `total ${mL(v.total)} mL` : s.status === 'error' ? 'failed' : 'waiting…';
    return `<button class="${cls}" data-i="${i}" type="button"><span class="d">${esc(dayLbl(dayOf(s)) || 'Scan ' + (n + 1))}</span><span class="t">${esc(fmtDate(s) || s.name)}</span><br><span class="t">${sub}</span></button>`;
  }).join('');
  el.querySelectorAll('.chip').forEach(b => b.onclick = () => { const i = +b.dataset.i; if (scans[i].status === 'done') select(i); });
}

function renderVols() {
  const s = scans[cur];
  if (!s || s.status !== 'done') { $('vols').innerHTML = ''; $('warn').hidden = true; $('details').innerHTML = ''; return; }
  const v = s.res.volumes;
  const card = (k, val, sub, c, cls = '') => `<div class="vol ${cls}" style="--c:${c}"><div class="k">${k}</div><div class="v">${mL(val)} <small>mL</small></div><div class="s">${sub}</div></div>`;
  $('vols').innerHTML = card('SAH (SAHV)', v.SAH, 'subarachnoid', 'var(--sah)') + card('IPH', v.IPH, 'intraparenchymal', 'var(--iph)') +
    card('IVH', v.IVH, 'intraventricular', 'var(--ivh)') + card('Total blood', v.total, 'SAH + IPH + IVH', 'var(--mayo-blue)', 'total');
  for (const k of ['SAH', 'IPH', 'IVH']) $('lv-' + k).textContent = `${mL(v[k])} mL`;
  const w = s.res.warnings || [];
  $('warn').hidden = !w.length; $('warn').innerHTML = w.map(esc).join('<br>');
  const i = s.res.info || {}, r = s.res;
  $('details').innerHTML = [
    i.series_description !== undefined ? `Series <b>${esc(i.series_description || '?')}</b> (${esc(i.series_selection || '')})` : `Source <b>${esc(i.source || s.name)}</b>`,
    s.date ? `acquired <b>${esc(fmtDate(s))}</b> (${esc(s.dateFrom || 'entered')})` : 'no date',
    `native ${i.native_size.join('×')} at ${i.native_spacing_mm.join(' × ')} mm → ${i.grid_size.join('×')} at ${i.grid_spacing_mm.join(' × ')} mm`,
    `rotation ${r.angle}°`, `${r.device === 'webgpu' ? 'WebGPU' : 'CPU'}, ${r.seconds.toFixed(0)} s`,
    s.surf ? `3D: ${(Object.values(s.surf.layers).reduce((a, l) => a + l.tris, 0) / 1000).toFixed(0)}k triangles in ${s.surf.ms.toFixed(0)} ms` : '',
    '<b>research use only, not for clinical decisions</b>',
  ].filter(Boolean).join(' · ');
}

function select(i) {
  const s = scans[i]; if (!s || s.status !== 'done') return;
  const prev = scans[cur], first = cur < 0 || !prev?.res;
  cur = i;
  const g = s.res.grid, P = g.nx * g.ny;
  // crosshair on the slice with most blood (kept in millimetres when stepping through a timeline)
  if (first || !cross.mm) {
    let best = Math.floor(g.nz / 2), bv = -1, sx = 0, sy = 0, n = 0;
    for (let z = 0; z < g.nz; z++) { let c = 0; for (let p = 0; p < P; p++) if (s.res.labels[z * P + p]) c++; if (c > bv) { bv = c; best = z; } }
    for (let p = 0; p < P; p++) if (s.res.labels[best * P + p]) { sx += p % g.nx; sy += (p / g.nx) | 0; n++; }
    cross.x = n ? Math.round(sx / n) : g.nx >> 1; cross.y = n ? Math.round(sy / n) : g.ny >> 1; cross.z = best;
  } else {
    cross.x = clamp(Math.round(cross.mm[0] / g.sp[0]), 0, g.nx - 1); cross.y = clamp(Math.round(cross.mm[1] / g.sp[1]), 0, g.ny - 1); cross.z = clamp(Math.round(cross.mm[2] / g.sp[2]), 0, g.nz - 1);
  }
  refreshAll(); setupMpr(); show3d(i);
}
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;

// ---------------------------------------------------------------- 3D view
const VIEWS = {
  bottom: [0, -1, 0.002], top: [0, 1, 0.002], front: [0, 0, 1], back: [0, 0, -1], left: [1, 0, 0], right: [-1, 0, 0],
};
const v3d = $('v3d'), glc = $('gl');
let renderer = null, scene, camera, controls, keyLight, planes, current3d = null, radius = 90;
try {
  renderer = new THREE.WebGLRenderer({ canvas: glc, antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(30, 1, 1, 5000);
  scene.add(camera);
  scene.add(new THREE.HemisphereLight(0xe6eeff, 0x1a1f2b, 1.25));
  keyLight = new THREE.DirectionalLight(0xffffff, 1.9); keyLight.position.set(0.6, 0.8, 1); camera.add(keyLight);
  const rim = new THREE.DirectionalLight(0x9fb3ff, 0.6); rim.position.set(-1, -0.3, -0.6); camera.add(rim);
  controls = new THREE.OrbitControls(camera, glc);
  controls.enableDamping = true; controls.dampingFactor = 0.12; controls.rotateSpeed = 0.85; controls.autoRotateSpeed = 2.2;
  controls.addEventListener('start', () => document.querySelectorAll('.views button').forEach(b => b.classList.remove('on')));
  planes = new THREE.Group(); planes.visible = false; scene.add(planes);
  new ResizeObserver(resize3d).observe(v3d); resize3d();
  setView('front', false);
  requestAnimationFrame(loop);
} catch (e) {
  renderer = null; log(`3D view unavailable: ${e.message}`);
  v3d.insertAdjacentHTML('beforeend', '<div class="busy3d">3D needs WebGL, which is not available in this browser.</div>');
}

function resize3d() {
  if (!renderer) return;
  const w = v3d.clientWidth, h = v3d.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix();
}

let tween = null;
function loop(now) {
  requestAnimationFrame(loop);
  if (tween) {
    const f = Math.min(1, (now - tween.t0) / 450), e = f < .5 ? 2 * f * f : 1 - (-2 * f + 2) ** 2 / 2;
    const d = tween.a.clone().lerp(tween.b, e).normalize().multiplyScalar(tween.r);
    camera.position.copy(controls.target).add(d); camera.lookAt(controls.target);
    if (f >= 1) tween = null;
  }
  controls.update();
  renderer.render(scene, camera);
  drawTriad();
}

function setView(name, animate = true) {
  if (!renderer) return;
  const r = radius / Math.sin(THREE_DEG(camera.fov) / 2) * 1.08;
  const b = new THREE.Vector3(...VIEWS[name]).normalize();
  controls.autoRotate = false; $('spin').checked = false;
  document.querySelectorAll('.views button').forEach(x => x.classList.toggle('on', x.dataset.view === name));
  if (!animate) { camera.position.copy(controls.target).addScaledVector(b, r); camera.lookAt(controls.target); return; }
  const a = camera.position.clone().sub(controls.target).normalize();
  if (a.dot(b) < -0.98) a.add(new THREE.Vector3(0.05, 0.08, 0.03)).normalize();   // avoid a 180-degree flip through zero
  tween = { a, b, r, t0: performance.now() };
}
function THREE_DEG(d) { return d * Math.PI / 180; }
document.querySelectorAll('.views button').forEach(b => b.onclick = () => setView(b.dataset.view));
$('reset').onclick = () => { if (!renderer) return; controls.target.set(0, 0, 0); setView('front'); };
$('spin').onchange = e => { if (controls) controls.autoRotate = e.target.checked; };
$('png3d').onclick = () => {
  if (!renderer) return;
  renderer.render(scene, camera);
  glc.toBlob(b => save(b, `sahvai-3d${scans[cur]?.date ? '-' + scans[cur].date.slice(0, 10) : ''}.png`));
};

// orientation triad: L/R red, A/P green, S/I blue, seen from the current camera
function drawTriad() {
  const c = $('triad'), ctx = c.getContext('2d'), W = c.width, cx = W / 2, cy = W / 2, L = W * 0.34;
  ctx.clearRect(0, 0, W, W);
  const q = camera.quaternion.clone().invert();
  const axes = [[[1, 0, 0], 'L', 'R', '#ff7676'], [[0, 0, 1], 'A', 'P', '#7ad67a'], [[0, 1, 0], 'S', 'I', '#78aaff']]
    .map(([v, p, n, col]) => ({ v: new THREE.Vector3(...v).applyQuaternion(q), p, n, col }))
    .sort((a, b) => a.v.z - b.v.z);
  ctx.font = '700 12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (const a of axes) {
    const x = a.v.x * L, y = -a.v.y * L;
    ctx.strokeStyle = a.col; ctx.globalAlpha = 0.35; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx - x, cy - y); ctx.stroke();
    ctx.fillStyle = a.col; ctx.fillText(a.n, cx - x * 1.3, cy - y * 1.3);
    ctx.globalAlpha = 1; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + x, cy + y); ctx.stroke();
    ctx.fillText(a.p, cx + x * 1.3, cy + y * 1.3);
  }
}

const MATS = {};
function mats() {
  if (MATS.SAH) return MATS;
  const blood = c => new THREE.MeshPhongMaterial({ color: c, shininess: 55, specular: 0x3a3a3a });
  MATS.SAH = blood(0xffb000); MATS.IPH = blood(0xff2d78); MATS.IVH = blood(0x1ec8ff);
  MATS.vent = new THREE.MeshPhongMaterial({ color: 0x6f8fd0, transparent: true, opacity: 0.42, depthWrite: false, shininess: 20 });
  MATS.brainBack = new THREE.MeshPhongMaterial({ color: 0xc9d3e6, transparent: true, opacity: 0.08, depthWrite: false, side: THREE.BackSide, shininess: 10 });
  MATS.brain = new THREE.MeshPhongMaterial({ color: 0xd5deef, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.FrontSide, shininess: 40, specular: 0x555566 });
  return MATS;
}
function geom(l) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(l.pos, 3)); g.setAttribute('normal', new THREE.BufferAttribute(l.nrm, 3));
  g.computeBoundingSphere(); return g;
}
function ensureSurf(s) {
  if (s.surf) return s.surf;
  const t = performance.now();
  s.surf = buildSurfaces(s.res.grid, s.res.mask, s.res.labels);
  log(`3D surfaces for ${s.name}: ${Object.entries(s.surf.layers).map(([k, l]) => `${k} ${(l.tris / 1000).toFixed(1)}k`).join(', ')} triangles, ${(performance.now() - t).toFixed(0)} ms` +
    (s.surf.ventricleMl != null ? `; approximate ventricles ${s.surf.ventricleMl.toFixed(1)} mL` : ''));
  return s.surf;
}
function groupFor(s) {
  if (s.group) return s.group;
  const surf = ensureSurf(s), M = mats(), G = new THREE.Group();
  const add = (k, m, order) => { const l = surf.layers[k]; if (!l || !l.tris) return; const mesh = new THREE.Mesh(geom(l), m); mesh.renderOrder = order; mesh.userData.layer = k; G.add(mesh); };
  add('SAH', M.SAH, 0); add('IPH', M.IPH, 0); add('IVH', M.IVH, 0);
  add('vent', M.vent, 1); add('brain', M.brainBack, 2); add('brain', M.brain, 3);
  s.group = G; return G;
}
function disposeSurf(s) {
  if (s.group) { s.group.traverse(o => o.geometry?.dispose()); if (scene && current3d === s.group) scene.remove(s.group); s.group = null; }
  if (current3d && !scans.some(x => x.group === current3d)) current3d = null;
}

function show3d(i) {
  if (!renderer) return;
  const s = scans[i];
  $('busy3d').hidden = !!s.group;
  // let the overlay paint before the (synchronous) meshing
  setTimeout(() => {
    const G = groupFor(s);
    if (current3d && current3d !== G) scene.remove(current3d);
    if (current3d !== G) scene.add(G);
    const firstTime = !current3d;
    current3d = G; applyLayers(); updatePlanes(); hud(); renderVols();
    $('busy3d').hidden = true;
    const brain = G.children.find(m => m.userData.layer === 'brain');
    if (brain && firstTime) { radius = brain.geometry.boundingSphere.radius; setView('front', false); }
  }, 30);
}
function applyLayers() {
  if (!current3d) return;
  const on = {}; document.querySelectorAll('[data-layer]').forEach(c => on[c.dataset.layer] = c.checked);
  const op = $('brainOp').value / 100;
  current3d.children.forEach(m => { m.visible = !!on[m.userData.layer] && !(m.userData.layer === 'brain' && op === 0); });
  if (MATS.brain) { MATS.brain.opacity = op; MATS.brainBack.opacity = op * 0.5; }
}
document.querySelectorAll('[data-layer]').forEach(c => c.onchange = applyLayers);
$('brainOp').oninput = applyLayers;
function hud() {
  const s = scans[cur]; if (!s?.res) return;
  const v = s.res.volumes, o = order().filter(i => scans[i].status === 'done'), n = o.indexOf(cur);
  $('hud').innerHTML = `<b>${esc(o.length > 1 ? (dayLbl(dayOf(s)) || `Scan ${n + 1}`) : 'SAHVAI-3D')}</b>${s.date ? ' · ' + esc(fmtDate(s)) : ''}<br>` +
    `<span style="color:#ffc94d">SAH ${mL(v.SAH)}</span> · <span style="color:#ff6fa0">IPH ${mL(v.IPH)}</span> · <span style="color:#6fdcff">IVH ${mL(v.IVH)}</span> · total ${mL(v.total)} mL`;
}

// slice planes in 3D, following the crosshair
$('planes').onchange = () => { if (planes) { planes.visible = $('planes').checked; updatePlanes(); } };
function updatePlanes() {
  if (!planes || !planes.visible) return;
  const s = scans[cur]; if (!s?.surf) return;
  const g = s.res.grid, c = s.surf.centre;
  const X = x => x * g.sp[0] - c[0], Y = z => z * g.sp[2] - c[2], Z = y => -(y * g.sp[1] - c[1]);
  const x0 = X(0), x1 = X(g.nx - 1), y0 = Y(0), y1 = Y(g.nz - 1), z0 = Z(0), z1 = Z(g.ny - 1);
  const px = X(cross.x), py = Y(cross.z), pz = Z(cross.y);
  const rect = (pts, col) => {
    const a = []; for (let i = 0; i < 4; i++) a.push(...pts[i], ...pts[(i + 1) % 4]);
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(a, 3));
    return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.85 }));
  };
  planes.children.forEach(o => { o.geometry.dispose(); o.material.dispose(); }); planes.clear();
  planes.add(rect([[x0, py, z0], [x1, py, z0], [x1, py, z1], [x0, py, z1]], 0xffe066));   // axial
  planes.add(rect([[x0, y0, pz], [x1, y0, pz], [x1, y1, pz], [x0, y1, pz]], 0x7ad67a));   // coronal
  planes.add(rect([[px, y0, z0], [px, y1, z0], [px, y1, z1], [px, y0, z1]], 0xff7676));   // sagittal
}

// build the other scans' surfaces in the background so 4D playback is instant
function prebuild() {
  const todo = scans.filter(s => s.status === 'done' && !s.group);
  const step = () => { const s = todo.shift(); if (!s) { renderVols(); return; } if (!s.group) groupFor(s); setTimeout(step, 60); };
  if (renderer) setTimeout(step, 200);
}

// ---------------------------------------------------------------- 4D playback
let playing = null;
function renderPlay() {
  const o = order().filter(i => scans[i].status === 'done');
  $('play').hidden = o.length < 2;
  $('tslider').max = Math.max(0, o.length - 1); $('tslider').value = Math.max(0, o.indexOf(cur));
  const s = scans[cur];
  $('playLbl').textContent = s ? `${dayLbl(dayOf(s)) || 'Scan ' + (o.indexOf(cur) + 1)} of ${o.length} scans` : '';
}
$('tslider').oninput = e => { const o = order().filter(i => scans[i].status === 'done'); const i = o[+e.target.value]; if (i !== undefined && i !== cur) select(i); };
$('playBtn').onclick = () => {
  if (playing) { clearInterval(playing); playing = null; $('playBtn').textContent = '▶ Play'; return; }
  $('playBtn').textContent = '❚❚ Pause';
  const step = () => { const o = order().filter(i => scans[i].status === 'done'); if (o.length < 2) return; select(o[(o.indexOf(cur) + 1) % o.length]); };
  step(); playing = setInterval(step, 1600);
};

// ---------------------------------------------------------------- orthogonal slices
const cross = { x: 0, y: 0, z: 0, mm: null };
const cvs = { ax: $('ax'), co: $('co'), sa: $('sa') };
function setupMpr() {
  const s = scans[cur]; if (!s) return;
  const g = s.res.grid, up = g.sp[2] / g.sp[0];   // stretch factor for the reformats
  cvs.ax.width = g.nx; cvs.ax.height = Math.round(g.ny * g.sp[1] / g.sp[0]);
  cvs.co.width = g.nx; cvs.co.height = Math.round(g.nz * up);
  cvs.sa.width = g.ny; cvs.sa.height = Math.round(g.nz * up);
  drawMpr();
}
function drawMpr() {
  const s = scans[cur]; if (!s) return;
  const g = s.res.grid, hu = g.hu, lab = s.res.labels, P = g.nx * g.ny;
  cross.mm = [cross.x * g.sp[0], cross.y * g.sp[1], cross.z * g.sp[2]];
  const [c, w] = document.querySelector('input[name=win]:checked').value.split(',').map(Number), lo = c - w / 2;
  const ov = $('overlay').checked, a = $('opacity').value / 100;
  const paint = (cv, W, H, idx) => {
    const ctx = cv.getContext('2d'), img = ctx.createImageData(W, H), d = img.data;
    for (let r = 0; r < H; r++) for (let q = 0; q < W; q++) {
      const p = idx(q, r); let v = (hu[p] - lo) / w * 255; v = v < 0 ? 0 : v > 255 ? 255 : v;
      let R = v, G = v, B = v; const l = lab[p];
      if (ov && l) { const col = COLORS[l]; R = R * (1 - a) + col[0] * a; G = G * (1 - a) + col[1] * a; B = B * (1 - a) + col[2] * a; }
      const o = (r * W + q) * 4; d[o] = R; d[o + 1] = G; d[o + 2] = B; d[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return ctx;
  };
  const zr = (r, H) => g.nz - 1 - Math.min(g.nz - 1, Math.floor(r * g.nz / H));   // superior at the top
  const ya = r => Math.min(g.ny - 1, Math.floor(r * g.ny / cvs.ax.height));
  const line = (ctx, x0, y0, x1, y1, col) => { ctx.strokeStyle = col; ctx.globalAlpha = 0.7; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.globalAlpha = 1; };
  // axial: x right = patient left (radiological), anterior at the top
  let ctx = paint(cvs.ax, g.nx, cvs.ax.height, (q, r) => cross.z * P + ya(r) * g.nx + q);
  const ay = (cross.y + 0.5) * cvs.ax.height / g.ny;
  line(ctx, cross.x + 0.5, 0, cross.x + 0.5, cvs.ax.height, '#ff7676'); line(ctx, 0, ay, g.nx, ay, '#7ad67a');
  // coronal: seen from the front, superior at the top
  const Hc = cvs.co.height, zy = (Hc - (cross.z + 0.5) * Hc / g.nz);
  ctx = paint(cvs.co, g.nx, Hc, (q, r) => zr(r, Hc) * P + cross.y * g.nx + q);
  line(ctx, cross.x + 0.5, 0, cross.x + 0.5, Hc, '#ff7676'); line(ctx, 0, zy, g.nx, zy, '#ffe066');
  // sagittal: anterior on the left, superior at the top
  ctx = paint(cvs.sa, g.ny, Hc, (q, r) => zr(r, Hc) * P + q * g.nx + cross.x);
  line(ctx, cross.y + 0.5, 0, cross.y + 0.5, Hc, '#7ad67a'); line(ctx, 0, zy, g.ny, zy, '#ffe066');
  const cnt = (sel) => { const c3 = [0, 0, 0, 0]; sel(c3); return [1, 2, 3].filter(k => c3[k]).map(k => `${NAMES[k]} ${(c3[k] * g.sp[0] * g.sp[1] * g.sp[2] / 1000).toFixed(2)} mL`).join(' · '); };
  const axs = cnt(c3 => { for (let p = 0; p < P; p++) c3[lab[cross.z * P + p]]++; });
  $('axcap').textContent = `Axial · slice ${cross.z + 1}/${g.nz}${axs ? ' · ' + axs : ''}`;
  $('cocap').textContent = `Coronal · ${(cross.y * g.sp[1]).toFixed(0)} mm from front of grid`;
  $('sacap').textContent = `Sagittal · ${(cross.x * g.sp[0]).toFixed(0)} mm from right of grid`;
  updatePlanes();
}
function pick(view, e) {
  const s = scans[cur]; if (!s) return;
  const g = s.res.grid, cv = cvs[view], r = cv.getBoundingClientRect();
  const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
  if (view === 'ax') { cross.x = clamp(Math.floor(fx * g.nx), 0, g.nx - 1); cross.y = clamp(Math.floor(fy * g.ny), 0, g.ny - 1); }
  if (view === 'co') { cross.x = clamp(Math.floor(fx * g.nx), 0, g.nx - 1); cross.z = clamp(g.nz - 1 - Math.floor(fy * g.nz), 0, g.nz - 1); }
  if (view === 'sa') { cross.y = clamp(Math.floor(fx * g.ny), 0, g.ny - 1); cross.z = clamp(g.nz - 1 - Math.floor(fy * g.nz), 0, g.nz - 1); }
  drawMpr();
}
for (const v of ['ax', 'co', 'sa']) {
  let down = false;
  cvs[v].addEventListener('pointerdown', e => { down = true; cvs[v].setPointerCapture(e.pointerId); pick(v, e); });
  cvs[v].addEventListener('pointermove', e => { if (down) pick(v, e); });
  cvs[v].addEventListener('pointerup', () => { down = false; });
  cvs[v].addEventListener('wheel', e => {
    const s = scans[cur]; if (!s) return; e.preventDefault();
    const g = s.res.grid, d = e.deltaY > 0 ? -1 : 1;
    if (v === 'ax') cross.z = clamp(cross.z + d, 0, g.nz - 1);
    if (v === 'co') cross.y = clamp(cross.y - d * 4, 0, g.ny - 1);
    if (v === 'sa') cross.x = clamp(cross.x - d * 4, 0, g.nx - 1);
    drawMpr();
  }, { passive: false });
}
document.querySelectorAll('input[name=win], #overlay, #opacity').forEach(el => el.addEventListener('input', drawMpr));

// ---------------------------------------------------------------- volume over time
function series() {
  const o = order().filter(i => scans[i].status === 'done'), pts = o.map(i => scans[i]);
  const dated = pts.length > 0 && pts.every(s => s.date);
  const t0 = dated ? time(pts[0]) : 0;
  return { idx: o, pts, dated, xs: pts.map((s, n) => dated ? (time(s) - t0) / DAY : n) };
}

function renderTimeline() {
  const doneN = scans.filter(s => s.status === 'done').length;
  $('t4d').hidden = scans.length < 2;
  if (scans.length < 2) return;
  $('legend4d').innerHTML = SERIES.map(S => `<label><input type="checkbox" data-s="${S.k}" ${shown[S.k] ? 'checked' : ''}><i style="background:${S.color}"></i>${S.label}</label>`).join('');
  $('legend4d').querySelectorAll('input').forEach(c => c.onchange = () => { shown[c.dataset.s] = c.checked; drawChart(); });
  renderTable(); drawChart(); summary(doneN);
}

let chartGeom = null;
function drawChart(hover = -1) {
  const cv = $('tchart'); if ($('t4d').hidden) return;
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext('2d'); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  const { pts, dated, xs, idx } = series();
  const m = { l: 52, r: 16, t: 14, b: dated ? 46 : 34 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
  ctx.font = '11px system-ui'; ctx.fillStyle = '#5f6673';
  if (!pts.length) { ctx.fillText('Waiting for the first scan…', m.l, m.t + 20); return; }
  const keys = SERIES.filter(S => shown[S.k]);
  let ymax = Math.max(1, ...pts.flatMap(s => keys.map(S => s.res.volumes[S.k])));
  const step = niceStep(ymax / 5); ymax = Math.ceil(ymax * 1.08 / step) * step;
  let xmin = Math.min(...xs), xmax = Math.max(...xs); if (xmax - xmin < 1e-6) { xmin -= 1; xmax += 1; }
  const pad = (xmax - xmin) * 0.06; xmin -= pad; xmax += pad;
  const X = x => m.l + (x - xmin) / (xmax - xmin) * pw, Y = y => m.t + ph - y / ymax * ph;
  // grid and y axis
  ctx.strokeStyle = '#e6e9ef'; ctx.lineWidth = 1; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let y = 0; y <= ymax + 1e-9; y += step) { ctx.beginPath(); ctx.moveTo(m.l, Y(y)); ctx.lineTo(W - m.r, Y(y)); ctx.stroke(); ctx.fillText(+y.toFixed(1), m.l - 6, Y(y)); }
  ctx.save(); ctx.translate(12, m.t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('Blood volume (mL)', 0, 0); ctx.restore();
  // current scan band
  const ci = idx.indexOf(cur);
  if (ci >= 0) { ctx.fillStyle = 'rgba(0,61,165,0.07)'; ctx.fillRect(X(xs[ci]) - 10, m.t, 20, ph); }
  // x axis: a tick per scan
  ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = '#5f6673'; ctx.strokeStyle = '#c3cad6';
  ctx.beginPath(); ctx.moveTo(m.l, m.t + ph); ctx.lineTo(W - m.r, m.t + ph); ctx.stroke();
  let lastX = -1e9;
  xs.forEach((x, n) => {
    const px = X(x); ctx.beginPath(); ctx.moveTo(px, m.t + ph); ctx.lineTo(px, m.t + ph + 4); ctx.stroke();
    if (px - lastX < 46) return; lastX = px;
    ctx.fillStyle = n === ci ? '#003da5' : '#5f6673'; ctx.font = (n === ci ? '700 ' : '') + '11px system-ui';
    ctx.fillText(dated ? dayLbl(x) : `Scan ${n + 1}`, px, m.t + ph + 6);
    if (dated) { const d = new Date(time(pts[n])); ctx.font = '10px system-ui'; ctx.fillStyle = '#8a91a0'; ctx.fillText(d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' }), px, m.t + ph + 20); }
  });
  if (!dated) { ctx.textAlign = 'right'; ctx.fillStyle = '#9a6700'; ctx.font = '11px system-ui'; ctx.fillText('Enter dates in the table to plot by date', W - m.r, m.t + ph + 20); }
  // lines
  for (const S of keys) {
    ctx.strokeStyle = S.color; ctx.lineWidth = S.k === 'total' ? 2.6 : 2; ctx.setLineDash(S.k === 'total' ? [] : []);
    ctx.beginPath(); pts.forEach((s, n) => { const px = X(xs[n]), py = Y(s.res.volumes[S.k]); n ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }); ctx.stroke();
    pts.forEach((s, n) => { ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(X(xs[n]), Y(s.res.volumes[S.k]), n === ci ? 5 : 3.5, 0, 7); ctx.fill(); ctx.fillStyle = S.color; ctx.beginPath(); ctx.arc(X(xs[n]), Y(s.res.volumes[S.k]), n === ci ? 3.5 : 2.3, 0, 7); ctx.fill(); });
  }
  // hover tooltip
  if (hover >= 0 && pts[hover]) {
    const s = pts[hover], px = X(xs[hover]), lines = [`${dated ? dayLbl(xs[hover]) + ' · ' : ''}${fmtDate(s) || s.name}`, ...SERIES.map(S => `${S.label}: ${mL(s.res.volumes[S.k])} mL`)];
    ctx.font = '11px system-ui'; const tw = Math.max(...lines.map(l => ctx.measureText(l).width)) + 16, th = lines.length * 15 + 8;
    const bx = Math.min(W - m.r - tw, Math.max(m.l, px + 10)), by = m.t + 4;
    ctx.fillStyle = 'rgba(0,32,91,0.93)'; ctx.beginPath(); ctx.roundRect(bx, by, tw, th, 6); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; lines.forEach((l, k) => ctx.fillText(l, bx + 8, by + 5 + k * 15));
  }
  chartGeom = { X, xs, idx };
}
function niceStep(x) { const p = 10 ** Math.floor(Math.log10(x || 1)), f = x / p; return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p; }
const nearestPoint = e => {
  if (!chartGeom) return -1;
  const r = $('tchart').getBoundingClientRect(), x = e.clientX - r.left; let best = -1, bd = 24;
  chartGeom.xs.forEach((v, n) => { const d = Math.abs(chartGeom.X(v) - x); if (d < bd) { bd = d; best = n; } });
  return best;
};
$('tchart').addEventListener('mousemove', e => drawChart(nearestPoint(e)));
$('tchart').addEventListener('mouseleave', () => drawChart());
$('tchart').addEventListener('click', e => { const n = nearestPoint(e); if (n >= 0) select(chartGeom.idx[n]); });
window.addEventListener('resize', () => drawChart());

function renderTable() {
  const o = order();
  const head = '<thead><tr><th>#</th><th>Date and time</th><th>Source</th><th>Day</th><th>SAH</th><th>IPH</th><th>IVH</th><th>Total</th><th>Δ total</th><th></th></tr></thead>';
  let prev = null;
  const rows = o.map((i, n) => {
    const s = scans[i], v = s.res?.volumes, val = s.date ? s.date.slice(0, 16) : '';
    const dTot = v && prev ? v.total - prev.total : null; if (v) prev = v;
    const cells = s.status === 'done' ? [mL(v.SAH), mL(v.IPH), mL(v.IVH), `<b>${mL(v.total)}</b>`, dTot == null ? '' : (dTot > 0 ? '+' : '') + mL(dTot)].map(x => `<td>${x}</td>`).join('')
      : `<td colspan="5" style="text-align:left">${s.status === 'error' ? 'failed: ' + esc(s.err) : 'analysing…'}</td>`;
    return `<tr data-i="${i}" class="${i === cur ? 'on' : ''}"><td>${n + 1}</td><td><input type="datetime-local" value="${val}" data-i="${i}" aria-label="Date of scan ${n + 1}"></td>` +
      `<td class="src" title="${esc(s.name)}">${esc(s.name)}</td><td>${esc(dayLbl(dayOf(s)).replace('Day ', ''))}</td>${cells}<td><button type="button" data-del="${i}" title="Remove this scan" aria-label="Remove scan ${n + 1}">×</button></td></tr>`;
  }).join('');
  $('ttable').innerHTML = head + '<tbody>' + rows + '</tbody>';
  $('ttable').querySelectorAll('tbody tr').forEach(tr => tr.onclick = e => { if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'BUTTON') select(+tr.dataset.i); });
  $('ttable').querySelectorAll('input[type=datetime-local]').forEach(inp => inp.onchange = () => {
    const s = scans[+inp.dataset.i]; s.date = inp.value ? inp.value + ':00' : null; s.dateFrom = 'entered'; refreshAll(); hud();
  });
  $('ttable').querySelectorAll('button[data-del]').forEach(b => b.onclick = () => {
    if (busy) return; const i = +b.dataset.del; disposeSurf(scans[i]); const wasCur = i === cur;
    scans.splice(i, 1); if (cur > i) cur--; if (wasCur) { cur = -1; const o2 = order().filter(k => scans[k].status === 'done'); if (o2.length) select(o2[0]); else { if (current3d) scene.remove(current3d); current3d = null; } }
    if (!scans.length) $('results').hidden = true; refreshAll();
  });
}

function summary(doneN) {
  const { pts, dated, xs } = series(), out = [];
  if (pts.length >= 2) {
    const a = pts[0].res.volumes, b = pts[pts.length - 1].res.volumes, span = xs[xs.length - 1] - xs[0];
    const pct = (x, y) => x > 0 ? ` (${y >= x ? '+' : ''}${((y - x) / x * 100).toFixed(0)}%)` : '';
    out.push(`<span>SAH <b>${mL(a.SAH)} → ${mL(b.SAH)} mL</b>${pct(a.SAH, b.SAH)}${dated ? ` over ${span.toFixed(1)} days` : ''}</span>`);
    out.push(`<span>Total blood <b>${mL(a.total)} → ${mL(b.total)} mL</b>${pct(a.total, b.total)}</span>`);
    if (dated && span > 0.2) out.push(`<span>Mean SAH change <b>${((b.SAH - a.SAH) / span).toFixed(1)} mL/day</b></span>`);
    const peak = pts.reduce((p, s, n) => s.res.volumes.total > p.v ? { v: s.res.volumes.total, n } : p, { v: -1, n: 0 });
    out.push(`<span>Peak total <b>${mL(peak.v)} mL</b> ${dated ? dayLbl(xs[peak.n]).toLowerCase() : 'at scan ' + (peak.n + 1)}</span>`);
    // log-linear SAH clearance (exploratory): ln(SAH) = a - k t
    const fit = pts.map((s, n) => [xs[n], s.res.volumes.SAH]).filter(([, v]) => v > 0.5);
    if (dated && fit.length >= 3 && span >= 1) {
      const n = fit.length, mx = fit.reduce((s, p) => s + p[0], 0) / n, my = fit.reduce((s, p) => s + Math.log(p[1]), 0) / n;
      let sxy = 0, sxx = 0, syy = 0; for (const [x, v] of fit) { const y = Math.log(v); sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; syy += (y - my) ** 2; }
      const k = -sxy / sxx, r2 = sxx && syy ? (sxy * sxy) / (sxx * syy) : 0;
      if (k > 0) out.push(`<span>SAH half-life <b>${(Math.LN2 / k).toFixed(1)} days</b> (log-linear fit, ${n} scans, R² ${r2.toFixed(2)}; exploratory)</span>`);
    }
  }
  if (doneN < scans.length) out.push(`<span class="muted">${scans.length - doneN} scan(s) still analysing or failed</span>`);
  $('tsum').innerHTML = out.join('');
}

// ---------------------------------------------------------------- exports
function save(blob, name) { const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: name }); a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
function rows() {
  const { pts, dated, xs } = series(), r3 = x => +x.toFixed(3);
  return pts.map((s, n) => ({ scan: n + 1, datetime: s.date || '', day: dated ? r3(xs[n]) : '', date_source: s.dateFrom || '', source: s.name,
    SAH_mL: r3(s.res.volumes.SAH), IPH_mL: r3(s.res.volumes.IPH), IVH_mL: r3(s.res.volumes.IVH), total_mL: r3(s.res.volumes.total) }));
}
$('dlCSV').onclick = () => {
  const R = rows(); if (!R.length) return;
  const cols = Object.keys(R[0]), q = v => /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v;
  save(new Blob([[cols.join(','), ...R.map(r => cols.map(c => q(r[c])).join(','))].join('\n') + '\n'], { type: 'text/csv' }), 'sahvai-4d-volumes.csv');
};
$('dlJSON').onclick = () => {
  const R = rows(); if (!R.length) return;
  save(new Blob([JSON.stringify({
    disclaimer: 'Research use only. Not a medical device and not for diagnosis or treatment decisions.',
    model: 'Hybrid 2D/3D UNet (Hu et al., NeuroImage 2023;279:120321), BrainHemoAI weights, run with ONNX Runtime Web',
    note: 'SAH is all subarachnoid blood found by the model, not the 5-cistern ABC/2 SAHV.',
    created: new Date().toISOString(), scans: R,
  }, null, 2)], { type: 'application/json' }), 'sahvai-4d-report.json');
};
$('dlChart').onclick = () => $('tchart').toBlob(b => save(b, 'sahvai-4d-chart.png'));

// ---------------------------------------------------------------- input
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

// test hook (headless checks)
window.__sahvai3d = { get scans() { return scans; }, get cur() { return cur; }, select, setView, order };
