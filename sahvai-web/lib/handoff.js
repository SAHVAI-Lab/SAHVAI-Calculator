// Hand-off of a scan between SAHVAI Web tools (same origin): HEADS-UP stores the scan the user gave it
// (the original files, or a demo case URL) plus its findings in IndexedDB, then opens a segmentation tool
// with ?handoff=<id>. The tool reads the entry and runs the scan as if it had been dropped there.
// Nothing leaves the browser.

const DB = 'sahvai-handoff', STORE = 'scans', MAX_AGE = 24 * 3600 * 1000;

function db() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
}
function tx(mode, fn) {
  return db().then(d => new Promise((res, rej) => {
    const t = d.transaction(STORE, mode), out = fn(t.objectStore(STORE));
    t.oncomplete = () => { d.close(); res(out && 'result' in out ? out.result : undefined); };
    t.onerror = () => { d.close(); rej(t.error); };
  }));
}

// entry: { files: File[] } or { demo: url, demoName }, plus { from, summary, note }
export async function putHandoff(entry) {
  const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  await tx('readwrite', s => {
    const c = s.openCursor();   // drop entries older than a day
    c.onsuccess = () => { const k = c.result; if (!k) return; if (Date.now() - (k.value.created || 0) > MAX_AGE) k.delete(); k.continue(); };
    s.put({ ...entry, created: Date.now() }, id);
  });
  return id;
}

export async function getHandoff(id) {
  if (!id || !self.indexedDB) return null;
  try { return (await tx('readonly', s => s.get(id))) || null; } catch { return null; }
}

// the segmentation tools a HEADS-UP result can be sent to (page in sahvai-web/, tab id in the SAHVAI Calculator)
export const TARGETS = {
  web: { page: 'index.html', tab: 'web', name: 'SAHVAI (BrainHemoAI)', does: 'SAH, IPH and IVH volumes (Hybrid 2D/3D UNet)' },
  sahswin: { page: 'sahswin.html', tab: 'sahswin', name: 'SAH Swin UNETR', does: 'blood volume after aneurysmal SAH (Valladolid)' },
  claim: { page: 'claim.html', tab: 'claim', name: 'CLAIM aSAH', does: 'SAH, IVH, ICH, SDH, aneurysm and ventricles (Charité)' },
  deepbleed: { page: 'deepbleed.html', tab: 'deepbleed', name: 'DeepBleed ICH', does: 'ICH + IVH volume (MISTIE, Johns Hopkins)' },
  blast: { page: 'blast.html', tab: 'blast', name: 'BLAST-CT', does: 'traumatic IPH, extra-axial blood (SDH, EDH, tSAH), oedema, IVH' },
};
// which tools suit which subtype, best first
export const ROUTES = {
  SAH: ['web', 'sahswin', 'claim'],
  IPH: ['deepbleed', 'web', 'blast'],
  IVH: ['web', 'deepbleed', 'claim'],
  SDH: ['blast', 'claim'],
  EDH: ['blast'],
};

// open a tool with the hand-off: as a tab of the SAHVAI Calculator when embedded there, else as a page
export function openTarget(key, id) {
  const t = TARGETS[key];
  if (window.parent !== window) {
    window.parent.postMessage({ type: 'sahvai-open-tab', tab: t.tab, page: t.page, handoff: id }, location.origin);
  } else location.href = `${t.page}?handoff=${encodeURIComponent(id)}`;
}

// on a receiving page: the hand-off named in the URL, if any (the parameter is removed so a reload does not re-run it)
export async function receivedHandoff() {
  const u = new URL(location.href), id = u.searchParams.get('handoff');
  if (!id) return null;
  u.searchParams.delete('handoff'); history.replaceState(null, '', u.pathname + (u.search || '') + u.hash);
  return getHandoff(id);
}

// a short banner describing what HEADS-UP found, for the receiving page
export function handoffBanner(h, toolName) {
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const found = (h.summary || []).filter(s => s.key !== 'any' && s.call !== 'not detected')
    .map(s => `${s.key} ${(s.max * 100).toFixed(0)}%${s.call === 'possible' ? ' (possible)' : ''}`);
  const div = document.createElement('div');
  div.className = 'note handoff';
  div.innerHTML = `<h2>Sent from HEADS-UP</h2><p>${found.length ? `HEADS-UP flagged <b>${esc(found.join(', '))}</b>` : 'HEADS-UP found no hemorrhage subtype above its thresholds'} on <b>${esc(h.label || 'this scan')}</b>.
    ${esc(toolName)} is now running on the same scan. Review the result against the images; neither tool is for clinical use.</p>`;
  return div;
}

// the scan of a hand-off as files (a demo case URL is fetched into a File)
export async function handoffFiles(h) {
  if (h.files && h.files.length) return [...h.files];
  if (h.demo) {
    const r = await fetch(new URL(h.demo, location.href)); if (!r.ok) throw new Error(`could not download ${h.demo}`);
    return [new File([await r.blob()], h.demo.split('/').pop())];
  }
  return [];
}
