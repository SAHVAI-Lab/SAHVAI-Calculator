// Research Dataset page: the site's de-identified dataset built by the tools (lib/research.js), with outcome entry,
// exports (CSV, Parquet, REDCap, FHIR, OMOP, JSON Lines, backup), import/merge and site settings.
import * as RX from './lib/research.js';

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const f1 = v => v == null || v === '' ? '' : (+v).toFixed(1), f2 = v => v == null || v === '' ? '' : (+v).toFixed(2);
let records = [];

// ---------- site settings ----------
function loadSettings() {
  const s = RX.settings();
  $('siteId').value = s.siteId || ''; $('country').value = s.country || ''; $('siteKey').value = s.siteKey;
}
function saveSite() { const s = RX.settings(); s.siteId = $('siteId').value.trim().toUpperCase(); s.country = $('country').value.trim().toUpperCase(); RX.saveSettings(s); look(); }
$('siteId').oninput = saveSite; $('country').oninput = saveSite;
$('showKey').onclick = () => { const k = $('siteKey'); k.type = k.type === 'password' ? 'text' : 'password'; $('showKey').textContent = k.type === 'password' ? 'Show' : 'Hide'; };
$('copyKey').onclick = async () => { try { await navigator.clipboard.writeText(RX.settings().siteKey); msg('siteMsg', 'Key copied. Store it securely.'); } catch { msg('siteMsg', 'Copy failed: select the key and copy it.', true); } };
$('saveKey').onclick = () => {
  const k = $('siteKey').value.trim().toLowerCase();
  if (!/^[0-9a-f]{32,128}$/.test(k)) { msg('siteMsg', 'A site key is 32 to 128 hexadecimal characters (as copied from another computer).', true); return; }
  const s = RX.settings(); if (s.siteKey === k) { msg('siteMsg', 'This key is already in use.'); return; }
  s.siteKey = k; RX.saveSettings(s); msg('siteMsg', 'Site key saved: new pseudonyms will use it.'); look();
};
async function look() { const v = $('lookId').value; $('lookOut').textContent = v.trim() ? await RX.pseudonym(v) : '—'; }
$('lookId').oninput = look;
function msg(id, t, err) { const el = $(id); el.textContent = t; el.classList.toggle('err', !!err); }

// ---------- the table ----------
const TOOLNAME = Object.fromEntries(Object.entries(RX.TOOLS).map(([k, t]) => [k, t.name]));
const SEXL = { 1: 'M', 2: 'F', 3: '?' };
const shown = () => { const t = $('filterTool').value; return records.filter(r => !t || r.tool === t); };
async function reload() {
  records = (await RX.allRecords()).sort((a, b) => String(a.subject_id).localeCompare(String(b.subject_id)) || (a.scan_number ?? 0) - (b.scan_number ?? 0) || String(a.tool).localeCompare(String(b.tool)));
  const tools = [...new Set(records.map(r => r.tool))], sel = $('filterTool'), cur = sel.value;
  sel.innerHTML = '<option value="">all tools</option>' + tools.map(t => `<option value="${esc(t)}">${esc(TOOLNAME[t] || t)}</option>`).join('');
  sel.value = tools.includes(cur) ? cur : '';
  stats(); table();
}
function stats() {
  const subj = new Set(records.map(r => r.subject_id)).size, sites = new Set(records.map(r => r.site_id).filter(Boolean));
  const byTool = Object.entries(records.reduce((m, r) => (m[r.tool] = (m[r.tool] || 0) + 1, m), {})).sort((a, b) => b[1] - a[1]);
  const withOut = records.filter(r => r.mrs_90d != null || r.dci != null || r.death_in_hospital != null).length;
  const card = (k, v, s) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
  $('stats').innerHTML = card('Scans', records.length, 'records in this browser') + card('Subjects', subj, 'distinct pseudonyms') +
    card('Sites', sites.size || '–', [...sites].join(', ') || 'set your site code') + card('With outcomes', withOut, 'DCI, mRS 90 d or death entered') +
    card('Tools', byTool.length, byTool.map(([t, n]) => `${TOOLNAME[t] || t} ${n}`).join(' · ') || '—');
}
const sel = (r, k, choices) => `<select data-id="${esc(r.record_id)}" data-k="${k}"><option value="">—</option>${choices.map(([c, l]) => `<option value="${c}"${String(r[k] ?? '') === String(c) ? ' selected' : ''}>${l}</option>`).join('')}</select>`;
function table() {
  const rows = shown();
  $('shown').textContent = rows.length === records.length ? '' : `${rows.length} of ${records.length} shown`;
  if (!rows.length) { $('table').innerHTML = `<tbody><tr><td class="empty">No records yet. Run a scan in any tool and press <b>Add to site dataset</b> in its Research export box.</td></tr></tbody>`; return; }
  const yn = [['1', 'yes'], ['0', 'no']], mrs = [0, 1, 2, 3, 4, 5, 6].map(v => [String(v), String(v)]);
  $('table').innerHTML = `<thead><tr><th class="l">Subject</th><th>Scan</th><th>h from onset</th><th class="l">Tool</th><th>SAH mL</th><th>IPH mL</th><th>IVH mL</th><th>Total mL</th><th>ABC/2 SAHV</th><th>P(any ICH)</th>
    <th>Age</th><th>Sex</th><th>GCS</th><th>WFNS</th><th>H&amp;H</th><th>mFisher</th><th>DCI</th><th>mRS 90 d</th><th>Died</th><th></th></tr></thead><tbody>` +
    rows.map(r => `<tr><td class="l mono">${esc(r.subject_id)}</td><td>${esc(r.scan_number ?? '')}</td><td>${f1(r.hours_from_onset)}</td><td class="l">${esc(TOOLNAME[r.tool] || r.tool)}</td>
      <td>${f1(r.vol_sah_ml)}</td><td>${f1(r.vol_iph_ml ?? r.vol_ich_ivh_ml)}</td><td>${f1(r.vol_ivh_ml)}</td><td>${f1(r.vol_total_blood_ml)}</td><td>${f1(r.abc2_cisternal_sahv_ml)}</td><td>${f2(r.det_any_p)}</td>
      <td>${esc(r.age_years ?? '')}</td><td>${esc(SEXL[r.sex] || '')}</td><td>${esc(r.gcs_admission ?? '')}</td><td>${esc(r.wfns_grade ?? '')}</td><td>${esc(r.hunt_hess_grade ?? '')}</td><td>${esc(r.modified_fisher ?? '')}</td>
      <td>${sel(r, 'dci', yn)}</td><td>${sel(r, 'mrs_90d', mrs)}</td><td>${sel(r, 'death_in_hospital', yn)}</td>
      <td><button class="del" data-del="${esc(r.record_id)}" title="Delete this record">✕</button></td></tr>`).join('') + '</tbody>';
}
$('filterTool').onchange = table;
$('table').addEventListener('change', async e => {
  const s = e.target.closest('select[data-id]'); if (!s) return;
  const r = records.find(x => x.record_id === s.dataset.id); if (!r) return;
  r[s.dataset.k] = s.value === '' ? null : +s.value;
  await RX.putRecord(r); RX.notifyChange(); stats();
});
$('table').addEventListener('click', async e => {
  const b = e.target.closest('[data-del]'); if (!b) return;
  if (b.dataset.armed !== '1') { b.dataset.armed = '1'; b.textContent = 'delete?'; setTimeout(() => { b.dataset.armed = ''; b.textContent = '✕'; }, 3000); return; }
  await RX.deleteRecord(b.dataset.del); RX.notifyChange(); reload();
});

// ---------- exports ----------
const stamp = () => `${(RX.settings().siteId || 'site').toLowerCase()}_${new Date().toISOString().slice(0, 10)}`;
const blob = (s, type) => new Blob([s], { type });
const README = rows => `HEADS-UP / SAHVAI site research dataset (${RX.SCHEMA_VERSION})
${rows.length} scan records, ${new Set(rows.map(r => r.subject_id)).size} subjects; exported ${new Date().toISOString().slice(0, 10)} from ${location.origin}

dataset.csv                  all variables, one row per scan and tool (UTF-8); variables in data_dictionary.csv
dataset.parquet              the same, typed (pandas, Arrow, DuckDB, BigQuery / Terra)
data_dictionary.csv          label, type, unit, codes and NINDS CDE concept of every variable
redcap_data_dictionary.csv   upload in REDCap to create the project (forms: scan record, hemorrhage volumes, clinical, outcomes)
redcap_records.csv           then import with the REDCap Data Import Tool
fhir_bundle.json             HL7 FHIR R4 collection: Patient (pseudonym only), Device (model), Observations
omop_measurement.csv         OMOP-style MEASUREMENT staging rows (map concept ids and add dates at your site)
dataset.jsonl                JSON Lines, e.g. npx convex import --table hemorrhage_scans dataset.jsonl

De-identified at the source: pseudonymous subject IDs (HMAC-SHA-256 with the site key, which is not included), hours from
onset instead of dates, age capped at 90, no images, names, MRNs or free text. Share only under your site's approvals.
Research and education use only; not a medical device.
`;
async function exportAs(kind) {
  const rows = shown(); if (!rows.length) { msg('exMsg', 'The dataset is empty.', true); return; }
  const s = stamp();
  const files = {
    csv: () => [`dataset_${s}.csv`, blob(RX.toCSV(rows), 'text/csv')],
    dict: () => [`data_dictionary.csv`, blob(RX.dictionaryCSV(), 'text/csv')],
    redcapDict: () => [`redcap_data_dictionary.csv`, blob(RX.redcapDictionaryCSV(), 'text/csv')],
    redcapData: () => [`redcap_records_${s}.csv`, blob(RX.toCSV(rows), 'text/csv')],
    parquet: async () => [`dataset_${s}.parquet`, new Blob([await RX.toParquet(rows)], { type: 'application/vnd.apache.parquet' })],
    fhir: () => [`fhir_bundle_${s}.json`, blob(JSON.stringify(RX.toFHIR(rows), null, 2), 'application/fhir+json')],
    omop: () => [`omop_measurement_${s}.csv`, blob(RX.toOMOP(rows), 'text/csv')],
    jsonl: () => [`dataset_${s}.jsonl`, blob(RX.toJSONL(rows), 'application/x-ndjson')],
    backup: () => [`backup_${s}.json`, blob(JSON.stringify({ format: 'sahvai-research-backup', schema_version: RX.SCHEMA_VERSION, exported: new Date().toISOString(), records: rows }, null, 1), 'application/json')],
  };
  if (kind === 'all') {
    const z = await RX.zip([
      { name: 'README.txt', data: README(rows) }, { name: 'dataset.csv', data: RX.toCSV(rows) }, { name: 'dataset.parquet', data: new Uint8Array(await RX.toParquet(rows)) },
      { name: 'data_dictionary.csv', data: RX.dictionaryCSV() }, { name: 'redcap_data_dictionary.csv', data: RX.redcapDictionaryCSV() }, { name: 'redcap_records.csv', data: RX.toCSV(rows) },
      { name: 'fhir_bundle.json', data: JSON.stringify(RX.toFHIR(rows), null, 2) }, { name: 'omop_measurement.csv', data: RX.toOMOP(rows) }, { name: 'dataset.jsonl', data: RX.toJSONL(rows) }]);
    RX.saveBlob(z, `heads-up-research_${s}.zip`); msg('exMsg', `Downloaded ${rows.length} records in all formats.`); return;
  }
  const [name, b] = await files[kind](); RX.saveBlob(b, name); msg('exMsg', `Downloaded ${name} (${rows.length} records).`);
}
document.querySelectorAll('[data-x]').forEach(b => b.onclick = () => exportAs(b.dataset.x).catch(e => msg('exMsg', `Export failed: ${e.message}`, true)));

// ---------- import and clear ----------
function parseCSV(text) {
  const rows = []; let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c; }
    else if (c === '"') q = true; else if (c === ',') { row.push(cell); cell = ''; } else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(cell); rows.push(row); row = []; cell = ''; } else cell += c;
  }
  if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
  return rows.filter(r => r.some(x => x !== ''));
}
const typed = rec => {
  const out = {};
  for (const f of RX.FIELDS) {
    let v = rec[f.name]; if (v === '' || v === undefined) v = null;
    out[f.name] = v == null ? null : ['integer', 'number', 'yesno'].includes(f.type) ? (isFinite(+v) ? +v : null) : String(v);
  }
  return out;
};
$('importFile').onchange = async e => {
  const file = e.target.files[0]; e.target.value = ''; if (!file) return;
  try {
    const text = await file.text(); let recs = [];
    if (/\.csv$/i.test(file.name)) {
      const [h, ...body] = parseCSV(text);
      if (!h || !h.includes('record_id')) throw new Error('not a dataset CSV from these tools (no record_id column)');
      recs = body.map(r => Object.fromEntries(h.map((k, i) => [k, r[i]])));
    } else if (/\.jsonl$/i.test(file.name)) recs = text.split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
    else { const j = JSON.parse(text); recs = Array.isArray(j) ? j : j.records || []; }
    recs = recs.filter(r => r && r.record_id).map(typed);
    if (!recs.length) throw new Error('no records found');
    for (const r of recs) await RX.putRecord(r);
    RX.notifyChange(); await reload(); msg('imMsg', `Merged ${recs.length} record${recs.length === 1 ? '' : 's'} from ${file.name}.`);
  } catch (err) { msg('imMsg', `Import failed: ${err.message}`, true); }
};
$('clearAll').onclick = async () => {
  const b = $('clearAll');
  if (b.dataset.armed !== '1') { b.dataset.armed = '1'; b.textContent = 'Click again to delete every record (download a backup first)'; setTimeout(() => { b.dataset.armed = ''; b.textContent = 'Clear the dataset…'; }, 5000); return; }
  await RX.clearRecords(); RX.notifyChange(); b.dataset.armed = ''; b.textContent = 'Clear the dataset…'; msg('imMsg', 'The dataset was cleared.'); reload();
};

// ---------- start ----------
loadSettings(); reload().catch(e => { $('table').innerHTML = `<tbody><tr><td class="empty">This browser does not allow storage for this page (${esc(e.message)}). Use each tool's “Download this scan”.</td></tr></tbody>`; });
RX.onChange(() => reload());
window.addEventListener('focus', () => { loadSettings(); reload(); });
if (window.parent !== window) {
  document.body.classList.add('embedded');
  const rep = () => window.parent.postMessage({ type: 'sahvai-web-height', page: 'research', height: Math.ceil(document.body.getBoundingClientRect().height) + 24 }, location.origin);
  new ResizeObserver(rep).observe(document.body); rep();
}
