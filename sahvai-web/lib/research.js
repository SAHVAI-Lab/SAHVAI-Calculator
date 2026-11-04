// Research exports for the SAHVAI Lab & HEADS-UP (CODE ICH) hemorrhage tools.
//
// Each site keeps its own de-identified hemorrhage volumetric dataset: one row per scan and tool, stored in this
// browser (IndexedDB, shared by every tool page on this site) and exported in formats that local research systems
// take: CSV + data dictionary (NIH ScHARe, REDCap, R / Python), Parquet, REDCap data dictionary, FHIR R4 bundle,
// OMOP-style MEASUREMENT rows, JSON Lines (e.g. a Convex table) and a per-scan zip with a NIfTI label map.
//
// Privacy by design: images and names never leave the computer; the study ID typed here is turned into a pseudonym
// on this computer (HMAC-SHA-256 with a key that stays at the site), dates become hours from onset, age is capped at
// 90, and free text is not collected. What is exported is decided by the site, under its own approvals.

export const SCHEMA_VERSION = 'heads-up-research-1.0';
export const APP_VERSION = '2026-11-04';
const SITE_URL = 'https://sahvai-lab.github.io/SAHVAI-Calculator/';
const CODE_SYSTEM = SITE_URL + 'fhir/CodeSystem/heads-up-research';
const CDE_SAH = 'NINDS UIA & SAH CDE v1.0 (Suarez et al., Neurocrit Care 2019;30:4-19)';
const CDE_GEN = 'NINDS General CDE';

// ---------- the tools ----------
export const TOOLS = {
  sahvai: { name: 'SAHVAI Web (BrainHemoAI)', model: 'Hybrid 2D/3D UNet (Hu et al., NeuroImage 2023;279:120321), BrainHemoAI weights', version: 'v2', licence: 'MIT (with permission of Dr Ping Hu)' },
  sahvai_3d: { name: 'SAHVAI 3D-4D', model: 'Hybrid 2D/3D UNet (Hu et al., NeuroImage 2023;279:120321), BrainHemoAI weights', version: 'v2', licence: 'MIT (with permission of Dr Ping Hu)' },
  claim: { name: 'CLAIM aSAH', model: 'nnU-Net v2 2D multiclass aSAH model (Kiewitz et al., Front Neurol 2024;15:1490216)', version: 'c1', licence: 'CC BY-NC 4.0' },
  blast: { name: 'BLAST-CT', model: 'BLAST-CT DeepMedic (Monteiro et al., Lancet Digit Health 2020;2:e314-22)', version: 'b1', licence: 'Apache-2.0' },
  deepbleed: { name: 'DeepBleed ICH', model: 'DeepBleed 3D V-Net (Sharrock et al., Neuroinformatics 2021;19:403-415), MISTIE-2 weights', version: 'd2', licence: 'CC BY-NC 4.0 (with permission of Dr Matthew Sharrock)' },
  sahswin: { name: 'SAH Swin UNETR (Valladolid)', model: 'Swin UNETR (García-García, Cepeda et al., World Neurosurg 2024; Brain Sci 2024;14:10)', version: 's1', licence: 'Academic non-commercial (with permission of Dr Santiago Cepeda)' },
  headsup: { name: 'HEADS-UP detector', model: 'RSNA 2019 1st-place ICH detector (Wang et al., NeuroImage Clin 2021;32:102785)', version: 'h1', licence: 'see the HEADS-UP page' },
  hesigns: { name: 'ICH Expansion Signs', model: 'SAHVAI segmentation + rule-based CT signs; optional Yale HE model (Tran et al., npj Digit Med 2024)', version: 'e1', licence: 'see the ICH Expansion Signs page' },
  abc2: { name: 'SAHVAI Model 1 (ABC/2 manual)', model: 'Manual ABC/2 cisternal SAH volume (Föttinger et al., J Am Heart Assoc 2024;13:e032195)', version: 'm1', licence: 'n/a (manual measurement)' },
};

// ---------- the data dictionary ----------
// type: text | integer | number | category | yesno. choices: [[code, label], ...]
const F = (name, group, type, label, o = {}) => ({ name, group, type, label, ...o });
const VOL = (name, label, o = {}) => F(name, 'volumes', 'number', label, { unit: 'mL', ...o });
const SEX = [['1', 'Male'], ['2', 'Female'], ['3', 'Unknown or not reported']];
const RACE = [['1', 'American Indian or Alaska Native'], ['2', 'Asian'], ['3', 'Black or African American'], ['4', 'Hispanic or Latino'], ['5', 'Middle Eastern or North African'],
  ['6', 'Native Hawaiian or Pacific Islander'], ['7', 'White'], ['8', 'More than one category'], ['9', 'Unknown or not reported']];
export const FIELDS = [
  F('record_id', 'admin', 'text', 'Record ID', { note: 'random UUID made when the record is created' }),
  F('schema_version', 'admin', 'text', 'Schema version'),
  F('site_id', 'admin', 'text', 'Site code', { note: 'short code chosen by the site, e.g. MCF' }),
  F('site_country', 'admin', 'text', 'Site country', { note: 'ISO 3166-1 alpha-2 code, e.g. US, BR, ES' }),
  F('subject_id', 'admin', 'text', 'Subject pseudonym', { note: 'study ID turned into a pseudonym on the site computer (HMAC-SHA-256, site key), or a study code as entered; never an MRN or name' }),
  F('subject_id_method', 'admin', 'category', 'Subject ID method', { choices: [['hmac', 'Pseudonymised on site (HMAC-SHA-256, site key)'], ['entered', 'Study code as entered by the site']] }),
  F('scan_number', 'admin', 'integer', 'Scan number for this subject', { note: '1 = first (admission) CT', min: 1 }),
  F('hours_from_onset', 'admin', 'number', 'Hours from onset to scan', { unit: 'h', note: 'from ictus (or last known well) to CT; calendar dates are not exported', cde: 'Time from symptom onset to imaging', source: CDE_SAH }),
  F('record_created_month', 'admin', 'text', 'Month the record was created', { note: 'YYYY-MM' }),
  F('input_format', 'imaging', 'category', 'Input format', { choices: [['dicom', 'DICOM'], ['nifti', 'NIfTI'], ['nrrd', 'NRRD'], ['demo', 'Public demo case'], ['manual', 'Manual measurement']] }),
  F('scanner_manufacturer', 'imaging', 'text', 'Scanner manufacturer', { note: 'DICOM (0008,0070)' }),
  F('kvp', 'imaging', 'number', 'Tube voltage', { unit: 'kV', note: 'DICOM (0018,0060)' }),
  F('slice_thickness_mm', 'imaging', 'number', 'Slice thickness', { unit: 'mm' }),
  F('recon_kernel', 'imaging', 'text', 'Reconstruction kernel', { note: 'DICOM (0018,1210)' }),
  F('grid_spacing_mm', 'imaging', 'text', 'Analysis grid spacing (x, y, z)', { unit: 'mm' }),
  F('skull_stripped', 'imaging', 'yesno', 'Input was skull-stripped'),
  F('tool', 'analysis', 'category', 'Tool', { choices: Object.entries(TOOLS).map(([k, t]) => [k, t.name]) }),
  F('model', 'analysis', 'text', 'Model and citation'),
  F('model_version', 'analysis', 'text', 'Model file version'),
  F('model_licence', 'analysis', 'text', 'Model licence'),
  F('app_version', 'analysis', 'text', 'App version'),
  F('compute', 'analysis', 'category', 'Compute', { choices: [['webgpu', 'WebGPU'], ['wasm', 'CPU (WebAssembly)'], ['manual', 'Manual']] }),
  F('runtime_s', 'analysis', 'number', 'Run time', { unit: 's' }),
  F('n_warnings', 'analysis', 'integer', 'Number of warnings shown'),
  VOL('vol_sah_ml', 'Subarachnoid hemorrhage volume (AI)'),
  VOL('vol_iph_ml', 'Intraparenchymal (ICH) hemorrhage volume (AI)'),
  VOL('vol_ivh_ml', 'Intraventricular hemorrhage volume (AI)'),
  VOL('vol_sdh_ml', 'Subdural hematoma volume (AI)'),
  VOL('vol_eah_ml', 'Extra-axial hemorrhage volume (AI; SAH + SDH + EDH, BLAST-CT)'),
  VOL('vol_ich_ivh_ml', 'ICH + IVH volume, one class (AI; DeepBleed)'),
  VOL('vol_total_blood_ml', 'Total hemorrhage volume (AI)', { note: 'all hemorrhage classes of the tool; for DeepBleed ICH + IVH only' }),
  VOL('vol_oedema_ml', 'Perilesional oedema volume (AI; BLAST-CT)'),
  VOL('vol_ventricles_ml', 'Ventricular CSF volume (AI; CLAIM)'),
  VOL('vol_aneurysm_ml', 'Aneurysm volume (AI; CLAIM)'),
  VOL('vol_brain_ml', 'Brain volume seen by the model'),
  VOL('vol_template_space_ml', 'Hemorrhage volume in template space (SAH Swin UNETR)'),
  F('abc2_cisternal_sahv_ml', 'manual', 'number', 'Cisternal SAH volume, 5 spaces (ABC/2)', { unit: 'mL' }),
  F('abc2_iph_ml', 'manual', 'number', 'Intraparenchymal hemorrhage volume (ABC/2)', { unit: 'mL' }),
  F('abc2_sdh_ml', 'manual', 'number', 'Subdural hematoma volume (ABC/2 ÷ 2)', { unit: 'mL' }),
  F('abc2_total_ml', 'manual', 'number', 'Total ABC/2 hemorrhage volume', { unit: 'mL' }),
  F('mgraeb_score', 'manual', 'integer', 'Modified Graeb score', { min: 0, max: 32, cde: 'Modified Graeb score' }),
  F('esah_score', 'manual', 'integer', 'eSAH score', { min: 0, max: 5, note: 'Sharma et al., Sci Rep 2024' }),
  F('ref_blood_ml', 'reference', 'number', 'Reference (expert) hemorrhage volume', { unit: 'mL' }),
  F('ref_dice_blood', 'reference', 'number', 'Dice, AI vs reference hemorrhage', { min: 0, max: 1 }),
  ...['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'].map(k => F(`det_${k}_p`, 'detection', 'number', `HEADS-UP probability: ${k === 'any' ? 'any intracranial hemorrhage' : k}`, { min: 0, max: 1 })),
  F('det_any_call', 'detection', 'category', 'HEADS-UP call (any hemorrhage)', { choices: [['detected', 'Detected'], ['possible', 'Possible'], ['not_detected', 'Not detected']] }),
  F('he_hematoma_ml', 'expansion', 'number', 'Index hematoma volume (ICH Expansion Signs)', { unit: 'mL' }),
  F('he_signs_present', 'expansion', 'text', 'Expansion signs present (semicolon-separated)'),
  F('he_bat_score', 'expansion', 'integer', 'BAT score'),
  F('he_brain_score', 'expansion', 'integer', 'BRAIN score'),
  F('he_yale_p', 'expansion', 'number', 'Yale model probability of expansion ≥ 6 mL', { min: 0, max: 1 }),
  F('age_years', 'clinical', 'integer', 'Age', { unit: 'years', note: '90 = 90 or older', min: 0, max: 90, cde: 'Age in years', source: CDE_GEN }),
  F('sex', 'clinical', 'category', 'Sex', { choices: SEX, cde: 'Sex', source: CDE_GEN }),
  F('race_ethnicity', 'clinical', 'category', 'Race and ethnicity', { choices: RACE, note: 'US OMB SPD 15 (2024) categories; optional, may be left blank outside the US', source: 'OMB SPD 15 (2024)' }),
  F('gcs_admission', 'clinical', 'integer', 'Glasgow Coma Scale total, admission', { min: 3, max: 15, cde: 'Glasgow Coma Scale total score', source: CDE_SAH }),
  F('wfns_grade', 'clinical', 'integer', 'WFNS grade', { min: 1, max: 5, cde: 'World Federation of Neurological Surgeons grade', source: CDE_SAH }),
  F('hunt_hess_grade', 'clinical', 'integer', 'Hunt and Hess grade', { min: 1, max: 5, cde: 'Hunt and Hess grade', source: CDE_SAH }),
  F('modified_fisher', 'clinical', 'integer', 'Modified Fisher scale', { min: 0, max: 4, cde: 'Modified Fisher scale', source: CDE_SAH }),
  F('ich_score', 'clinical', 'integer', 'ICH score (Hemphill)', { min: 0, max: 6, cde: 'ICH score' }),
  F('dci', 'outcome', 'yesno', 'Delayed cerebral ischemia', { cde: 'Delayed cerebral ischemia', source: CDE_SAH, note: 'filled in later' }),
  F('mrs_90d', 'outcome', 'integer', 'Modified Rankin Scale at 90 days', { min: 0, max: 6, cde: 'Modified Rankin Scale score', source: CDE_SAH, note: 'filled in later' }),
  F('death_in_hospital', 'outcome', 'yesno', 'Died in hospital', { note: 'filled in later' }),
];
export const FIELD = Object.fromEntries(FIELDS.map(f => [f.name, f]));
export const CLINICAL = ['age_years', 'sex', 'race_ethnicity', 'gcs_admission', 'wfns_grade', 'hunt_hess_grade', 'modified_fisher', 'ich_score'];
export const OUTCOMES = ['dci', 'mrs_90d', 'death_in_hospital'];

// ---------- settings (site code, country, site key) ----------
const SKEY = 'sahvai-research-settings';
export function settings() {
  let s = {}; try { s = JSON.parse(localStorage.getItem(SKEY) || '{}'); } catch { /* ignore */ }
  if (!s.siteKey) { const b = crypto.getRandomValues(new Uint8Array(32)); s.siteKey = [...b].map(x => x.toString(16).padStart(2, '0')).join(''); saveSettings(s); }
  return s;
}
export function saveSettings(s) { try { localStorage.setItem(SKEY, JSON.stringify(s)); } catch { /* ignore */ } }

// pseudonym = site code + 12 base-32 characters of HMAC-SHA-256(site key, normalised study ID)
export async function pseudonym(studyId, s = settings()) {
  const id = String(studyId || '').trim().toUpperCase().replace(/\s+/g, '');
  if (!id) return '';
  const key = await crypto.subtle.importKey('raw', hexBytes(s.siteKey), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const mac = new Uint8Array(await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(id)));
  const A = 'ABCDEFGHJKMNPQRSTVWXYZ0123456789'; let out = '', bits = 0, v = 0;
  for (const b of mac) { v = (v << 8) | b; bits += 8; while (bits >= 5 && out.length < 12) { out += A[(v >> (bits - 5)) & 31]; bits -= 5; } }
  return `${(s.siteId || 'SITE').toUpperCase()}-${out}`;
}
const hexBytes = h => new Uint8Array((h.match(/../g) || []).map(x => parseInt(x, 16)));

// ---------- the site dataset (IndexedDB, shared by every tool page of this site) ----------
const DB = 'sahvai-research', STORE = 'records';
function db() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE, { keyPath: 'record_id' });
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
}
async function tx(mode, fn) {
  const d = await db();
  return new Promise((res, rej) => { const t = d.transaction(STORE, mode), st = t.objectStore(STORE); const out = fn(st); t.oncomplete = () => res(out && out.result !== undefined ? out.result : out); t.onerror = () => rej(t.error); });
}
export const allRecords = async () => (await tx('readonly', st => st.getAll())) || [];
export const putRecord = rec => tx('readwrite', st => st.put(rec));
export const deleteRecord = id => tx('readwrite', st => st.delete(id));
export const clearRecords = () => tx('readwrite', st => st.clear());
export async function countRecords() { try { return (await tx('readonly', st => st.count())) || 0; } catch { return 0; } }
export function notifyChange() { try { new BroadcastChannel('sahvai-research').postMessage('changed'); } catch { /* ignore */ } }
export function onChange(fn) { try { new BroadcastChannel('sahvai-research').onmessage = fn; } catch { /* ignore */ } }

// ---------- building a record ----------
const r3 = x => x == null || !isFinite(x) ? null : +(+x).toFixed(3);
const num = x => x === '' || x == null || !isFinite(+x) ? null : +x;
export function dicomAge(a) {   // "065Y", "006M", "003W", "010D" -> whole years, capped at 90
  const m = /^(\d{1,3})\s*([YMWD])?$/i.exec(String(a || '').trim()); if (!m) return null;
  const n = +m[1], u = (m[2] || 'Y').toUpperCase(), y = u === 'Y' ? n : u === 'M' ? n / 12 : u === 'W' ? n / 52 : n / 365;
  return Math.min(90, Math.floor(y));
}
export const dicomSex = s => ({ M: '1', F: '2' })[String(s || '').trim().toUpperCase()] || (s ? '3' : null);
export function hoursBetween(onset, scan) {
  if (!onset || !scan) return null;
  const a = Date.parse(onset), b = Date.parse(scan); if (!isFinite(a) || !isFinite(b)) return null;
  return +((b - a) / 3.6e6).toFixed(1);
}
// ctx: {tool, info, compute, seconds, warnings, measurements: {field: value}, model, version, licence}; form: values from the box
export async function buildRecord(ctx, form) {
  const s = settings(), T = TOOLS[ctx.tool] || {}, i = ctx.info || {}, m = i.research || {};
  const method = form.studyIsCode ? 'entered' : 'hmac';
  const subject = form.studyIsCode ? String(form.studyId || '').trim() : await pseudonym(form.studyId, s);
  const fmt = ctx.inputFormat || (i.source === 'DICOM' ? 'dicom' : /demo/i.test(i.source || '') || ctx.demo ? 'demo' : /\.nrrd$/i.test(i.source || '') ? 'nrrd' : /\.nii/i.test(i.source || '') ? 'nifti' : null);
  const rec = {
    record_id: crypto.randomUUID(), schema_version: SCHEMA_VERSION, site_id: (s.siteId || '').toUpperCase() || null, site_country: (s.country || '').toUpperCase() || null,
    subject_id: subject || null, subject_id_method: subject ? method : null, scan_number: num(form.scanNumber), hours_from_onset: hoursBetween(form.onset, form.scanTime),
    record_created_month: new Date().toISOString().slice(0, 7),
    input_format: fmt, scanner_manufacturer: m.manufacturer || null, kvp: num(m.kvp), slice_thickness_mm: num(m.slice_thickness_mm) ?? (i.native_spacing_mm ? r3(i.native_spacing_mm[2]) : null),
    recon_kernel: m.recon_kernel || null, grid_spacing_mm: i.grid_spacing_mm ? i.grid_spacing_mm.map(v => +(+v).toFixed(3)).join(' x ') : null, skull_stripped: i.skull_stripped ? 1 : (i.grid_spacing_mm ? 0 : null),
    tool: ctx.tool, model: ctx.model || T.model || null, model_version: ctx.version || T.version || null, model_licence: ctx.licence || T.licence || null, app_version: APP_VERSION,
    compute: ctx.compute || null, runtime_s: r3(ctx.seconds), n_warnings: ctx.warnings ? ctx.warnings.length : 0,
  };
  for (const [k, v] of Object.entries(ctx.measurements || {})) if (FIELD[k]) rec[k] = typeof v === 'number' ? r3(v) : v;
  for (const k of CLINICAL) { const v = form[k], n = num(v); rec[k] = v === '' || v == null ? null : FIELD[k].type === 'category' ? String(v) : n == null ? null : Math.min(FIELD[k].max ?? Infinity, Math.round(n)); }
  for (const k of OUTCOMES) rec[k] = null;
  for (const f of FIELDS) if (!(f.name in rec)) rec[f.name] = null;
  return rec;
}

// ---------- writers ----------
const csvCell = v => v == null ? '' : /[",\n\r]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v);
export const toCSV = rows => [FIELDS.map(f => f.name).join(','), ...rows.map(r => FIELDS.map(f => csvCell(r[f.name])).join(','))].join('\r\n') + '\r\n';
export const toJSONL = rows => rows.map(r => JSON.stringify(Object.fromEntries(FIELDS.map(f => [f.name, r[f.name] ?? null])))).join('\n') + '\n';
export function dictionaryCSV() {
  const cols = ['variable', 'group', 'type', 'label', 'unit', 'choices', 'min', 'max', 'cde_concept', 'cde_source', 'note'];
  return [cols.join(','), ...FIELDS.map(f => [f.name, f.group, f.type, f.label, f.unit, (f.choices || (f.type === 'yesno' ? [['1', 'Yes'], ['0', 'No']] : [])).map(([c, l]) => `${c}=${l}`).join(' | '),
    f.min, f.max, f.cde, f.source, f.note].map(csvCell).join(','))].join('\r\n') + '\r\n';
}
// REDCap data dictionary (Online Designer upload format); the dataset CSV then imports as records
export function redcapDictionaryCSV() {
  const H = ['Variable / Field Name', 'Form Name', 'Section Header', 'Field Type', 'Field Label', 'Choices, Calculations, OR Slider Labels', 'Field Note', 'Text Validation Type OR Show Slider Number',
    'Text Validation Min', 'Text Validation Max', 'Identifier?', 'Branching Logic (Show field only if...)', 'Required Field?', 'Custom Alignment', 'Question Number (surveys only)', 'Matrix Group Name', 'Matrix Ranking?', 'Field Annotation'];
  const FORM = { admin: 'scan_record', imaging: 'scan_record', analysis: 'scan_record', volumes: 'hemorrhage_volumes', manual: 'hemorrhage_volumes', reference: 'hemorrhage_volumes', detection: 'hemorrhage_volumes', expansion: 'hemorrhage_volumes', clinical: 'clinical', outcome: 'outcomes' };
  const HEAD = { admin: 'Record and site', imaging: 'Imaging', analysis: 'Analysis', volumes: 'AI hemorrhage volumes', manual: 'Manual measurements', reference: 'Comparison with a reference', detection: 'HEADS-UP detection', expansion: 'ICH expansion signs', clinical: 'Clinical (optional)', outcome: 'Outcomes' };
  let last = '';
  const rows = FIELDS.map(f => {
    const ch = f.choices ? f.choices.map(([c, l]) => `${c}, ${l}`).join(' | ') : '';
    const typ = f.type === 'category' ? 'dropdown' : f.type === 'yesno' ? 'yesno' : 'text';
    const val = f.type === 'integer' ? 'integer' : f.type === 'number' ? 'number' : '';
    const sec = f.group !== last ? HEAD[f.group] : ''; last = f.group;
    const note = [f.unit, f.note, f.cde ? `CDE: ${f.cde}` : ''].filter(Boolean).join('; ');
    return [f.name, FORM[f.group], sec, typ, f.label, ch, note, val, f.min ?? '', f.max ?? '', '', '', f.name === 'record_id' ? 'y' : '', '', '', '', '', f.name === 'record_id' ? '' : ''].map(csvCell).join(',');
  });
  return [H.map(csvCell).join(','), ...rows].join('\r\n') + '\r\n';
}
export async function toParquet(rows) {
  const { parquetWriteBuffer } = await import('../vendor/hyparquet-writer/hyparquet-writer.min.mjs');
  const typ = f => f.type === 'integer' || f.type === 'yesno' ? 'INT32' : f.type === 'number' ? 'DOUBLE' : 'STRING';
  const conv = (f, v) => v == null || v === '' ? null : typ(f) === 'STRING' ? String(v) : typ(f) === 'INT32' ? Math.round(+v) : +v;
  return parquetWriteBuffer({ columnData: FIELDS.map(f => ({ name: f.name, type: typ(f), data: rows.map(r => conv(f, r[f.name])) })) });
}
// OMOP-style MEASUREMENT staging rows: the site maps concept ids and fills dates from its own records
export function toOMOP(rows) {
  const cols = ['person_source_value', 'visit_source_value', 'measurement_concept_id', 'measurement_source_value', 'value_as_number', 'value_source_value', 'unit_concept_id', 'unit_source_value', 'measurement_date', 'measurement_type_concept_id', 'measurement_source_concept_id'];
  const keep = FIELDS.filter(f => ['volumes', 'manual', 'reference', 'detection', 'expansion', 'clinical', 'outcome'].includes(f.group) && f.type !== 'text');
  const out = [cols.join(',')];
  for (const r of rows) for (const f of keep) {
    const v = r[f.name]; if (v == null || v === '') continue;
    out.push([r.subject_id, `${r.subject_id || ''}#scan${r.scan_number ?? ''}`, 0, `heads-up:${f.name}`, isFinite(+v) ? +v : '', v, f.unit === 'mL' ? 8587 : 0, f.unit || '', '', 0, 0].map(csvCell).join(','));
  }
  return out.join('\r\n') + '\r\n';
}
// FHIR R4 collection bundle: one Patient per pseudonym (no identifiers beyond it), one Device per model, Observations
export function toFHIR(rows) {
  const entries = [], seenP = new Set(), seenD = new Set(), pid = s => 'p-' + String(s || 'unknown').replace(/[^A-Za-z0-9.-]/g, '-');
  for (const r of rows) {
    const p = pid(r.subject_id);
    if (!seenP.has(p)) { seenP.add(p); entries.push({ fullUrl: `urn:uuid:${p}`, resource: { resourceType: 'Patient', id: p, identifier: [{ system: `${SITE_URL}fhir/site/${r.site_id || 'SITE'}`, value: r.subject_id || 'unknown' }], gender: { 1: 'male', 2: 'female', 3: 'unknown' }[r.sex] } }); }
    const d = `dev-${r.tool}-${r.model_version}`;
    if (!seenD.has(d)) { seenD.add(d); entries.push({ fullUrl: `urn:uuid:${d}`, resource: { resourceType: 'Device', id: d, deviceName: [{ name: TOOLS[r.tool]?.name || r.tool, type: 'user-friendly-name' }], version: [{ value: `${r.model_version} (app ${r.app_version})` }], note: [{ text: `${r.model}. Licence: ${r.model_licence}. Research use only.` }] } }); }
    for (const f of FIELDS) {
      if (!['volumes', 'manual', 'reference', 'detection', 'expansion', 'clinical', 'outcome'].includes(f.group)) continue;
      const v = r[f.name]; if (v == null || v === '') continue;
      const code = { coding: [{ system: CODE_SYSTEM, code: f.name, display: f.label }], text: f.label };
      const value = f.type === 'category' ? { valueCodeableConcept: { coding: [{ system: `${CODE_SYSTEM}/${f.name}`, code: String(v), display: (f.choices.find(c => c[0] === String(v)) || [])[1] }] } }
        : f.type === 'yesno' ? { valueBoolean: !!+v } : f.type === 'text' ? { valueString: String(v) } : f.unit ? { valueQuantity: { value: +v, unit: f.unit, system: 'http://unitsofmeasure.org', code: f.unit === 'h' ? 'h' : f.unit } } : f.type === 'integer' ? { valueInteger: +v } : { valueQuantity: { value: +v } };
      entries.push({ fullUrl: `urn:uuid:${r.record_id}-${f.name}`, resource: { resourceType: 'Observation', id: `${r.record_id}-${f.name}`.slice(0, 64), status: 'final', category: [{ text: f.group === 'clinical' || f.group === 'outcome' ? 'clinical' : 'imaging' }], code, subject: { reference: `urn:uuid:${p}` },
        ...(f.group !== 'clinical' && f.group !== 'outcome' ? { device: { reference: `urn:uuid:${d}` } } : {}), ...value,
        note: [{ text: `record ${r.record_id}; scan ${r.scan_number ?? '?'}${r.hours_from_onset != null ? `, ${r.hours_from_onset} h from onset` : ''}; site ${r.site_id || '?'}` }] } });
    }
  }
  return { resourceType: 'Bundle', type: 'collection', meta: { tag: [{ system: CODE_SYSTEM, code: SCHEMA_VERSION }] }, entry: entries };
}
// file names, series descriptions and exact timestamps can carry identifiers: never put them in an export
const SCRUB = new Set(['source', 'file', 'input', 'series_description', 'series_in_upload', 'series_selection', 'body_part', 'created', 'research', 'patientKey', 'name', 'date']);
const scrub = x => Array.isArray(x) ? x.map(scrub) : x && typeof x === 'object' ? Object.fromEntries(Object.entries(x).filter(([k]) => !SCRUB.has(k)).map(([k, v]) => [k, scrub(v)])) : x;
export function provenance(rec, ctx) {
  return {
    schema_version: SCHEMA_VERSION, app: 'SAHVAI Lab & HEADS-UP (CODE ICH) Hemorrhage Tools', app_url: SITE_URL, app_version: APP_VERSION,
    tool: rec.tool, tool_name: TOOLS[rec.tool]?.name, model: rec.model, model_version: rec.model_version, model_licence: rec.model_licence,
    compute: rec.compute, runtime_s: rec.runtime_s, browser: navigator.userAgent, exported: new Date().toISOString().slice(0, 7),
    analysis: scrub(ctx.provenance || {}), warnings: (ctx.warnings || []).map(w => String(w).replace(/\S+\.(nii(\.gz)?|nrrd|dcm|zip)\b/gi, '[file]')),
    privacy: 'Images and names never leave the site computer. The subject ID is a pseudonym (HMAC-SHA-256 with a key kept at the site) or a study code entered by the site; calendar dates are reduced to hours from onset; age is capped at 90. The label map has no patient metadata. Export and sharing are governed by the site\'s own approvals (IRB/ethics, data use agreements).',
    disclaimer: 'Research and education use only. Not a medical device; not for diagnosis or treatment decisions.',
    data_dictionary: 'data_dictionary.csv', standards: { cde: CDE_SAH, fhir: 'HL7 FHIR R4 (local code system)', omop: 'OMOP CDM MEASUREMENT staging (concept ids to be mapped by the site; 8587 = millilitre)' },
  };
}
export const README = `HEADS-UP research export (${SCHEMA_VERSION})
SAHVAI Lab & HEADS-UP (CODE ICH) Hemorrhage Tools - ${SITE_URL}

record.csv            one row: this scan's de-identified measurements (variables in data_dictionary.csv)
data_dictionary.csv   every variable: label, type, unit, codes, NINDS CDE concept
provenance.json       tool, model, licence, versions, compute and privacy notes
fhir_bundle.json      the same measurements as HL7 FHIR R4 Observations (local code system)
labels.nii.gz         the segmentation (label map) on the analysis grid, no patient metadata (when the tool segments)

Import record.csv (or the whole site dataset from the Research Dataset tab) into REDCap, NIH ScHARe (Terra),
R / Python, or a Convex table (JSON Lines). Research and education use only; not a medical device.
`;

// ---------- zip (store, no compression; the label map is already gzipped) ----------
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
const crc32 = b => { let c = 0xFFFFFFFF; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
export async function zip(files) {   // files: [{name, data: string | Blob | Uint8Array}]
  const enc = new TextEncoder(), parts = [], central = []; let off = 0;
  const d = new Date(), dt = ((d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1)) & 0xFFFF, dd = (((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate()) & 0xFFFF;
  for (const f of files) {
    const data = typeof f.data === 'string' ? enc.encode(f.data) : f.data instanceof Blob ? new Uint8Array(await f.data.arrayBuffer()) : f.data;
    const name = enc.encode(f.name), c = crc32(data);
    const h = new DataView(new ArrayBuffer(30));
    [[0, 0x04034b50, 4], [4, 20, 2], [6, 0x0800, 2], [8, 0, 2], [10, dt, 2], [12, dd, 2], [14, c, 4], [18, data.length, 4], [22, data.length, 4], [26, name.length, 2], [28, 0, 2]].forEach(([o, v, n]) => n === 4 ? h.setUint32(o, v, true) : h.setUint16(o, v, true));
    parts.push(new Uint8Array(h.buffer), name, data);
    const cd = new DataView(new ArrayBuffer(46));
    [[0, 0x02014b50, 4], [4, 20, 2], [6, 20, 2], [8, 0x0800, 2], [10, 0, 2], [12, dt, 2], [14, dd, 2], [16, c, 4], [20, data.length, 4], [24, data.length, 4], [28, name.length, 2], [30, 0, 2], [32, 0, 2], [34, 0, 2], [36, 0, 2], [38, 0, 4], [42, off, 4]].forEach(([o, v, n]) => n === 4 ? cd.setUint32(o, v, true) : cd.setUint16(o, v, true));
    central.push(new Uint8Array(cd.buffer), name);
    off += 30 + name.length + data.length;
  }
  const size = central.reduce((a, b) => a + b.length, 0), e = new DataView(new ArrayBuffer(22));
  [[0, 0x06054b50, 4], [4, 0, 2], [6, 0, 2], [8, files.length, 2], [10, files.length, 2], [12, size, 4], [16, off, 4], [20, 0, 2]].forEach(([o, v, n]) => n === 4 ? e.setUint32(o, v, true) : e.setUint16(o, v, true));
  return new Blob([...parts, ...central, new Uint8Array(e.buffer)], { type: 'application/zip' });
}

export function saveBlob(blob, name) { const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: name }); document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000); }
// the existing JSON reports leave out the research metadata (DICOM age, sex, dates)
export const publicInfo = i => { if (!i) return {}; const { research, ...rest } = i; return rest; };
// where the Research Dataset page is: the calculator tab when embedded, else the stand-alone page
export function datasetHref() {
  const p = location.pathname, k = p.indexOf('/sahvai-web/');
  if (k < 0) return '#research';   // the calculator page itself
  return window.parent !== window ? p.slice(0, k + 1) + '#research' : 'research.html';
}

// ---------- the box shown under a result ----------
const CSS = `
.rx{margin:14px 0 4px;border:1px solid #b9c9e6;border-radius:12px;background:#f7f9fd;font-size:.86rem}
.rx>summary{cursor:pointer;padding:10px 14px;font-weight:700;color:#00205b;list-style:none}
.rx>summary::-webkit-details-marker{display:none}
.rx>summary:before{content:'\\25B8';display:inline-block;margin-right:8px;transition:transform .15s}
.rx[open]>summary:before{transform:rotate(90deg)}
.rx>summary .sub{font-weight:400;color:#5f6673;margin-left:6px}
.rx .in{padding:0 14px 12px}
.rx .row{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:flex-end;margin:6px 0}
.rx label{display:flex;flex-direction:column;gap:2px;font-size:.76rem;color:#3a4250;font-weight:600}
.rx label.chk{flex-direction:row;align-items:center;gap:6px;font-weight:500}
.rx input,.rx select{font:inherit;font-size:.84rem;padding:4px 6px;border:1px solid #c3cad6;border-radius:6px;background:#fff;min-width:0}
.rx input[type=checkbox]{padding:0}
.rx .w4{width:4.2rem}.rx .w6{width:6.5rem}.rx .w9{width:9.5rem}.rx .w12{width:12rem}
.rx .pseudo{font-family:ui-monospace,Menlo,monospace;background:#eef2f9;border-radius:6px;padding:4px 8px;color:#00205b;align-self:flex-end}
.rx details.more{margin:6px 0}
.rx details.more>summary{cursor:pointer;color:#003da5;font-weight:600}
.rx .acts{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
.rx .rbtn{background:#003da5;color:#fff;border:1px solid #003da5;border-radius:8px;padding:6px 14px;font:inherit;font-size:.84rem;cursor:pointer}
.rx .rbtn.ghost{background:#fff;color:#003da5}
.rx .rbtn:disabled{opacity:.5;cursor:default}
.rx .msg{color:#1e6b34;font-weight:600}
.rx .msg.err{color:#a3001b}
.rx .fine{color:#5f6673;font-size:.76rem;margin:6px 0 0}
.rx .meas{color:#3a4250;margin:2px 0 6px}
`;
function injectCSS() { if (document.getElementById('rx-css')) return; const s = document.createElement('style'); s.id = 'rx-css'; s.textContent = CSS; document.head.appendChild(s); }
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const opt = (choices, v) => `<option value="">—</option>` + choices.map(([c, l]) => `<option value="${c}"${String(v ?? '') === c ? ' selected' : ''}>${esc(l)}</option>`).join('');
const rng = (a, b, v) => opt(Array.from({ length: b - a + 1 }, (_, k) => [String(a + k), String(a + k)]), v);
const localDT = iso => iso ? iso.slice(0, 16) : '';
const fmtMeas = m => [...Object.entries(m || {}).filter(([k, v]) => v != null && FIELD[k] && FIELD[k].unit === 'mL').map(([k, v]) => `${FIELD[k].label.replace(/ \(AI.*\)| volume.*$/i, '')} ${(+v).toFixed(1)} mL`),
  ...Object.entries(m || {}).filter(([k, v]) => v != null && /^det_.*_p$/.test(k)).map(([k, v]) => `P(${k.slice(4, -2)}) ${(+v).toFixed(2)}`)].join(' · ');

// mount(host, getCtx): getCtx() returns the ctx of the current result (or null). opts.prefill: {age_years, gcs_admission, scanNumber, scanTime, studyId}
export function mountResearch(host, getCtx, opts = {}) {
  injectCSS();
  const s = settings(), ctx0 = getCtx() || {}, m = (ctx0.info && ctx0.info.research) || {}, pf = opts.prefill || {};
  const age = pf.age_years ?? dicomAge(m.patient_age), sex = pf.sex ?? dicomSex(m.patient_sex);
  const el = document.createElement('details'); el.className = 'rx'; if (opts.open) el.open = true;
  el.formValues = () => ({ studyId: el.querySelector('.study').value, onset: el.querySelector('.onset').value, iscode: el.querySelector('.iscode').checked, open: el.open,
    clinical: Object.fromEntries(CLINICAL.map(k => [k, el.querySelector('.c-' + k).value])) });
  el.innerHTML = `<summary>&#128300; Research export <span class="sub">add this scan to your site's de-identified hemorrhage dataset, or download it</span></summary>
  <div class="in">
    <div class="meas">${esc(opts.label || '')}${opts.label ? ': ' : ''}${esc(fmtMeas(ctx0.measurements)) || 'measurements of this run'}</div>
    <div class="row">
      <label>Site code<input class="site w6" maxlength="12" placeholder="e.g. MCF" value="${esc(s.siteId || '')}"></label>
      <label>Country<input class="country w4" maxlength="2" placeholder="US" value="${esc(s.country || '')}"></label>
      <label title="Hashed into a pseudonym on this computer; the ID itself is never stored or exported (unless you tick the box for an already-coded study ID)">Study / subject ID (hashed here, never stored)<input class="study w12" placeholder="e.g. SAH-0012" value="${esc(pf.studyId || '')}" autocomplete="off"></label>
      <span class="pseudo" title="What is stored and exported instead of the study ID">—</span>
      <label class="chk"><input type="checkbox" class="iscode"> already a coded study ID (keep as typed)</label>
    </div>
    <div class="row">
      <label>Scan no.<input class="scanno w4" type="number" min="1" value="${esc(pf.scanNumber ?? 1)}"></label>
      <label>Onset (ictus / last known well)<input class="onset w12" type="datetime-local" value="${esc(localDT(pf.onset))}"></label>
      <label>Scan date and time<input class="scant w12" type="datetime-local" value="${esc(localDT(pf.scanTime || m.scan_datetime))}"></label>
      <span class="hrs fine" style="align-self:center">hours from onset: —</span>
    </div>
    <details class="more"><summary>Clinical variables (optional; NINDS CDE)</summary>
      <div class="row">
        <label>Age (y)<input class="c-age_years w4" type="number" min="0" max="120" value="${esc(age ?? '')}"></label>
        <label>Sex<select class="c-sex">${opt(SEX, sex)}</select></label>
        <label>Race and ethnicity<select class="c-race_ethnicity w12">${opt(RACE, '')}</select></label>
        <label>GCS<select class="c-gcs_admission">${rng(3, 15, pf.gcs_admission)}</select></label>
        <label>WFNS<select class="c-wfns_grade">${rng(1, 5)}</select></label>
        <label>Hunt &amp; Hess<select class="c-hunt_hess_grade">${rng(1, 5)}</select></label>
        <label>mod. Fisher<select class="c-modified_fisher">${rng(0, 4)}</select></label>
        <label>ICH score<select class="c-ich_score">${rng(0, 6)}</select></label>
      </div>
      <p class="fine">Outcomes (delayed cerebral ischemia, mRS at 90 days, in-hospital death) are added later on the Research Dataset tab. Age is stored as whole years, capped at 90.</p>
    </details>
    <div class="acts">
      <button class="rbtn add" type="button">Add to site dataset</button>
      <button class="rbtn ghost dl" type="button">Download this scan (.zip)</button>
      <span class="msg"></span>
      <span class="fine count"></span>
    </div>
    <p class="fine">Nothing leaves this computer unless you download it. The study ID becomes a pseudonym with a key kept in this browser (back it up on the Research Dataset tab; use the same key on every computer at your site); dates become hours from onset; images and names are never exported. Use under your site's approvals.</p>
  </div>`;
  host.appendChild(el);
  if (pf.iscode) el.querySelector('.iscode').checked = true;
  for (const [k, v] of Object.entries(pf.clinical || {})) if (v !== '' && v != null) el.querySelector('.c-' + k).value = v;
  const q = c => el.querySelector(c), msg = (t, err) => { q('.msg').textContent = t; q('.msg').classList.toggle('err', !!err); };
  const form = () => ({ studyId: q('.study').value, studyIsCode: q('.iscode').checked, scanNumber: q('.scanno').value, onset: q('.onset').value, scanTime: q('.scant').value,
    ...Object.fromEntries(CLINICAL.map(k => [k, q('.c-' + k).value])) });
  const keep = () => { const st = settings(); st.siteId = q('.site').value.trim().toUpperCase(); st.country = q('.country').value.trim().toUpperCase(); saveSettings(st); };
  const refresh = async () => {
    keep();
    const f = form(); q('.pseudo').textContent = !f.studyId.trim() ? '—' : f.studyIsCode ? f.studyId.trim() : await pseudonym(f.studyId);
    const h = hoursBetween(f.onset, f.scanTime); q('.hrs').textContent = `hours from onset: ${h == null ? '—' : h}${h != null && h < 0 ? ' (check: scan before onset)' : ''}`;
  };
  const count = async () => { const n = await countRecords(); q('.count').innerHTML = `Site dataset in this browser: <b>${n}</b> scan${n === 1 ? '' : 's'} · <a href="${datasetHref()}" target="${window.parent !== window ? '_top' : '_self'}">open Research Dataset</a>`; };
  el.addEventListener('input', refresh); el.addEventListener('change', refresh); refresh(); count(); onChange(count);
  const record = async () => {
    const ctx = getCtx(); if (!ctx) { msg('Run a scan first.', true); return null; }
    const f = form(); if (!f.studyId.trim()) { msg('Enter a study ID (it is turned into a pseudonym).', true); q('.study').focus(); return null; }
    if (!settings().siteId) { msg('Enter your site code.', true); q('.site').focus(); return null; }
    return { rec: await buildRecord(ctx, f), ctx };
  };
  q('.add').onclick = async () => {
    const r = await record(); if (!r) return;
    try { await putRecord(r.rec); notifyChange(); msg(`Added: ${r.rec.subject_id}, scan ${r.rec.scan_number ?? '?'}, ${TOOLS[r.rec.tool]?.name}.`); count(); } catch (e) { msg(`Could not save in this browser (${e.message}). Use “Download this scan”.`, true); }
  };
  q('.dl').onclick = async () => {
    const r = await record(); if (!r) return;
    const files = [{ name: 'record.csv', data: toCSV([r.rec]) }, { name: 'data_dictionary.csv', data: dictionaryCSV() }, { name: 'provenance.json', data: JSON.stringify(provenance(r.rec, r.ctx), null, 2) },
      { name: 'fhir_bundle.json', data: JSON.stringify(toFHIR([r.rec]), null, 2) }, { name: 'README.txt', data: README }];
    if (r.ctx.labelMap) { try { files.push({ name: 'labels.nii.gz', data: await r.ctx.labelMap() }); } catch { /* label map optional */ } }
    const nm = `${r.rec.subject_id}_scan${r.rec.scan_number ?? 'x'}_${r.rec.tool}.zip`.replace(/[^A-Za-z0-9._-]/g, '_');
    saveBlob(await zip(files), nm); msg(`Downloaded ${nm}.`);
  };
  return el;
}
