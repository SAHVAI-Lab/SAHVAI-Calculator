// Self-test of lib/hesigns.js on synthetic phantoms with known signs (node tools/hesigns_selftest.mjs).
// Each phantom: a brain (35 HU, noise SD 3 HU) in a 0.5 x 0.5 x 5 mm grid, a 70 HU haematoma labelled IPH,
// and one injected sign. The test checks that the injected sign is found and that a plain haematoma has none.
import { analyseSigns } from '../lib/hesigns.js';

const nx = 256, ny = 256, nz = 12, P = nx * ny, sp = [0.5, 0.5, 5];
let seed = 7; const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());

function phantom(paint) {
  const hu = new Float32Array(P * nz), labels = new Uint8Array(P * nz), mask = new Uint8Array(P * nz);
  for (let k = 0; k < nz; k++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const p = k * P + y * nx + x, inBrain = ((x - 128) / 110) ** 2 + ((y - 128) / 120) ** 2 <= 1;
    hu[p] = inBrain ? 35 : -1000; mask[p] = inBrain ? 1 : 0;
  }
  // haematoma: ellipsoid centred (128, 128, slice 6), 40 x 30 mm in plane, 4 slices
  const inClot = (x, y, k) => k >= 4 && k <= 7 && ((x - 128) / 40) ** 2 + ((y - 128) / 30) ** 2 <= 1 - (k === 4 || k === 7 ? 0.35 : 0);
  for (let k = 0; k < nz; k++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) if (inClot(x, y, k)) { hu[k * P + y * nx + x] = 70; labels[k * P + y * nx + x] = 2; }
  paint({ hu, labels, set: (x, y, k, v, l = 2) => { const p = k * P + y * nx + x; hu[p] = v; if (l != null) labels[p] = l; }, inClot });
  for (let p = 0; p < P * nz; p++) if (mask[p]) hu[p] += 3 * gauss();
  return { g: { nx, ny, nz, sp, hu }, labels, raw: labels, mask };
}
const disk = (cx, cy, r, f) => { for (let y = Math.floor(cy - r); y <= cy + r; y++) for (let x = Math.floor(cx - r); x <= cx + r; x++) if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) f(x, y); };

const cases = {
  plain: { paint: () => {}, expect: {} },
  hypodensity: { paint: ({ set }) => { for (const k of [5, 6]) disk(128, 128, 10, (x, y) => set(x, y, k, 52)); },   // 5 mm across, 18 HU below
    expect: { hypodensity: true, blackhole: false } },
  blackhole: { paint: ({ set }) => { for (const k of [5, 6]) disk(125, 130, 9, (x, y) => set(x, y, k, 38)); },        // 32 HU below, iso to brain
    expect: { hypodensity: true, blackhole: true, swirl: true } },
  blend: { paint: ({ set, inClot }) => { for (const k of [5, 6]) for (let y = 0; y < ny; y++) for (let x = 150; x < nx; x++) if (inClot(x, y, k)) set(x, y, k, 48); },
    expect: { blend: true, hypodensity: false } },
  fluid: { paint: ({ set, inClot }) => { for (const k of [5, 6]) for (let y = 0; y < 122; y++) for (let x = 0; x < nx; x++) if (inClot(x, y, k)) set(x, y, k, 46); },
    expect: { fluid: true } },
  satellite: { paint: ({ set }) => { disk(128, 78, 5, (x, y) => set(x, y, 6, 68)); },   // 5 mm focus, 15 mm in front of the clot
    expect: { satellite: true, island: false } },
  island: { paint: ({ set }) => { for (const [cx, cy] of [[128, 84], [80, 110], [176, 110]]) disk(cx, cy, 4, (x, y) => set(x, y, 6, 68)); },
    expect: { island: true, satellite: true } },
  irregular: { paint: ({ set }) => { for (const k of [4, 5, 6, 7]) for (const [cx, cy] of [[128, 92], [168, 128], [92, 150]]) disk(cx, cy, 9, (x, y) => set(x, y, k, 70)); },
    expect: { irregular: true } },
  spot: { paint: ({ hu, set }) => {
      for (let k = 2; k < 10; k++) for (let i = 0; i < 40; i++) { const x = 40 + Math.floor(rnd() * 176), y = 40 + Math.floor(rnd() * 176); if (hu[k * P + y * nx + x] === 35) disk(x, y, 3, (a, b) => { if (hu[k * P + b * nx + a] === 35) set(a, b, k, 250, null); }); }
      disk(128, 128, 4, (x, y) => set(x, y, 6, 190)); },
    expect: { spot: true } },
};

let fail = 0;
for (const [name, c] of Object.entries(cases)) {
  const { g, labels, raw, mask } = phantom(c.paint), r = analyseSigns(g, labels, raw, mask);
  const found = Object.entries(r.signs).filter(([, s]) => s.auto).map(([k]) => k);
  const errs = [];
  for (const [k, v] of Object.entries(c.expect)) if (!!r.signs[k].auto !== v) errs.push(`${k} ${r.signs[k].auto} (expected ${v})`);
  if (name === 'plain' && found.length) errs.push(`unexpected: ${found.join(', ')}`);
  console.log(`${errs.length ? 'FAIL' : 'ok  '} ${name.padEnd(12)} found: ${found.join(', ') || 'none'}${errs.length ? '  <- ' + errs.join('; ') : ''}`);
  if (errs.length) { fail++; for (const k of Object.keys(c.expect)) console.log(`      ${k}: ${r.signs[k].detail}`); }
}
console.log(fail ? `${fail} phantom(s) failed` : 'all phantoms passed');
process.exit(fail ? 1 : 0);
