// Skull-stripped input (e.g. the RHUH dataset: brain only, background exactly 0, no air or bone).
// The three models expect a whole head in air, so such a volume is wrapped: background -> -1000 HU (air),
// and SAHVAI is given the brain mask (the non-background voxels) instead of finding it from the skull.
import { toGrid } from './grid.js';

export async function detectStripped(vol) {
  const n = vol.dims[2], ks = [0.3, 0.5, 0.7].map(f => Math.min(n - 1, Math.floor(f * n)));
  let tot = 0, zero = 0, air = 0, bone = 0;
  for (const k of ks) {
    const s = await vol.slice(k);
    for (let i = 0; i < s.length; i++) { const v = s[i]; tot++; if (v === 0) zero++; else if (v < -200) air++; else if (v > 300) bone++; }
  }
  return zero / tot > 0.3 && air / tot < 0.01 && bone / tot < 0.01;
}

export function airBackground(vol) {
  return { ...vol, async slice(k) { const s = await vol.slice(k), o = new Float32Array(s.length); for (let i = 0; i < s.length; i++) o[i] = s[i] === 0 ? -1000 : s[i]; return o; } };
}

// brain mask on the grid that `gridOpts` produces (the wrapped volume's non-air voxels)
export async function brainMaskFor(vol, gridOpts = {}) {
  const g = await toGrid(vol, () => {}, () => {}, { ...gridOpts, interp: 'auto', label: undefined, pre: v => (v > -999 ? 1 : 0) });
  const m = new Uint8Array(g.hu.length);
  for (let i = 0; i < m.length; i++) m[i] = g.hu[i] > 0.5 ? 1 : 0;
  return m;
}
