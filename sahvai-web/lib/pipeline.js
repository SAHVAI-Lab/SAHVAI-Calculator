// Working grid -> brain mask -> alignment -> UNet (18-slice windows) -> labels on the grid -> volumes.
import { extractBrain, rectify } from './brain.js';
import { H, W, DEPTH, estimateAngle, prepare, windowsFor, windowInput, toGridLabels, postprocess, volumes } from './hemo.js';
import { runUnet } from './unet.js';

export async function analyse(g, { call, log = () => {}, stage = () => {}, angle: angleIn = null, mask: maskIn = null, slab = 4 } = {}) {
  const { nx, ny, nz, hu, sp } = g, P = nx * ny, t0 = performance.now();
  stage('Extracting the brain', 0);
  const mask = maskIn ? rectify(hu, maskIn, nx, ny, nz) : extractBrain(g, log, f => stage('Extracting the brain', f));
  // a crude contrast check: enhancing vessels put many voxels above 150 HU inside the skull
  let inside = 0, hot = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) inside++;
  const warnings = [];
  if (inside * sp[0] * sp[1] * sp[2] / 1000 < 300) warnings.push('Very little brain was found; check that this is a head CT and that the brain outline (QC view) looks right.');
  const angle = angleIn ?? estimateAngle(mask, nx, ny, nz);
  log(`alignment: rotate ${angle} degrees${angleIn === null ? ' (estimated from left-right symmetry)' : ' (set by hand)'}`);
  stage('Preparing the network input', 0);
  const prep = prepare(hu, mask, nx, ny, nz, angle);
  const starts = windowsFor(prep.s, prep.e);
  log(`brain box ${prep.ch}x${prep.cw} px from (${prep.top}, ${prep.left}); slices ${prep.s}-${prep.e} of 0-${nz - 1} used, in ${starts.length} window(s) of ${DEPTH}`);
  const HW = H * W, labels = new Uint8Array(nz * HW), covered = new Uint8Array(nz);
  for (let k = 0; k < starts.length; k++) {
    const z0 = starts[k], { x, nreal } = windowInput(prep.x2, z0, prep.e);
    const lab = await runUnet(x, DEPTH, H, W, call, { slab, progress: (f, name) => stage(`Segmenting (window ${k + 1} of ${starts.length})`, (k + f) / starts.length, name) });
    for (let d = 0; d < nreal; d++) { const z = z0 + d; if (!covered[z]) { labels.set(lab.subarray(d * HW, (d + 1) * HW), z * HW); covered[z] = 1; } }
  }
  stage('Post-processing', 0);
  const raw = toGridLabels(labels, prep, nx, ny, nz);
  const lab = postprocess(raw, nx, ny, nz);
  const vol = volumes(lab, sp);
  log(`volumes: SAH ${vol.SAH.toFixed(1)} mL, IPH ${vol.IPH.toFixed(1)} mL, IVH ${vol.IVH.toFixed(1)} mL (analysis ${((performance.now() - t0) / 1000).toFixed(1)}s)`);
  return { mask, angle, slices: [prep.s, prep.e], windows: starts, raw, labels: lab, volumes: vol, rawVolumes: volumes(raw, sp), warnings };
}
