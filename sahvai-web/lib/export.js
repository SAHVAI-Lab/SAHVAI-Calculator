// NIfTI-1 writer (.nii.gz) for the working grid. The grid is stored L-P-S (x towards Left,
// y towards Posterior); NIfTI uses RAS, so the first two rows of the affine are negated.

export async function niftiGz(data, g, { datatype = 2, description = '' } = {}) {
  const { nx, ny, nz, sp } = g;
  const bytesPer = { 2: 1, 4: 2, 16: 4 }[datatype];
  const hdr = new ArrayBuffer(352), dv = new DataView(hdr);
  dv.setInt32(0, 348, true);
  [3, nx, ny, nz, 1, 1, 1, 1].forEach((v, i) => dv.setInt16(40 + 2 * i, v, true));
  dv.setInt16(70, datatype, true); dv.setInt16(72, bytesPer * 8, true);
  [1, sp[0], sp[1], sp[2], 0, 0, 0, 0].forEach((v, i) => dv.setFloat32(76 + 4 * i, v, true));
  dv.setFloat32(108, 352, true);            // vox_offset
  dv.setFloat32(112, 1, true);              // scl_slope
  dv.setUint8(123, 2);                      // xyzt_units: mm
  const desc = new TextEncoder().encode(description.slice(0, 79));
  new Uint8Array(hdr, 148, 80).set(desc);
  const A = g.affine || [[sp[0], 0, 0, 0], [0, sp[1], 0, 0], [0, 0, sp[2], 0]];
  const ras = [[-A[0][0], -A[0][1], -A[0][2], -A[0][3]], [-A[1][0], -A[1][1], -A[1][2], -A[1][3]], A[2]];
  dv.setInt16(252, 0, true); dv.setInt16(254, 1, true);   // qform_code 0, sform_code 1 (scanner)
  ras.forEach((row, r) => row.forEach((v, c) => dv.setFloat32(280 + 16 * r + 4 * c, v, true)));
  new Uint8Array(hdr, 344, 4).set([0x6e, 0x2b, 0x31, 0]);   // "n+1\0"
  const body = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const blob = new Blob([hdr, body]);
  return new Response(blob.stream().pipeThrough(new CompressionStream('gzip'))).blob();
}

export function toInt16(hu) {
  const out = new Int16Array(hu.length);
  for (let i = 0; i < hu.length; i++) out[i] = Math.max(-32768, Math.min(32767, Math.round(hu[i])));
  return out;
}
