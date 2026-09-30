// NRRD (.nrrd / .seg.nrrd, attached data, raw or gzip) as a volume in the same form as
// series.js returns: dims, spacing, dir[a] = LPS direction of axis a, slice(k) -> Float32 along axis 2.

export async function nrrdVolume(file, log = () => {}) {
  const buf = new Uint8Array(await file.arrayBuffer());
  // header ends at the first blank line
  let end = -1;
  for (let i = 0; i + 1 < Math.min(buf.length, 1 << 16); i++) {
    if (buf[i] === 10 && buf[i + 1] === 10) { end = i + 2; break; }
    if (buf[i] === 13 && buf[i + 1] === 10 && buf[i + 2] === 13 && buf[i + 3] === 10) { end = i + 4; break; }
  }
  if (end < 0) throw new Error('not an NRRD file (no header)');
  const text = new TextDecoder().decode(buf.subarray(0, end));
  if (!/^NRRD\d{4}/.test(text)) throw new Error('not an NRRD file');
  const h = {};
  for (const line of text.split(/\r?\n/).slice(1)) {
    if (!line || line.startsWith('#')) continue;
    const k = line.indexOf(':'); if (k < 0) continue;
    h[line.slice(0, k).trim().toLowerCase()] = line.slice(k + 1).replace(/^=/, '').trim();
  }
  if (h['data file'] || h.datafile) throw new Error('detached NRRD (separate data file) is not supported; save as a single .nrrd');
  const dim = +h.dimension;
  const sizes = h.sizes.split(/\s+/).map(Number);
  if (dim !== 3) throw new Error(`need a 3D NRRD, got dimension ${dim}`);
  const enc = (h.encoding || 'raw').toLowerCase();
  let data = buf.subarray(end);
  if (enc === 'gzip' || enc === 'gz') {
    data = new Uint8Array(await new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  } else if (enc !== 'raw') throw new Error(`NRRD encoding "${enc}" is not supported`);
  const t = (h.type || '').toLowerCase();
  const T = /^(u?char|u?int8|signed char|unsigned char|uint8_t|int8_t)$/.test(t) ? (/^(uchar|unsigned char|uint8|uint8_t)$/.test(t) ? Uint8Array : Int8Array)
    : /short|int16/.test(t) ? (/unsigned|ushort|uint16/.test(t) ? Uint16Array : Int16Array)
    : /^(int|signed int|int32|int32_t|uint|unsigned int|uint32|uint32_t)$/.test(t) ? (/^u|unsigned/.test(t) ? Uint32Array : Int32Array)
    : /float/.test(t) ? Float32Array : /double/.test(t) ? Float64Array : null;
  if (!T) throw new Error(`NRRD type "${h.type}" is not supported`);
  if (T.BYTES_PER_ELEMENT > 1 && (h.endian || 'little') !== 'little') throw new Error('big-endian NRRD is not supported');
  const n = sizes[0] * sizes[1] * sizes[2];
  const arr = new T(data.buffer.slice(data.byteOffset, data.byteOffset + n * T.BYTES_PER_ELEMENT));
  // geometry
  const space = (h.space || 'left-posterior-superior').toLowerCase();
  const ras = /right-anterior|^ras$/.test(space);
  let dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], spacing = [1, 1, 1];
  if (h['space directions']) {
    const vs = [...h['space directions'].matchAll(/\(([^)]*)\)/g)].map(m => m[1].split(',').map(Number));
    if (vs.length === 3) {
      spacing = vs.map(v => Math.hypot(...v) || 1);
      dirs = vs.map((v, i) => v.map(x => x / spacing[i]));
    }
  } else if (h.spacings) spacing = h.spacings.split(/\s+/).map(Number).map(x => x > 0 ? x : 1);
  let origin = h['space origin'] ? h['space origin'].replace(/[()]/g, '').split(',').map(Number) : null;
  if (ras) { dirs = dirs.map(v => [-v[0], -v[1], v[2]]); if (origin) origin = [-origin[0], -origin[1], origin[2]]; }
  const plane = sizes[0] * sizes[1];
  log(`NRRD ${sizes.join('x')}, spacing ${spacing.map(s => s.toFixed(3)).join(', ')} mm, type ${h.type}, ${enc}`);
  return {
    dims: sizes, spacing, dir: dirs, origin,
    async slice(k) {
      const out = new Float32Array(plane), s = arr.subarray(k * plane, (k + 1) * plane);
      for (let i = 0; i < plane; i++) out[i] = s[i];
      return out;
    },
  };
}
