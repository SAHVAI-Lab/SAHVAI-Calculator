// Reusable axial viewer for one panel: CT slice with label overlay, optional outlines
// (brain mask, expert label), window presets, slice slider, wheel/keys and a per-slice chart.
// classes: { id: { key, name, color: [r,g,b] } }

const WINDOWS = [['Brain', 40, 80], ['Blood', 60, 120], ['Subdural', 75, 215], ['Bone', 600, 2800]];

export class Viewer {
  constructor(root, classes, { onSlice = () => {} } = {}) {
    this.root = root; this.classes = classes; this.onSlice = onSlice; this.R = null; this.z = 0;
    const id = 'v' + Math.random().toString(36).slice(2, 8);
    root.innerHTML = `
      <div class="canvaswrap"><canvas class="view" aria-label="Axial CT slice with segmentation overlay"></canvas>
        <span class="ori t">A</span><span class="ori b">P</span><span class="ori l">R</span><span class="ori r">L</span>
        <span class="sliceinfo"></span></div>
      <div class="side">
        <label class="row">Slice <input class="slice" type="range" min="0" max="0" value="0" aria-label="Slice"></label>
        <div class="row presets" role="radiogroup" aria-label="Window">${WINDOWS.map(([n, c, w], i) =>
          `<label><input type="radio" name="${id}-win" value="${c},${w}"${i ? '' : ' checked'}> ${n}</label>`).join('')}</div>
        <label class="row"><input class="overlay" type="checkbox" checked> Segmentation overlay</label>
        <label class="row">Opacity <input class="opacity" type="range" min="0" max="100" value="55" aria-label="Overlay opacity"></label>
        <label class="row outline-row" hidden><input class="outline" type="checkbox"> Brain outline (quality check)</label>
        <label class="row gt-row" hidden><input class="gtline" type="checkbox" checked> Expert label outline</label>
        <div class="legend">${Object.values(classes).map(c => `<span><i style="background:rgb(${c.color.join(',')})"></i>${c.key}</span>`).join('')}<span class="gt-legend" hidden><i class="gt-swatch"></i>Expert</span></div>
        <p class="tip">Scroll over the image or use ↑/↓ to change slice. Click the chart to jump.</p>
        <canvas class="chart" height="120" aria-label="Segmented volume per slice"></canvas>
        <div class="dl"></div>
      </div>`;
    const q = s => root.querySelector(s);
    this.cv = q('.view'); this.chartCv = q('.chart'); this.info = q('.sliceinfo'); this.slider = q('.slice');
    this.ov = q('.overlay'); this.op = q('.opacity'); this.outl = q('.outline'); this.gtl = q('.gtline'); this.dl = q('.dl');
    this.winName = `${id}-win`;
    this.slider.addEventListener('input', e => this.setSlice(+e.target.value));
    root.querySelectorAll(`input[name=${id}-win], .overlay, .opacity, .outline, .gtline`).forEach(el => el.addEventListener('input', () => this.render()));
    q('.canvaswrap').addEventListener('wheel', e => { if (!this.R) return; e.preventDefault(); this.setSlice(this.z + (e.deltaY > 0 ? -1 : 1)); }, { passive: false });
    this.cv.tabIndex = 0;
    this.cv.addEventListener('keydown', e => { if (e.key === 'ArrowUp') { this.setSlice(this.z + 1); e.preventDefault(); } if (e.key === 'ArrowDown') { this.setSlice(this.z - 1); e.preventDefault(); } });
    this.chartCv.addEventListener('click', e => { if (!this.R) return; const r = e.target.getBoundingClientRect(); this.setSlice(Math.floor((e.clientX - r.left) / r.width * this.R.g.nz)); });
    window.addEventListener('resize', () => this.chart());
  }

  static edges(mask, g) {
    const P = g.nx * g.ny, e = new Uint8Array(P * g.nz);
    for (let k = 0; k < g.nz; k++) for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) {
      const p = k * P + y * g.nx + x; if (!mask[p]) continue;
      if (x === 0 || y === 0 || x === g.nx - 1 || y === g.ny - 1 || !mask[p - 1] || !mask[p + 1] || !mask[p - g.nx] || !mask[p + g.nx]) e[p] = 1;
    }
    return e;
  }

  // g: {nx, ny, nz, sp, hu}; labels Uint8; opts: {mask, gt, range: [z0, z1]}
  show(g, labels, { mask = null, gt = null, range = null } = {}) {
    const P = g.nx * g.ny, ids = Object.keys(this.classes).map(Number), maxId = Math.max(...ids);
    const per = Array.from({ length: g.nz }, () => new Array(maxId + 1).fill(0));
    for (let k = 0; k < g.nz; k++) for (let p = 0; p < P; p++) per[k][labels[k * P + p]]++;
    this.R = { g, labels, per, vox: g.sp[0] * g.sp[1] * g.sp[2] / 1000, range: range || [0, g.nz - 1],
      edge: mask ? Viewer.edges(mask, g) : null, gtEdge: gt ? Viewer.edges(gt, g) : null };
    this.root.querySelector('.outline-row').hidden = !mask;
    this.root.querySelector('.gt-row').hidden = !gt; this.root.querySelector('.gt-legend').hidden = !gt;
    this.cv.width = g.nx; this.cv.height = g.ny; this.cv.style.aspectRatio = `${g.nx * g.sp[0]} / ${g.ny * g.sp[1]}`;
    this.slider.max = g.nz - 1;
    let best = Math.floor(g.nz / 2), bv = 0;
    per.forEach((c, k) => { let b = 0; for (const i of ids) b += c[i]; if (gt) for (let p = 0; p < P; p++) b += gt[k * P + p] ? 0.5 : 0; if (b > bv) { bv = b; best = k; } });
    this.setSlice(best);
  }

  setSlice(k) { if (!this.R) return; this.z = Math.max(0, Math.min(this.R.g.nz - 1, k)); this.slider.value = this.z; this.render(); this.chart(); this.onSlice(this.z); }

  render() {
    const R = this.R; if (!R) return;
    const g = R.g, P = g.nx * g.ny, o = this.z * P, ctx = this.cv.getContext('2d');
    const [c, w] = this.root.querySelector(`input[name=${this.winName}]:checked`).value.split(',').map(Number), lo = c - w / 2;
    const ov = this.ov.checked, a = this.op.value / 100, outl = R.edge && this.outl.checked, gtl = R.gtEdge && this.gtl.checked;
    const img = ctx.createImageData(g.nx, g.ny), d = img.data;
    for (let p = 0; p < P; p++) {
      let v = (g.hu[o + p] - lo) / w * 255; v = v < 0 ? 0 : v > 255 ? 255 : v;
      let r = v, gg = v, b = v;
      const l = R.labels[o + p], cl = l && this.classes[l];
      if (ov && cl) { const col = cl.color; r = r * (1 - a) + col[0] * a; gg = gg * (1 - a) + col[1] * a; b = b * (1 - a) + col[2] * a; }
      if (outl && R.edge[o + p]) { r = 60; gg = 230; b = 90; }
      if (gtl && R.gtEdge[o + p]) { r = 255; gg = 255; b = 255; }
      const q = p * 4; d[q] = r; d[q + 1] = gg; d[q + 2] = b; d[q + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const s = R.per[this.z], inR = this.z >= R.range[0] && this.z <= R.range[1];
    this.info.textContent = [`slice ${this.z + 1}/${g.nz}${inR ? '' : ' (not analysed)'}`,
      ...Object.entries(this.classes).filter(([i]) => s[i]).map(([i, c]) => `${c.key} ${(s[i] * R.vox).toFixed(2)} mL`)].join(' · ');
  }

  chart() {
    const R = this.R; if (!R) return;
    const cv = this.chartCv, dpr = window.devicePixelRatio || 1, Wc = cv.clientWidth, Hc = cv.clientHeight;
    if (!Wc) return;
    cv.width = Wc * dpr; cv.height = Hc * dpr;
    const ctx = cv.getContext('2d'); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, Wc, Hc);
    const css = getComputedStyle(this.root);
    const ids = Object.keys(this.classes).map(Number);
    const n = R.g.nz, bw = Wc / n, max = Math.max(1, ...R.per.map(c => ids.reduce((t, i) => t + c[i], 0)));
    ctx.fillStyle = css.getPropertyValue('--band').trim() || '#eef2f9';
    ctx.fillRect(R.range[0] * bw, 0, (R.range[1] - R.range[0] + 1) * bw, Hc);
    R.per.forEach((c, k) => {
      let y = Hc - 14;
      for (const t of ids) {
        const h = c[t] / max * (Hc - 22); if (!h) continue;
        ctx.fillStyle = `rgb(${this.classes[t].color.join(',')})`; ctx.fillRect(k * bw + 1, y - h, Math.max(1, bw - 2), h); y -= h;
      }
    });
    ctx.strokeStyle = css.getPropertyValue('--mayo-blue').trim() || '#003da5'; ctx.lineWidth = 2; ctx.strokeRect(this.z * bw + 0.5, 1, bw - 1, Hc - 2);
    ctx.fillStyle = css.getPropertyValue('--muted').trim() || '#5f6673'; ctx.font = '10px system-ui';
    ctx.fillText('inferior', 3, Hc - 3); ctx.textAlign = 'right'; ctx.fillText('superior', Wc - 3, Hc - 3);
  }

  png(name) { this.cv.toBlob(b => save(b, name)); }
}

export function save(blob, name) {
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: name });
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
