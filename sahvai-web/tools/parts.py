# Split the 2D-only TriHybridUNet into parts that keep browser memory low.
# Full-resolution stages run on slabs of n slices plus a 1-slice halo each side;
# every 3D InstanceNorm there needs whole-volume statistics, so each part returns
# per-slice channel sums (s) and sums of squares (q) of its raw conv output and the
# caller normalises in the next part. The 1/2-resolution-and-below middle (d2..u2)
# runs on the whole volume in one call.
#
# Slab inputs with a halo have n+2 slices; `v` (n+2,1,1,1) is 1 for slices inside
# the volume and 0 for padding, so out-of-volume activations are zero exactly as the
# zero padding of the original 3D convolutions.
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, 'tools')
import export as E

net = E.Net2D().eval()


def c3v(c3, x):
    """'valid' 3D conv along slices: n+2 input slices -> n output slices."""
    co = c3.co; w = c3.w
    w0, w1, w2 = w[:co], w[co:2 * co], w[2 * co:]
    p = c3.pad
    return F.conv2d(x[:-2], w0, None, padding=p) + F.conv2d(x[1:-1], w1, None, padding=p) + F.conv2d(x[2:], w2, None, padding=p)


def nrm(r, m, i): return F.relu((r - m) * i)
def sums(r): return r.sum(dim=(2, 3)), (r * r).sum(dim=(2, 3))
def dc2(m, x): return m(x)
def sa(conv_fn, y):
    return torch.cat([y.mean(1, keepdim=True), y.max(1, keepdim=True)[0]], 1)


class Part(nn.Module):
    def __init__(self):
        super().__init__(); self.net = net


class P1(Part):   # x(n+2,5) -> a2(n,32), r1(n,32)
    def forward(self, x):
        a2 = net.inc2d(x[1:-1]); r1 = c3v(net.inc3d.c1, x)
        return (a2, r1, *sums(r1))

class P2(Part):   # r1(n+2) -> r2(n)
    def forward(self, r, v, m, i):
        r2 = c3v(net.inc3d.c2, nrm(r, m, i) * v)
        return (r2, *sums(r2))

class P3(Part):   # r2(n+2), a2(n+2) -> b2(n,64,H/2,W/2), r3(n,64)
    def forward(self, r, a2, v, m, i):
        a3 = nrm(r, m, i) * v
        d = net.d1
        y2 = d.c2(torch.cat([a2[1:-1], a3[1:-1]], 1))
        y2 = torch.sigmoid(d.a2(sa(None, y2))) * y2
        b2 = F.max_pool2d(y2, 2)
        r3 = c3v(d.c3.c1, torch.cat([a3, a2], 1))
        return (b2, r3, *sums(r3))

class P4(Part):   # r3(n+2) -> r4(n)
    def forward(self, r, v, m, i):
        r4 = c3v(net.d1.c3.c2, nrm(r, m, i) * v)
        return (r4, *sums(r4))

class P5(Part):   # r4(n+2) -> b3(n,64,H/2,W/2)
    def forward(self, r, v, m, i):
        y3 = nrm(r, m, i) * v
        att = torch.sigmoid(c3v(net.d1.a3, sa(None, y3)))
        return F.max_pool2d(att * y3[1:-1], 2)

class PM(Part):   # whole volume at 1/2 resolution: b2, b3 -> h (64ch)
    def forward(self, b2, b3):
        c2, c3 = net.d2(b2, b3); e2, e3 = net.d3(c2, c3)
        h = net.hc(torch.cat([e3, e2], 1))
        h = net.u1(e2, e3, h); h = net.u2(c2, c3, h)
        return h

class P6(Part):   # b2,b3,h (n+2, 1/2 res) -> r5(n,32)
    def forward(self, b2, b3, h, v):
        u = net.u3
        x = F.conv_transpose2d(torch.cat([b2, b3, h], 1), u.w, u.b, stride=2) * v
        r5 = c3v(u.c.c1, x)
        return (r5, *sums(r5))

class P7(Part):   # r5(n+2) -> r6(n)
    def forward(self, r, v, m, i):
        r6 = c3v(net.u3.c.c2, nrm(r, m, i) * v)
        return (r6, *sums(r6))

class P8(Part):   # r6(n) -> logits(n,4)
    def forward(self, r, m, i):
        return net.out(nrm(r, m, i))


def stats(s, q, N):
    S = s.sum(0, dtype=np.float64); Q = q.sum(0, dtype=np.float64)
    mean = S / N; var = Q / N - mean * mean
    return mean.astype(np.float32).reshape(1, -1, 1, 1), (1 / np.sqrt(var + 1e-5)).astype(np.float32).reshape(1, -1, 1, 1)


def run_parts(x, call, n=4):
    """x: (D,5,H,W). call(name, inputs dict)->list of outputs. Mirrors the JS orchestration."""
    D = x.shape[0]; N = D * x.shape[2] * x.shape[3]
    def halo(a, s, e):   # slices s-1..e (inclusive e) with zeros outside the volume
        shp = (e - s + 2,) + a.shape[1:]
        out = np.zeros(shp, np.float32); lo, hi = max(s - 1, 0), min(e + 1, D)
        out[lo - (s - 1): hi - (s - 1)] = a[lo:hi]; return out
    def valid(s, e):
        v = np.array([1.0 if 0 <= k < D else 0.0 for k in range(s - 1, e + 1)], np.float32); return v.reshape(-1, 1, 1, 1)
    slabs = [(s, min(s + n, D)) for s in range(0, D, n)]
    def loop(name, make, nout):
        outs = [[] for _ in range(nout)]
        for s, e in slabs:
            r = call(name, make(s, e))
            for k in range(nout): outs[k].append(r[k])
        return [np.concatenate(o, 0) for o in outs]
    a2, r1, s, q = loop('p1', lambda s, e: {'x': halo(x, s, e)}, 4); m, i = stats(s, q, N)
    r2, s, q = loop('p2', lambda s, e: {'r': halo(r1, s, e), 'v': valid(s, e), 'm': m, 'i': i}, 3); m, i = stats(s, q, N)
    b2, r3, s, q = loop('p3', lambda s, e: {'r': halo(r2, s, e), 'a2': halo(a2, s, e), 'v': valid(s, e), 'm': m, 'i': i}, 4); m, i = stats(s, q, N)
    r4, s, q = loop('p4', lambda s, e: {'r': halo(r3, s, e), 'v': valid(s, e), 'm': m, 'i': i}, 3); m, i = stats(s, q, N)
    (b3,) = loop('p5', lambda s, e: {'r': halo(r4, s, e), 'v': valid(s, e), 'm': m, 'i': i}, 1)
    (h,) = call('pm', {'b2': b2, 'b3': b3})[:1]
    r5, s, q = loop('p6', lambda s, e: {'b2': halo(b2, s, e), 'b3': halo(b3, s, e), 'h': halo(h, s, e), 'v': valid(s, e)}, 3); m, i = stats(s, q, N)
    r6, s, q = loop('p7', lambda s, e: {'r': halo(r5, s, e), 'v': valid(s, e), 'm': m, 'i': i}, 3); m, i = stats(s, q, N)
    (lg,) = loop('p8', lambda s, e: {'r': r6[s:e], 'm': m, 'i': i}, 1)
    return lg


SPEC = {  # name: (module, input names, output names)
    'p1': (P1(), ['x'], ['a2', 'r', 's', 'q']),
    'p2': (P2(), ['r', 'v', 'm', 'i'], ['r2', 's', 'q']),
    'p3': (P3(), ['r', 'a2', 'v', 'm', 'i'], ['b2', 'r3', 's', 'q']),
    'p4': (P4(), ['r', 'v', 'm', 'i'], ['r4', 's', 'q']),
    'p5': (P5(), ['r', 'v', 'm', 'i'], ['b3']),
    'pm': (PM(), ['b2', 'b3'], ['h']),
    'p6': (P6(), ['b2', 'b3', 'h', 'v'], ['r5', 's', 'q']),
    'p7': (P7(), ['r', 'v', 'm', 'i'], ['r6', 's', 'q']),
    'p8': (P8(), ['r', 'm', 'i'], ['logits']),
}


def torch_call(name, inp):
    mod, ins, _ = SPEC[name]
    with torch.no_grad():
        out = mod(*[torch.tensor(inp[k]) for k in ins])
    if not isinstance(out, tuple): out = (out,)
    return [o.numpy() for o in out]


def example_inputs(name, n=2, H=32, W=32):
    C = {'p2': 32, 'p4': 64, 'p5': 64, 'p7': 32, 'p8': 32}
    r = lambda *s: torch.randn(*s)
    if name == 'p1': return (r(n + 2, 5, H, W),)
    if name == 'p3': return (r(n + 2, 32, H, W), r(n + 2, 32, H, W), r(n + 2, 1, 1, 1), r(1, 32, 1, 1), r(1, 32, 1, 1))
    if name == 'pm': return (r(n, 64, H // 2, W // 2), r(n, 64, H // 2, W // 2))
    if name == 'p6': return (r(n + 2, 64, H // 2, W // 2), r(n + 2, 64, H // 2, W // 2), r(n + 2, 64, H // 2, W // 2), r(n + 2, 1, 1, 1))
    c = C[name]
    if name == 'p8': return (r(n, c, H, W), r(1, c, 1, 1), r(1, c, 1, 1))
    return (r(n + 2, c, H, W), r(n + 2, 1, 1, 1), r(1, c, 1, 1), r(1, c, 1, 1))


if __name__ == '__main__':
    import onnxruntime as ort, subprocess, os
    torch.set_num_threads(2)
    mode = sys.argv[1]
    if mode == 'torchcheck':   # orchestrated torch vs reference logits
        ref = np.load('ref.npz')
        x = np.ascontiguousarray(ref['x'].transpose(1, 0, 2, 3))
        L = ref['logits'].astype(np.float32).transpose(1, 0, 2, 3)
        lg = run_parts(x, torch_call, n=4)
        print('maxdiff', np.abs(lg - L).max(), 'agree', (lg.argmax(1) == L.argmax(1)).mean(), 'mismatch', int((lg.argmax(1) != L.argmax(1)).sum()))
    if mode == 'export':
        os.makedirs('models', exist_ok=True)
        for name, (mod, ins, outs) in SPEC.items():
            dyn = {k: {0: 'n', 2: 'h', 3: 'w'} for k in ins if k not in ('v', 'm', 'i')}
            if 'v' in ins: dyn['v'] = {0: 'n'}
            # every output gets its own symbolic dims: halo inputs have n+2 slices and some outputs are at half resolution
            for k in outs: dyn[k] = {0: f'{k}_n', 2: f'{k}_h', 3: f'{k}_w'} if k not in ('s', 'q') else {0: f'{k}_n'}
            f = f'models/{name}.raw.onnx'
            torch.onnx.export(mod, example_inputs(name), f, input_names=ins, output_names=outs, opset_version=17, dynamo=False, dynamic_axes=dyn)
            import onnx
            mm = onnx.load(f); del mm.graph.value_info[:]; onnx.save(mm, f)   # drop exporter shape guesses
            o = ort.SessionOptions(); o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            o.optimized_model_filepath = f.replace('.raw', '.opt')
            ort.InferenceSession(f, o, providers=['CPUExecutionProvider'])
            subprocess.run(['python3', 'tools/to_fp16.py', o.optimized_model_filepath, f'models/{name}.onnx'], check=True)
            os.remove(f); os.remove(o.optimized_model_filepath)
        print('exported')
