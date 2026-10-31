# SAH Swin UNETR (García-García, Cepeda et al.; github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction,
# academic non-commercial licence) -> models/sahswin/sahswin.onnx.{0,1,2} for the browser.
# Needs: torch, monai (1.6), onnx, onnxruntime; the checkpoint SWIN_UNETR_48_100_best_metric_model.pth in ./w/.
# Steps: (1) export one 128x128x64 window to ONNX (weights unchanged, fp32), with window attention in chunks of 64 windows
# (identical arithmetic; keeps the browser within 4 GB) and the 2x2x2 transposed convolutions as 1x1x1 convolutions plus an
# interleave; (2) fold constants with ONNX Runtime (basic level); (3) replace the folded shifted-window masks (0 / -100, up to
# 235 MB) by a few nodes that compute them from per-window region ids, tied to the input so they are not folded back;
# (4) merge duplicated constants; (5) split into three parts.
# python sahswin_export.py   (writes swin.onnx, swin_b.onnx, swin_m.onnx and the parts in ./out/)

# ---- (1) export ----
# Export the García-García/Cepeda SAH Swin UNETR (weights unchanged, fp32) to ONNX for one 128x128x64 window.
# ConvTranspose3d(k=2,s=2) is rewritten as a 1x1x1 conv to 8C channels + interleave (ORT WebGPU lacks 3D ConvTranspose);
# the result is mathematically identical.
import torch, torch.nn as nn, numpy as np, onnx
from monai.networks.nets import SwinUNETR
class CT2(nn.Module):
    def __init__(s, ct):
        super().__init__(); Ci, Co = ct.in_channels, ct.out_channels
        w = ct.weight.data  # (Ci, Co, 2,2,2)
        s.conv = nn.Conv3d(Ci, Co*8, 1, bias=ct.bias is not None)
        # out channel index = co*8 + (a*4+b*2+c)
        s.conv.weight.data = w.permute(1, 2, 3, 4, 0).reshape(Co*8, Ci, 1, 1, 1).contiguous()
        if ct.bias is not None: s.conv.bias.data = ct.bias.data.repeat_interleave(8)
        s.Co = Co
    def forward(s, x):
        B, _, D, H, W = x.shape
        y = s.conv(x).reshape(B, s.Co, 2, 2, 2, D, H, W).permute(0, 1, 5, 2, 6, 3, 7, 4)
        return y.reshape(B, s.Co, D*2, H*2, W*2)
# Window attention in chunks of windows: identical arithmetic, but the (windows x heads x 343 x 343) attention
# matrices of the first stage (about 700 MB each for a 128x128x64 window) are formed a chunk at a time, which keeps
# ONNX Runtime Web inside the 4 GB WebAssembly memory.
from monai.networks.nets import swin_unetr as SU
_wa_forward = SU.WindowAttention.forward
CHUNK = 64
def _chunked(self, x, mask):
    b = x.shape[0]
    if b <= CHUNK: return _wa_forward(self, x, mask)
    assert mask is None or mask.shape[0] == b   # batch size 1
    outs = [_wa_forward(self, x[s:s + CHUNK], None if mask is None else mask[s:s + CHUNK]) for s in range(0, b, CHUNK)]
    return torch.cat(outs, 0)
SU.WindowAttention.forward = _chunked
def swap(m):
    for n, c in m.named_children():
        if isinstance(c, nn.ConvTranspose3d) and c.kernel_size == (2,2,2) and c.stride == (2,2,2) and c.padding == (0,0,0): setattr(m, n, CT2(c))
        else: swap(c)
model = SwinUNETR(in_channels=1, out_channels=2, feature_size=48, use_checkpoint=False, norm_name='batch')
model.load_state_dict(torch.load('w/SWIN_UNETR_48_100_best_metric_model.pth', map_location='cpu', weights_only=True)); model.eval()
x = torch.randn(1, 1, 128, 128, 64)
with torch.no_grad(): ref = model(x)
print('ref done', flush=True)
swap(model)
n = sum(isinstance(c, CT2) for c in model.modules()); print('swapped', n)
with torch.no_grad(): alt = model(x)
print('rewrite max diff', (ref-alt).abs().max().item())
print('exporting', flush=True)
with torch.no_grad(): torch.onnx.export(model, x, 'swin.onnx', input_names=['x'], output_names=['y'], opset_version=17, dynamo=False, do_constant_folding=True)
np.save('swin_testx.npy', x.numpy()); np.save('swin_testref.npy', ref.numpy())
import onnxruntime as ort
s = ort.InferenceSession('swin.onnx', providers=['CPUExecutionProvider'])
y = s.run(None, {'x': x.numpy()})[0]
print('onnx vs torch max diff', np.abs(y-ref.numpy()).max(), 'argmax disagreements', int((y.argmax(1) != ref.numpy().argmax(1)).sum()))
m = onnx.load('swin.onnx'); print('ops', sorted(set(nd.op_type for nd in m.graph.node)))

# ---- (2) constant folding ----
import onnxruntime as ort
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC; so.optimized_model_filepath = 'swin_b.onnx'
ort.InferenceSession('swin.onnx', so, providers=['CPUExecutionProvider'])

# ---- (3)-(4) masks and duplicates ----
# Replace the constant-folded shifted-window attention masks (0 / -100, up to 235 MB) with a tiny runtime
# computation from per-window region ids: mask[w,i,j] = 0 if id[w,i]==id[w,j] else -100 (exactly what MONAI computes).
import onnx, numpy as np
from onnx import numpy_helper as nh, helper as oh
m = onnx.load('swin_b.onnx'); g = m.graph
inits = {i.name: i for i in g.initializer}
k = 0
for name in list(inits):
    i = inits[name]; dims = tuple(i.dims)
    if len(dims) == 5 and dims[0] == 1 and dims[2] == 1 and dims[3] == dims[4] == 343:
        M = nh.to_array(i)[0, :, 0]                      # (nW,343,343)
        same = M == 0
        ids = np.argmax(same, axis=2).astype(np.int32)   # first token in the same region
        R = np.where(ids[:, :, None] == ids[:, None, :], 0, -100).astype(np.float32)
        assert np.array_equal(R, M), name
        p = f'mask{k}_'; k += 1
        g.initializer.remove(i)
        g.initializer.extend([nh.from_array(ids, p+'ids0'), nh.from_array(np.array([2], np.int64), p+'a2'), nh.from_array(np.array([1], np.int64), p+'a1'),
                              nh.from_array(np.array(0, np.float32), p+'z'), nh.from_array(np.array(-100, np.float32), p+'n'),
                              nh.from_array(np.array(dims, np.int64), p+'shape')])
        # ids + 0 * x[0,0,0,0,0]: ties the masks to the input so that ONNX Runtime does not constant-fold them back
        # into 300 MB of tensors when the session is created
        nodes = [oh.make_node('Add', [p+'ids0', 'xzero_i32'], [p+'idsr']),
                 oh.make_node('Unsqueeze', [p+'idsr', p+'a2'], [p+'A']), oh.make_node('Unsqueeze', [p+'idsr', p+'a1'], [p+'B']),
                 oh.make_node('Equal', [p+'A', p+'B'], [p+'E']), oh.make_node('Where', [p+'E', p+'z', p+'n'], [p+'W']),
                 oh.make_node('Reshape', [p+'W', p+'shape'], [name])]
        for j, nd in enumerate(nodes): g.node.insert(j, nd)
        print('replaced', name, dims)
g.initializer.extend([nh.from_array(np.zeros(5, np.int64), 'xz_s'), nh.from_array(np.ones(5, np.int64), 'xz_e'), nh.from_array(np.zeros(0, np.int64), 'xz_shape'), nh.from_array(np.array(0, np.float32), 'xz_0')])
pre = [oh.make_node('Slice', ['x', 'xz_s', 'xz_e'], ['xz1']), oh.make_node('Reshape', ['xz1', 'xz_shape'], ['xz2']),
       oh.make_node('Mul', ['xz2', 'xz_0'], ['xz3']), oh.make_node('Cast', ['xz3'], ['xzero_i32'], to=onnx.TensorProto.INT32)]
for j, nd in enumerate(pre): g.node.insert(j, nd)
onnx.checker.check_model(m)
onnx.save(m, 'swin_m.onnx')
import onnxruntime as ort, time
x = np.load('swin_testx.npy'); r = np.load('swin_testref.npy')
s = ort.InferenceSession('swin_m.onnx', providers=['CPUExecutionProvider']); t = time.time()
y = s.run(None, {'x': x})[0]
print('t %.1f' % (time.time()-t), 'max diff', np.abs(y-r).max(), 'argmax disagreements', int((y.argmax(1) != r.argmax(1)).sum()))
# merge identical constant tensors (the relative-position biases repeated for each chunk of windows)
import hashlib
m = onnx.load('swin_m.onnx'); g = m.graph; seen = {}; rename = {}
for i in list(g.initializer):
    h = hashlib.sha1(i.raw_data + str(i.data_type).encode() + str(list(i.dims)).encode()).hexdigest() if i.raw_data else None
    if h is None: continue
    if h in seen: rename[i.name] = seen[h]; g.initializer.remove(i)
    else: seen[h] = i.name
for nd in g.node:
    for k, nm in enumerate(nd.input):
        if nm in rename: nd.input[k] = rename[nm]
print('merged', len(rename), 'duplicate constants')
onnx.checker.check_model(m); onnx.save(m, 'swin_m.onnx')
s = ort.InferenceSession('swin_m.onnx', providers=['CPUExecutionProvider'])
y = s.run(None, {'x': x})[0]
print('after merge: max diff', np.abs(y-r).max(), 'argmax disagreements', int((y.argmax(1) != r.argmax(1)).sum()))

# ---- (5) parts ----
import os
os.makedirs('out', exist_ok=True)
b = open('swin_m.onnx', 'rb').read(); sz = -(-len(b) // 3)
for i in range(3): open(f'out/sahswin.onnx.{i}', 'wb').write(b[i * sz:(i + 1) * sz])
print('parts written; total', len(b), 'bytes')

# ---- the registration template: the repository's ct_template2mni.nii.gz (1 mm), smoothed (sigma 0.85 voxel), every 2nd voxel ----
import nibabel as nib
from scipy.ndimage import gaussian_filter
t = nib.load('ct_template2mni.nii.gz'); d = t.get_fdata().astype(np.float32)
s2 = gaussian_filter(d, 0.85)[::2, ::2, ::2]; A = t.affine.copy(); A[:3, :3] *= 2
img = nib.Nifti1Image(np.round(s2 * 10).astype(np.int16), A); img.header['scl_slope'] = 0.1; img.header['scl_inter'] = 0; img.set_qform(A, 1); img.set_sform(A, 1)
nib.save(img, 'out/ct_template2mni_2mm.nii.gz')
