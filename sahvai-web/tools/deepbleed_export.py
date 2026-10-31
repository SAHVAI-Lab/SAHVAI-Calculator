"""DeepBleed VNet (Sharrock et al., Neuroinformatics 2021; github.com/msharrock/deepbleed, CC BY-NC-ND 4.0)
re-expressed in PyTorch with the published MISTIE-2 weights, checked against a TensorFlow reference built from the
same checkpoint, exported to ONNX (float32, weights unchanged: a format conversion only).
Usage: python3 export_vnet.py /path/to/weights out.onnx   (weights = the TF checkpoint prefix)"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import tensorflow as tf
CK, OUT = sys.argv[1], sys.argv[2]
r = tf.train.load_checkpoint(CK)
W = lambda n: r.get_tensor(f'{n}/.ATTRIBUTES/VARIABLE_VALUE')

# ---- TensorFlow reference with the checkpoint arrays (NDHWC), mirroring blocks/vnet.py (norm=False, drop=False)
def tconv(x, n, stride=1, padding='SAME', act=None):
    y = tf.nn.conv3d(x, W(n + '/kernel'), [1, stride, stride, stride, 1], padding) + W(n + '/bias')
    return tf.nn.relu(y) if act == 'relu' else y
def tup(x, n):
    k = W(n + '/kernel'); s = tf.shape(x)
    y = tf.nn.conv3d_transpose(x, k, [s[0], s[1] * 2, s[2] * 2, s[3] * 2, k.shape[3]], [1, 2, 2, 2, 1], 'VALID') + W(n + '/bias')
    return y
def tf_vnet(x):
    x16 = tconv(x, 'input_layer/convolution', act='relu') + tf.concat(16 * [x], -1)
    def down(x, name, n):
        d = tf.nn.relu(tconv(x, name + '/downsample', 2, 'VALID')); y = d
        for _ in range(n): y = tf.nn.relu(tconv(y, name + '/convolution'))
        return y + d
    def up(x, skip, name, n):
        u = tf.nn.relu(tup(x, name + '/upsample')); cat = tf.concat([u, skip], -1); y = cat
        for _ in range(n): y = tf.nn.relu(tconv(y, name + '/convolution'))
        return y + cat
    x32 = down(x16, 'down_1', 2); x64 = down(x32, 'down_2', 3); x128 = down(x64, 'down_3', 3); x256 = down(x128, 'down_4', 3)
    y = up(x256, x128, 'up_4', 3); y = up(y, x64, 'up_3', 3); y = up(y, x32, 'up_2', 2); y = up(y, x16, 'up_1', 1)
    y = tconv(y, 'outblock/final', act='relu')
    return tf.sigmoid(tconv(y, 'outblock/binary'))

# ---- PyTorch (NCDHW). Each block's 5x5x5 convolution is ONE layer applied n times (shared weights), as in the original.
class Down(nn.Module):
    def __init__(s, cin, c, n): super().__init__(); s.ds = nn.Conv3d(cin, c, 2, 2); s.conv = nn.Conv3d(c, c, 5, padding=2); s.n = n
    def forward(s, x):
        d = F.relu(s.ds(x)); y = d
        for _ in range(s.n): y = F.relu(s.conv(y))
        return y + d
class Up(nn.Module):
    def __init__(s, cin, c, n): super().__init__(); s.us = nn.ConvTranspose3d(cin, c // 2, 2, 2); s.conv = nn.Conv3d(c, c, 5, padding=2); s.n = n
    def forward(s, x, skip):
        cat = torch.cat([F.relu(s.us(x)), skip], 1); y = cat
        for _ in range(s.n): y = F.relu(s.conv(y))
        return y + cat
class VNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.inp = nn.Conv3d(1, 16, 5, padding=2)
        s.d1, s.d2, s.d3, s.d4 = Down(16, 32, 2), Down(32, 64, 3), Down(64, 128, 3), Down(128, 256, 3)
        s.u4, s.u3, s.u2, s.u1 = Up(256, 256, 3), Up(256, 128, 3), Up(128, 64, 2), Up(64, 32, 1)
        s.final = nn.Conv3d(32, 2, 1); s.binary = nn.Conv3d(2, 1, 1)
    def forward(s, x):
        x16 = F.relu(s.inp(x)) + x.expand(-1, 16, -1, -1, -1)
        x32 = s.d1(x16); x64 = s.d2(x32); x128 = s.d3(x64); x256 = s.d4(x128)
        y = s.u4(x256, x128); y = s.u3(y, x64); y = s.u2(y, x32); y = s.u1(y, x16)
        return torch.sigmoid(s.binary(F.relu(s.final(y))))

net = VNet().eval()
conv = lambda m, n: (m.weight.data.copy_(torch.from_numpy(W(n + '/kernel').transpose(4, 3, 0, 1, 2).copy())), m.bias.data.copy_(torch.from_numpy(W(n + '/bias'))))
tconvw = lambda m, n: (m.weight.data.copy_(torch.from_numpy(W(n + '/kernel').transpose(4, 3, 0, 1, 2).copy())), m.bias.data.copy_(torch.from_numpy(W(n + '/bias'))))
conv(net.inp, 'input_layer/convolution')
for i, b in enumerate([net.d1, net.d2, net.d3, net.d4], 1): conv(b.ds, f'down_{i}/downsample'); conv(b.conv, f'down_{i}/convolution')
for i, b in zip([4, 3, 2, 1], [net.u4, net.u3, net.u2, net.u1]): tconvw(b.us, f'up_{i}/upsample'); conv(b.conv, f'up_{i}/convolution')
conv(net.final, 'outblock/final'); conv(net.binary, 'outblock/binary')
print('params', sum(p.numel() for p in net.parameters()))

# test input: a brain-like volume (HU 0-80 blob, a bright 60 HU 'bleed', zeros outside) at the network size
N = int(sys.argv[3]) if len(sys.argv) > 3 else 128
g = np.mgrid[:N, :N, :N].astype(np.float32) / N - 0.5
brain = (g[0] ** 2 / 0.16 + g[1] ** 2 / 0.12 + g[2] ** 2 / 0.14 < 1).astype(np.float32)
rng = np.random.default_rng(0)
x = brain * (30 + 5 * rng.standard_normal(brain.shape).astype(np.float32))
x[(g[0] - 0.1) ** 2 + (g[1] + 0.05) ** 2 + (g[2] - 0.05) ** 2 < 0.006] = 65
ref = tf_vnet(tf.constant(x[None, ..., None])).numpy()[0, ..., 0]
with torch.no_grad(): mine = net(torch.from_numpy(x)[None, None]).numpy()[0, 0]
print(f'torch vs TF reference: max |diff| {np.abs(mine - ref).max():.2e}, mask mismatches {((mine > .5) != (ref > .5)).sum()}, TF mask voxels {(ref > .5).sum()}')
torch.onnx.export(net, (torch.from_numpy(x)[None, None],), OUT, input_names=['x'], output_names=['prob'], opset_version=17, dynamo=False)
import onnxruntime as ort
y = ort.InferenceSession(OUT, providers=['CPUExecutionProvider']).run(None, {'x': x[None, None]})[0][0, 0]
print(f'ONNX Runtime vs TF reference: max |diff| {np.abs(y - ref).max():.2e}, mask mismatches {((y > .5) != (ref > .5)).sum()}')
np.save(OUT + '.testx.npy', x); np.save(OUT + '.testref.npy', ref)
