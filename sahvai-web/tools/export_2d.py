# Rewrites BrainHemoAI's TriHybridUNet (hybrid branch only, auxiliary decoders
# unused at inference) with 2D ops only, slices as the batch axis, then exports ONNX.
#   3D conv k=(3,kh,kw)  -> one 2D conv producing 3*Cout channels + shifted sum over slices
#   InstanceNorm3d        -> mean/var over (slices, H, W) per channel
#   BatchNorm2d           -> folded into the preceding conv
#   MaxPool3d/ConvT3d (1,2,2) -> 2D equivalents
# Both branches then share the layout (D, C, H, W), so the 2D<->3D rearranges vanish.
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, 'BrainHemoAI')
from ml.tri_hybrid_unet import TriHybridUNet

sd = torch.load('BrainHemoAI/ml/pth/tri_hybrid_unet.pth', map_location='cpu')
orig = TriHybridUNet(5, 4, 32, auxiliary=False); orig.load_state_dict(sd); orig.eval()


def fold_bn(w, p):
    g, b, m, v = (sd[p + s] for s in ('.weight', '.bias', '.running_mean', '.running_var'))
    k = g / torch.sqrt(v + 1e-5)
    return w * k.view(-1, *[1] * (w.dim() - 1)), b - m * k


class C2(nn.Module):           # conv2d (+ folded BN)
    def __init__(self, w, b, pad):
        super().__init__(); self.w = nn.Parameter(w); self.b = nn.Parameter(b) if b is not None else None; self.pad = pad
    def forward(self, x): return F.conv2d(x, self.w, self.b, padding=self.pad)


class C3(nn.Module):           # 3D conv with depth kernel 3 as a single 2D conv + slice shift
    def __init__(self, w3, pad_hw):
        super().__init__()
        co = w3.shape[0]; self.co = co; self.pad = pad_hw
        self.w = nn.Parameter(torch.cat([w3[:, :, 0], w3[:, :, 1], w3[:, :, 2]], 0))
    def forward(self, x):
        y = F.conv2d(x, self.w, None, padding=self.pad)
        co = self.co
        y0, y1, y2 = y[:, :co], y[:, co:2 * co], y[:, 2 * co:]
        # out[d] = y0[d-1] + y1[d] + y2[d+1]  (zero outside the volume)
        z = y1[:1] * 0
        return y1 + torch.cat([z, y0[:-1]], 0) + torch.cat([y2[1:], z], 0)


def inorm(x):                  # InstanceNorm3d(affine=False) in (D,C,H,W) layout
    m = x.mean(dim=(0, 2, 3), keepdim=True)
    d = x - m
    v = (d * d).mean(dim=(0, 2, 3), keepdim=True)
    return d / torch.sqrt(v + 1e-5)


class DC2(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.c1 = C2(*fold_bn(sd[p + '.double_conv.0.weight'], p + '.double_conv.1'), 1)
        self.c2 = C2(*fold_bn(sd[p + '.double_conv.3.weight'], p + '.double_conv.4'), 1)
    def forward(self, x): return F.relu(self.c2(F.relu(self.c1(x))))


class DC3(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.c1 = C3(sd[p + '.double_conv.0.weight'], 1); self.c2 = C3(sd[p + '.double_conv.3.weight'], 1)
    def forward(self, x): return F.relu(inorm(self.c2(F.relu(inorm(self.c1(x))))))


def sa2(w, x):
    y = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)
    return torch.sigmoid(w(y)) * x


class Down(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.c2 = DC2(p + '.hybird_net.conv2d'); self.c3 = DC3(p + '.hybird_net.conv3d')
        self.a2 = C2(sd[p + '.s_atten2d.conv.weight'], None, 2)
        self.a3 = C3(sd[p + '.s_atten3d.conv.weight'], 2)
    def forward(self, a2, a3):
        y2 = self.c2(torch.cat([a2, a3], 1)); y3 = self.c3(torch.cat([a3, a2], 1))
        return F.max_pool2d(sa2(self.a2, y2), 2), F.max_pool2d(sa2(self.a3, y3), 2)


class Up(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.w = nn.Parameter(sd[p + '.up.weight'][:, :, 0]); self.b = nn.Parameter(sd[p + '.up.bias'])
        self.c = DC3(p + '.conv')
    def forward(self, a2, a3, h):
        return self.c(F.conv_transpose2d(torch.cat([a2, a3, h], 1), self.w, self.b, stride=2))


class Net2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc2d = DC2('inc2d'); self.inc3d = DC3('inc3d')
        self.d1, self.d2, self.d3 = Down('d1'), Down('d2'), Down('d3')
        self.hc = DC3('hybrid_concat.conv3d')
        self.u1, self.u2, self.u3 = Up('u_hybrid_1'), Up('u_hybrid_2'), Up('u_hybrid_3')
        self.out = C2(sd['outc_hybrid.weight'][:, :, 0], sd['outc_hybrid.bias'], 0)
    def forward(self, x):          # x: (D, 5, H, W)
        a2, a3 = self.inc2d(x), self.inc3d(x)
        b2, b3 = self.d1(a2, a3); c2, c3 = self.d2(b2, b3); e2, e3 = self.d3(c2, c3)
        h = self.hc(torch.cat([e3, e2], 1))
        h = self.u1(e2, e3, h); h = self.u2(c2, c3, h); h = self.u3(b2, b3, h)
        return self.out(h)         # (D, 4, H, W) logits


if __name__ == '__main__':
    torch.set_num_threads(2)
    net = Net2D().eval()
    print('params', sum(p.numel() for p in net.parameters()))
    ref = np.load('ref.npz')
    x = torch.tensor(ref['x'])     # (5, 18, H, W)
    with torch.no_grad():
        t = time.time(); o_orig = orig(x.permute(1, 0, 2, 3), x.unsqueeze(0))[0][0].permute(1, 0, 2, 3); print('orig', time.time() - t)
        t = time.time(); o_new = net(x.permute(1, 0, 2, 3)); print('2d', time.time() - t)
    print('max abs diff', (o_orig - o_new).abs().max().item(), 'logit range', o_orig.abs().max().item())
    print('argmax agree', (o_orig.argmax(1) == o_new.argmax(1)).float().mean().item(),
          'mismatch voxels', (o_orig.argmax(1) != o_new.argmax(1)).sum().item())
    torch.onnx.export(net, (x.permute(1, 0, 2, 3),), 'sahvai_fp32.onnx', input_names=['x'], output_names=['logits'],
                      opset_version=17, dynamo=False)
    print('exported')
