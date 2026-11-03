# Rebuild models/yale/yale_seg.onnx and yale_cls.onnx from the public checkpoints of
# github.com/anhtrnyaleedu/HE (Tran et al., npj Digit Med 2024;7:26).
#   git clone --depth 1 https://github.com/anhtrnyaleedu/HE /tmp/HE
#   python3 tools/yale_export.py /tmp/HE models/yale
# yale_seg.onnx: MONAI SegResNet, x[1,1,256,256,32] -> logit (one sliding window).
# yale_cls.onnx: MyDenseNet121 (two passes through the trunk: slices 0-95 and 12-75), x[1,2,128,128,96], mc[1] -> logit.
#   nn.Dropout3d (p = 0.1) is replaced by an in-graph random channel mask scaled by 1/(1-p), used only when mc = 1,
#   so Monte-Carlo dropout runs in the browser. Weights are unchanged; initializers are stored as float16 with a
#   Cast to float32 (arithmetic stays float32): logits change by < 0.003 on the demo cases.
# The checkpoints carry no licence: ask the authors before publishing the ONNX files.
import sys, os, subprocess, torch, torch.nn as nn
src, dst = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
from monai.networks.nets import SegResNet
from MyDenseNet import MyDenseNet121
os.makedirs(dst, exist_ok=True)
seg = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=1, out_channels=1, dropout_prob=0.2)
seg.load_state_dict(torch.load(os.path.join(src, 'full_6000_new_best_metric_model_segmentation3d_1_1.pth'), map_location='cpu', weights_only=True)); seg.eval()
torch.onnx.export(seg, torch.rand(1, 1, 256, 256, 32), dst + '/yale_seg32.onnx', input_names=['x'], output_names=['logit'], opset_version=17, dynamo=False)

class MCDrop(nn.Module):
    def __init__(self, p): super().__init__(); self.p = p; self.mc = None
    def forward(self, x):
        mask = (torch.rand_like(x[:, :, :1, :1, :1]) >= self.p).float() / (1 - self.p)
        return x * (1 + self.mc * (mask - 1))
class Wrap(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net; self.drops = []
        for _, mod in list(net.named_modules()):
            for cname, child in list(mod.named_children()):
                if 'Dropout' in type(child).__name__:
                    d = MCDrop(child.p); setattr(mod, cname, d); self.drops.append(d)
    def forward(self, x, mc):
        for d in self.drops: d.mc = mc
        return self.net(x)
cls = MyDenseNet121(spatial_dims=3, in_channels=2, out_channels=128, dropout_prob=0.1)
cls.load_state_dict(torch.load(os.path.join(src, '96dilation_best_metric_model_classification3d_11.pth'), map_location='cpu', weights_only=True)); cls.eval()
torch.onnx.export(Wrap(cls).eval(), (torch.rand(1, 2, 128, 128, 96), torch.zeros(1)), dst + '/yale_cls32.onnx', input_names=['x', 'mc'], output_names=['logit'], opset_version=17, dynamo=False)
here = os.path.dirname(os.path.abspath(__file__))
for n in ('seg', 'cls'):
    subprocess.check_call([sys.executable, os.path.join(here, 'to_fp16.py'), f'{dst}/yale_{n}32.onnx', f'{dst}/yale_{n}.onnx'])
    os.remove(f'{dst}/yale_{n}32.onnx')
