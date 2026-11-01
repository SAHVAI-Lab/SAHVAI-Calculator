# Rebuild models/headsup/ from the RSNA 2019 1st-place checkpoints (SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection,
# fold 0 of DenseNet121_change_avg_512, DenseNet169_change_avg_256 and se_resnext101_32x4d_256, from the README's Google Drive links).
# Usage: python3 tools/headsup_export.py <folder with densenet121_512_fold0.pth, densenet169_256_fold0.pth, seresnext101_256_fold0.pth>
# Needs torch, torchvision, pretrainedmodels, onnx, onnxruntime. Then: tools/to_fp16.py on each file and split the SE-ResNeXt101 file in two.
import re
import torch, torch.nn as nn, torchvision, pretrainedmodels
class DenseNet121_change_avg(nn.Module):
    def __init__(s):
        super().__init__(); s.densenet121=torchvision.models.densenet121(weights=None).features
        s.avgpool=nn.AdaptiveAvgPool2d(1); s.relu=nn.ReLU(); s.mlp=nn.Linear(1024,6)
    def forward(s,x):
        x=s.avgpool(s.relu(s.densenet121(x))); return s.mlp(x.flatten(1))
class DenseNet169_change_avg(nn.Module):
    def __init__(s):
        super().__init__(); s.densenet169=torchvision.models.densenet169(weights=None).features
        s.avgpool=nn.AdaptiveAvgPool2d(1); s.relu=nn.ReLU(); s.mlp=nn.Linear(1664,6)
    def forward(s,x):
        x=s.avgpool(s.relu(s.densenet169(x))); return s.mlp(x.flatten(1))
class se_resnext101_32x4d(nn.Module):
    def __init__(s):
        super().__init__(); s.model_ft=pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000,pretrained=None)
        s.model_ft.avg_pool=nn.AdaptiveAvgPool2d((1,1)); s.model_ft.last_linear=nn.Sequential(nn.Linear(2048,6,bias=True))
    def forward(s,x): return s.model_ft(x)
def load(name,path):
    m={'dn121':DenseNet121_change_avg,'dn169':DenseNet169_change_avg,'sx101':se_resnext101_32x4d}[name]()
    st=torch.load(path,map_location='cpu',weights_only=False)
    sd={re.sub(r'\.se_fc([12])\.',r'.se_module.fc\1.',k.replace('module.','',1)):v for k,v in st['state_dict'].items()}
    r=m.load_state_dict(sd,strict=True); m.eval(); return m, st.get('epoch'), st.get('valLoss')
# Export SeuTao's RSNA 2019 1st-place 2D CNNs to ONNX with two outputs:
#   prob [N,6]  sigmoid of the original logits (any, epidural, intraparenchymal, intraventricular, subarachnoid, subdural)
#   cam  [N,6,h,w] class activation map: per-cell logit contribution W_c . relu(f) (mean over cells + bias = logit, exactly)
import sys, torch, torch.nn as nn, numpy as np, onnx, onnxruntime as ort, warnings
warnings.filterwarnings('ignore')

P = sys.argv[1].rstrip('/') + '/'
class Wrap(nn.Module):
    def __init__(s, m, kind):
        super().__init__(); s.m, s.kind = m, kind
    def forward(s, x):
        m = s.m
        if s.kind == 'sx101':
            f = m.model_ft; y = f.layer4(f.layer3(f.layer2(f.layer1(f.layer0(x))))); lin = f.last_linear[0]
        else:
            y = m.relu(getattr(m, 'densenet121' if s.kind == 'dn121' else 'densenet169')(x)); lin = m.mlp
        cam = torch.einsum('kc,nchw->nkhw', lin.weight, y)
        logit = cam.mean(dim=(2, 3)) + lin.bias
        return torch.sigmoid(logit), cam
for kind, f, S in [('dn121', 'densenet121_512_fold0.pth', 512), ('dn169', 'densenet169_256_fold0.pth', 256), ('sx101', 'seresnext101_256_fold0.pth', 256)]:
    m, *_ = load(kind, P + f); w = Wrap(m, kind).eval()
    x = torch.randn(2, 3, S, S)
    with torch.no_grad():
        p, cam = w(x); ref = torch.sigmoid(m(x))
    print(kind, 'wrapper vs original max|dp| =', float((p - ref).abs().max()), 'cam', tuple(cam.shape))
    out = f'models/headsup/headsup_{kind}.fp32.onnx'
    import os; os.makedirs('models/headsup', exist_ok=True)
    torch.onnx.export(w, x, out, input_names=['x'], output_names=['prob', 'cam'], dynamic_axes={'x': {0: 'n'}, 'prob': {0: 'n'}, 'cam': {0: 'n'}}, opset_version=17, dynamo=False)
    s = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    po, co = s.run(None, {'x': x.numpy()})
    print(kind, 'onnx vs torch max|dp| =', np.abs(po - p.numpy()).max(), 'max|dcam| =', np.abs(co - cam.numpy()).max(), 'size MB', os.path.getsize(out) / 1e6)
