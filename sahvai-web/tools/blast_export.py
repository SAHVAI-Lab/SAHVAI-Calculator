"""Export the BLAST-CT DeepMedic models to ONNX for SAHVAI Web (panel 3).

    pip install blast-ct onnx onnxruntime
    python blast_export.py /path/to/blast-ct 1 2 ... 12      # writes blast_m{k}.raw.onnx
    python to_fp16.py blast_m1.raw.onnx ../models/blast/blast_m1.onnx   (for each k)

BLAST-CT (github.com/biomedia-mira/blast-ct) is Copyright 2020 Imperial College London, Apache-2.0.
The forward pass is re-expressed with static shapes (identical arithmetic) because the original computes
crop sizes from tensor shapes, which the ONNX tracer cannot export. Output tile 60^3, input 140^3, softmax.
"""
import json, sys, time, torch, torch.nn.functional as F, numpy as np, onnxruntime as ort
REPO = sys.argv[1]; MODELS = sys.argv[2:] or ['1']; OUT = 60
sys.path.insert(0, REPO)
from blast_ct.read_config import get_model
from blast_ct.models.base import DownSample, UpSample
cfg = json.load(open(f'{REPO}/blast_ct/data/config.json'))
m = get_model(cfg).eval()
IN = m.calculate_input_size((OUT,) * 3)

def crop(x, size):
    sl = [slice(None), slice(None)] + [slice(c // 2 - s // 2, c // 2 + s // 2 + s % 2) for c, s in zip(x.shape[2:], size)]
    return x[tuple(sl)]

def block(b, x):
    out = b.conv(b.prelu(b.bn(x)))
    if x.shape[1] == out.shape[1]: out = out + x[b.shortcut_slices]
    return out

class Static(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x):
        acts = []
        for p in s.m.paths:
            out = crop(x, p.calculate_input_size((OUT,) * 3))
            for layer in p.layers:
                if isinstance(layer, DownSample): out = out[layer.slices]
                elif isinstance(layer, UpSample): out = F.interpolate(out, scale_factor=layer.scale_factor, mode='nearest')
                else: out = block(layer, out)
            acts.append(crop(out, (OUT,) * 3))
        out = torch.cat(acts, 1)
        for b in s.m.fully_connected: out = block(b, out)
        return torch.softmax(out, 1)

for k in MODELS:
    m.load_state_dict(torch.load(f'{REPO}/blast_ct/data/saved_models/model_{k}.torch_model', map_location='cpu'))
    w = Static(m).eval(); x = torch.randn(1, 1, *IN)
    with torch.no_grad(): ref = torch.softmax(m(x)[0], 1).numpy(); mine = w(x).numpy()
    f = f'blast_m{k}.raw.onnx'
    torch.onnx.export(w, (x,), f, input_names=['x'], output_names=['prob'], opset_version=17, dynamo=False)
    y = ort.InferenceSession(f, providers=['CPUExecutionProvider']).run(None, {'x': x.numpy()})[0]
    print(f'model {k}: input {IN}; static vs original {np.abs(mine - ref).max():.2e}; ONNX Runtime vs original {np.abs(y - ref).max():.2e}')
