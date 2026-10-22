"""Export the CLAIM aSAH nnU-Net v2 2D model to ONNX for SAHVAI Web (panel 2).

    pip install torch dynamic-network-architectures onnx onnxruntime
    python claim_export.py /path/to/aSAH-multiclass-segmentation /path/to/2d_checkpoint_best.pth ../models/claim

Weights: Kiewitz et al., Front Neurol 2024; Zenodo 10.5281/zenodo.12579491; CC BY-NC 4.0 (non-commercial).
Writes claim2d.raw.onnx (fp32), then claim2d.onnx (fp16 weights via to_fp16.py) split into claim2d.onnx.0, .1
(GitHub Pages limits files to 100 MB; the worker joins the parts). Input x (1,1,512,512) normalised CT, output logits (1,7,512,512).
"""
import json, os, sys, subprocess, numpy as np, torch, onnxruntime as ort
from dynamic_network_architectures.architectures.unet import PlainConvUNet
REPO, CKPT, OUT = sys.argv[1:4]
p = json.load(open(f'{REPO}/nnUNET_results/Dataset001_SAH/nnUNetTrainer__nnUNetPlans__2d/plans.json'))
c = p['configurations']['2d']; n = len(c['pool_op_kernel_sizes'])
feats = [min(c['UNet_base_num_features'] * 2 ** i, c['unet_max_num_features']) for i in range(n)]
net = PlainConvUNet(1, n, feats, torch.nn.Conv2d, c['conv_kernel_sizes'], c['pool_op_kernel_sizes'], c['n_conv_per_stage_encoder'], 7,
                    c['n_conv_per_stage_decoder'], conv_bias=True, norm_op=torch.nn.InstanceNorm2d, norm_op_kwargs={'eps': 1e-5, 'affine': True},
                    nonlin=torch.nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False).eval()
sd = torch.load(CKPT, map_location='cpu', weights_only=False)['network_weights']
sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}
print(net.load_state_dict(sd, strict=False))
x = torch.randn(1, 1, 512, 512)
with torch.no_grad(): ref = net(x).numpy()
os.makedirs(OUT, exist_ok=True)
raw = os.path.join(OUT, 'claim2d.raw.onnx'); fp16 = os.path.join(OUT, 'claim2d.onnx')
torch.onnx.export(net, (x,), raw, input_names=['x'], output_names=['logits'], opset_version=17, dynamo=False)
y = ort.InferenceSession(raw, providers=['CPUExecutionProvider']).run(None, {'x': x.numpy()})[0]
print(f'ONNX Runtime vs PyTorch: max logit difference {np.abs(y - ref).max():.2e}, argmax mismatches {(y.argmax(1) != ref.argmax(1)).sum()}')
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), 'to_fp16.py'), raw, fp16])
y16 = ort.InferenceSession(fp16, providers=['CPUExecutionProvider']).run(None, {'x': x.numpy()})[0]
print(f'fp16 vs PyTorch: max logit difference {np.abs(y16 - ref).max():.2e}, argmax mismatches {(y16.argmax(1) != ref.argmax(1)).sum()}')
b = open(fp16, 'rb').read(); half = (len(b) + 1) // 2
for i, part in enumerate([b[:half], b[half:]]): open(f'{fp16}.{i}', 'wb').write(part)
os.remove(fp16)
if not os.environ.get('KEEP_RAW'): os.remove(raw)
print(f'wrote {fp16}.0 and .1 ({len(b)} bytes in total); set MODELS.claim.files.claim2d.size = {len(b)} in worker.js')
