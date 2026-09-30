# SAHVAI in the browser

**https://sahvai-lab.github.io/SAHVAI-Calculator/sahvai-web/** (also the *SAHVAI Web (AI)* tab of the [SAHVAI Calculator](https://sahvai-lab.github.io/SAHVAI-Calculator/))

Segments subarachnoid (SAH), intraparenchymal (IPH) and intraventricular (IVH) hemorrhage on a non-contrast head CT and reports their volumes, entirely in the browser. It runs the Hybrid 2D/3D UNet of Hu et al. (*NeuroImage* 2023) with the weights published in [BrainHemoAI](https://github.com/BrainHemo/BrainHemoAI), converted to ONNX and executed with ONNX Runtime Web on WebGPU or the CPU (WebAssembly, multi-threaded). The design follows [RADAR in the browser](https://github.com/jarrelscy/radar-web), the browser port of Alibaba DAMO Academy's RADAR abdominal CT model (Zhang, Ding et al., *Science* 2026).

Images never leave the computer: files are read and processed in the browser tab and nothing is uploaded. The model (26.5 MB) is served from this site and cached by the browser.

> **Research use only.** This is not a medical device. It has not been validated or approved for clinical use and must not be used for diagnosis or treatment decisions. Always review the segmentation against the images.

## Inputs

- a zip of a DICOM study, a DICOM folder or loose DICOM files
- a `.nii` / `.nii.gz` volume, or a `.nrrd` volume (e.g. from 3D Slicer)
- **Try the demo case** loads the aneurysmal SAH case shipped with BrainHemoAI

From a DICOM study the series is chosen in this order: axial; not a bone/lung kernel, contrast, angiography, reformat or scout; labelled head/brain; slice thickness nearest 5 mm; most images. Transfer syntaxes: uncompressed, deflated, RLE, JPEG lossless, JPEG-LS, JPEG 2000 and HTJ2K. Enhanced multi-frame DICOM isn't supported.

## Outputs

- SAH, IPH, IVH and total volumes (mL) and per-slice volumes
- an axial viewer with brain / blood / subdural / bone windows, overlay opacity and a brain-outline quality check
- downloads: label map (`.nii.gz`, 1 = SAH, 2 = IPH, 3 = IVH) and the resampled CT it was computed on (`.nii.gz`, same grid; both open together in 3D Slicer or ITK-SNAP), a JSON report and a PNG of the current view

## How it works

1. **Resample.** The volume is reoriented to standard axial order (x → left, y → posterior, z → superior) and resampled to the training geometry: 0.5 × 0.5 mm in plane and 5 mm slices. Native spacing already within 0.45–0.55 mm / 4.5–5.5 mm is kept. Thin slices are area-averaged into 5 mm slabs.
2. **Brain extraction.** BrainHemoAI does not include its brain-extraction step, so this port adds one: soft tissue (−40 to 130 HU) that a ~5 mm ball cannot reach from the skin without crossing bone, air or fat, restricted to tissue walled in by bone in ≥ 12 of 16 directions, joined in 3D (largest component after a 2 mm erosion). It is then cleaned with the upstream `rectify_brain_mask` (−20 to 100 HU, holes < 10 000 px filled, pieces < 1 000 px removed, calcification ≥ 120 HU removed). On the demo case it matches the upstream brain mask with Dice 0.995.
3. **Alignment.** The head is rotated to the midline. The upstream platform leaves this to the user; here the angle that makes the brain outline most left-right symmetric is used (17° on the demo case, where upstream used 17° by hand), or it can be set by hand. The network is not very sensitive to it (SAH volume within 1.5% between 0° and 25° on the demo case).
4. **Network input** (`tri_hybrid_unet_wrapper.predict`). Crop to the brain, resize to 352 × 288, keep slices with ≥ 20% brain, split into five HU windows (0–100, 10–90, 20–80, 30–80, 40–80) with z-score normalisation. The network takes 18 slices; upstream only analyses the first 18, this port adds an overlapping window so slices above them are analysed too (the first window is identical to upstream).
5. **Network.** The Hybrid 2D/3D UNet (inference branch only; the auxiliary decoders are unused) was rewritten with 2D operations only, slices as the batch axis: a 3×3×3 convolution becomes three 2D convolutions summed over neighbouring slices, 3D InstanceNorm becomes a mean/variance over slices and pixels, BatchNorm is folded in. It is exported as nine ONNX parts (`models/`, float16 weights, float32 arithmetic). The full-resolution parts run a few slices at a time with a one-slice halo; each 3D InstanceNorm there gets whole-volume statistics from per-slice sums, so peak memory stays within a browser tab. The part at half resolution and below runs on the whole window.
6. **Back to the scan and post-processing** (`utils/postprocess.py`, "connect" mode): labels are resized and rotated back per class, gaps closed (disk 3), fragments < 100 px removed and mixed regions reconciled (`rectify_blood_type`, twice). Volume = voxels × voxel size.

## Validation against the PyTorch original (demo case)

| check | result |
|---|---|
| 2D-only network vs original `TriHybridUNet` (same input) | max logit difference 2e-5, 0 voxels differ |
| ONNX parts (float16 weights) via ONNX Runtime vs PyTorch | 14 of 1.8 M voxels differ |
| network input built in JavaScript vs Python (same mask, 17°) | identical crop, slices and values (max difference < 0.001) |
| full JavaScript pipeline vs `demo.run(..., post_mode='connect')` (same mask, 17°, float input) | 6 of 7.3 M voxels differ; SAH Dice 1.000, IVH Dice 0.9998 |
| fully automatic (own brain mask, estimated angle) | SAH 77.2 mL, IVH 3.4 mL vs 77.5 / 2.7 mL with the upstream mask |

Upstream `demo.py` casts the rotated scan back to int16; with that truncation the upstream result differs from the float path by about 2.5% in SAH volume (Dice 0.976). The tools in `tools/` reproduce the conversion (`export_2d.py`, `parts.py`, `to_fp16.py`).

## Requirements

Recent desktop Chrome or Edge (tested in Chromium); Firefox and Safari may work on the CPU. 8 GB RAM or more. Measured on the demo case (2 windows of 18 slices), Intel Mac with 8 threads: **43 s on WebGPU, 118 s on the CPU**, identical volumes. Multi-threaded CPU needs the page to be cross-origin isolated, which `coi-serviceworker.js` provides when the page is opened on its own; embedded as a tab of the calculator it runs single-threaded on the CPU (WebGPU is unaffected).

## Running locally

Any static file server: `python3 -m http.server 8000` in the repository root, then open `http://localhost:8000/sahvai-web/`.

## Licence and attribution

See [LICENSE.md](LICENSE.md). Please cite:

- Hu P, Zhou H, Yan T, et al. Deep learning-assisted identification and quantification of aneurysmal subarachnoid hemorrhage in non-contrast CT scans: development and external validation of Hybrid 2D/3D UNet. *NeuroImage* 2023;279:120321. doi:10.1016/j.neuroimage.2023.120321
- Zhang Q, Zhang J, …, Liang T. An expert-level generalist AI for abdominal CT diagnosis. *Science* 2026;393(6817):eaec6129. doi:10.1126/science.aec6129 (RADAR; the browser architecture of this port follows radar-web)
