# SAHVAI Web: hemorrhage segmentation models in the browser

**https://sahvai-lab.github.io/SAHVAI-Calculator/sahvai-web/** (also the *SAHVAI Web (AI)* tab of the [SAHVAI Calculator](https://sahvai-lab.github.io/SAHVAI-Calculator/))

Nine pages, each its own tab in the SAHVAI Calculator, all running entirely in the browser with ONNX Runtime Web (WebGPU, or the CPU via multi-threaded WebAssembly). Each has drag-and-drop for DICOM / NIfTI / NRRD, a demo case, an axial viewer with per-slice volumes, downloads, and its citation, licence and disclaimer. The SAHVAI page (`index.html`, `app.js`, `worker.js`) is the original BrainHemoAI tool and loads only that model; the SAHVAI 3D-4D page (`3d.html`, `app-3d.js`) reuses its `worker.js`; the other five (`claim.html`, `blast.html`, `deepbleed.html`, `sahswin.html`, `rhuh.html`) share `app-models.js`, `worker-models.js` and `style-models.css`, and each loads only its own model.

| page | model / data | from | classes | licence |
|---|---|---|---|---|
| HEADS-UP detector (`headsup.html`) | 2D CNNs of the RSNA 2019 1st-place solution (Wang et al., *NeuroImage Clin* 2021): DenseNet121, or an ensemble with DenseNet169 and SE-ResNeXt101 | Mayo Clinic HEADS-UP (Salman et al., *Mayo Clin Proc Digit Health* 2023); weights by Tao Shen and colleagues | detection of any ICH and EDH, IPH, IVH, SAH, SDH per slice; hand-off to the segmentation pages | non-commercial, with Tao Shen's written permission; RSNA data non-commercial |
| SAHVAI (`index.html`) | Hybrid 2D/3D UNet, BrainHemoAI weights (Hu et al., *NeuroImage* 2023) | BrainHemoAI | SAH, IPH, IVH | MIT (with permission) |
| CLAIM aSAH (`claim.html`) | nnU-Net v2 2D multiclass aSAH model (Kiewitz et al., *Front Neurol* 2024) | CLAIM, Charité Berlin | SAH, IVH, ventricles, ICH, aneurysm, SDH | CC BY-NC 4.0 |
| BLAST-CT (`blast.html`) | DeepMedic, 1 or 12 models (Monteiro et al., *Lancet Digit Health* 2020) | Cambridge / Imperial College London | IPH, extra-axial haemorrhage, oedema, IVH (traumatic brain injury) | Apache-2.0 |
| DeepBleed ICH (`deepbleed.html`) | 3D V-Net, MISTIE-2 weights (Sharrock et al., *Neuroinformatics* 2021) | MISTIE investigators, Johns Hopkins | ICH + IVH (one class) | CC BY-NC 4.0 (with permission; published CC BY-NC-ND) |
| ICH expansion signs (`hesigns.html`) | the SAHVAI model above, then rule-based readers of the CT signs of hematoma expansion; DeepMedX sICH_HE ROI step re-implemented (Yu et al., *npj Digit Med* 2026) | SAHVAI Lab | blend, black hole, hypodensity, swirl, island, satellite, irregular shape, heterogeneous density, fluid level (NCCT); spot-sign candidates (CTA); BAT, BRAIN and target expansion (Morotti et al., *Neurology* 2026) scores | app code CC BY-NC-SA 4.0; no third-party weights |
| SAH Swin UNETR (`sahswin.html`) | Swin UNETR from the Valladolid SAH mortality pipeline (García-García, Cepeda et al., *World Neurosurg* 2024; *Brain Sci* 2024) | Río Hortega University Hospital, Valladolid / University of Augsburg | blood (SAH with any extension; one class) | academic non-commercial (with permission) |
| SAHVAI 3D-4D (`3d.html`) | the SAHVAI model above, plus 3D surfaces (three.js) and volumes over serial scans | SAHVAI Lab | SAH, IPH, IVH in 3D; volume vs date | MIT (with permission) |
| RHUH dataset (`rhuh.html`) | open aSAH NCCT scans with expert bleeding masks (Cepeda & García-García, Zenodo 2023) | Río Hortega University Hospital, Valladolid | expert mask vs any of four models (Dice, volumes) | CC BY 4.0 (cases 001–003 hosted as demos) |

The design follows [RADAR in the browser](https://github.com/jarrelscy/radar-web), the browser port of Alibaba DAMO Academy's RADAR abdominal CT model (Zhang, Ding et al., *Science* 2026).

Images never leave the computer: files are read and processed in the browser tab and nothing is uploaded. Models are served from this site and cached by the browser (HEADS-UP 14.3 MB, or 134 MB for its ensemble; SAHVAI 26.5 MB, BLAST-CT 2.1 MB per model, CLAIM 92.7 MB and DeepBleed 90.2 MB, each in two parts; SAH Swin UNETR 276.8 MB in three).

> **Research and education use only.** This is not a medical device. None of these models has been validated or approved for clinical use in this form; do not use them for diagnosis, triage or treatment decisions. Always review the segmentation against the images.

## HEADS-UP detector — `headsup.html`

Step 1 of the suite. HEADS-UP (Hemorrhage Evaluation and Detector System for Underserved Populations; Salman et al., *Mayo Clin Proc Digit Health* 2023;1:547–556) answers: is there intracranial hemorrhage, which of the five types, and on which slices? It then sends the same scan to the segmentation page that suits the type found. The 2023 HEADS-UP was a Google Cloud AutoML classifier; this version runs open networks on the device, including in a phone browser (`app-headsup.js`, `lib/headsup.js`, model `headsup` in `worker-models.js`).

- **Networks.** The 2D CNNs of the RSNA 2019 Intracranial Hemorrhage Detection 1st-place solution ([SeuTao](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection); Wang et al., *NeuroImage Clin* 2021;32:102785), fold 0: DenseNet121 at 512 px by default (14.3 MB), or the average of DenseNet121, DenseNet169 and SE-ResNeXt101 (134 MB). The solution's sequence models have no published weights and are not used.
- **Preprocessing** (as `prepare_data.py` / `predict.py`): slices thinner than 4 mm averaged to 5 mm; each slice in the DICOM's brain window (else 40/80), min-max to 8 bits, 512 px, stacked with the neighbouring slices as three channels, 80 % centre crop, ImageNet-style normalisation. OpenCV's linear resize is reproduced to its fixed-point rounding.
- **Outputs.** Per-slice probabilities of any hemorrhage, EDH, IPH, IVH, SAH and SDH; the study result per type is the highest slice probability (detected ≥ 0.5, possible ≥ 0.2; provisional thresholds). An approximate localisation (hyperdense 45–100 HU voxels inside each detected type's class-activation hot spot) is overlaid and downloadable (`heads-up-labels.nii.gz`, 1 EDH, 2 IPH, 3 IVH, 4 SAH, 5 SDH); it is not a segmentation. JSON report with per-slice probabilities; CT of the analysed grid.
- **Hand-off** (`lib/handoff.js`). The tools are ranked by the types found (SAH → SAHVAI first; ICH without SAH → DeepBleed; SDH/EDH alone → BLAST-CT). The scan the user gave (original files or demo URL) and the HEADS-UP findings are stored in the browser's IndexedDB and the chosen page opens with `?handoff=<id>`, shows a banner and runs the scan. In the SAHVAI Calculator the HEADS-UP tab switches to the chosen tab (`sahvai-open-tab` message). Entries expire after a day; nothing is uploaded.
- **Checks** (headless Chromium, CPU). Browser vs Python reference (OpenCV + ONNX Runtime) on BHSD SAH_002: slice probabilities within 8.4e-6. ONNX vs PyTorch: < 1e-7; float16 storage: < 2e-5. Study results on the demos: BHSD SAH_002 IPH 0.97, IVH 1.00, SAH 1.00 (expert labels: IPH, IVH, SAH); MISTIE/CLEAR 01 IPH 0.96, IVH 0.96; BLAST scan_0 (TBI) any 1.00. On the three BHSD demo cases (96 slices) the slice-level AUC against the expert labels was 0.99 (any), 1.00 (IPH), 0.99 (IVH), 0.97 (SAH). About 30 s for 32 slices on 2 CPU threads. Full flow tested: Calculator → HEADS-UP tab → demo → SAHVAI tab segments the same scan.
- **Not yet done.** No external validation with normal scans: specificity and the thresholds still need a cohort such as CQ500 (Chilamkurthy et al., *Lancet* 2018) or Mayo data.

## SAHVAI (BrainHemoAI Hybrid 2D/3D UNet) — `index.html`

### Inputs

- a zip of a DICOM study, a DICOM folder or loose DICOM files
- a `.nii` / `.nii.gz` volume, or a `.nrrd` volume (e.g. from 3D Slicer)
- **Try the demo case** loads the aneurysmal SAH case shipped with BrainHemoAI

From a DICOM study the series is chosen in this order: axial; not a bone/lung kernel, contrast, angiography, reformat or scout; labelled head/brain; slice thickness nearest 5 mm; most images. Transfer syntaxes: uncompressed, deflated, RLE, JPEG lossless, JPEG-LS, JPEG 2000 and HTJ2K. Enhanced multi-frame DICOM isn't supported.

### Outputs

- SAH, IPH, IVH and total volumes (mL) and per-slice volumes
- an axial viewer with brain / blood / subdural / bone windows, overlay opacity and a brain-outline quality check
- downloads: label map (`.nii.gz`, 1 = SAH, 2 = IPH, 3 = IVH) and the resampled CT it was computed on (`.nii.gz`, same grid; both open together in 3D Slicer or ITK-SNAP), a JSON report and a PNG of the current view

### How it works

1. **Resample.** The volume is reoriented to standard axial order (x → left, y → posterior, z → superior) and resampled to the training geometry: 0.5 × 0.5 mm in plane and 5 mm slices. Native spacing already within 0.45–0.55 mm / 4.5–5.5 mm is kept. Thin slices are area-averaged into 5 mm slabs.
2. **Brain extraction.** BrainHemoAI does not include its brain-extraction step, so this port adds one: soft tissue (−40 to 130 HU) that a ~5 mm ball cannot reach from the skin without crossing bone, air or fat, restricted to tissue walled in by bone in ≥ 12 of 16 directions, joined in 3D (largest component after a 2 mm erosion). It is then cleaned with the upstream `rectify_brain_mask` (−20 to 100 HU, holes < 10 000 px filled, pieces < 1 000 px removed, calcification ≥ 120 HU removed). On the demo case it matches the upstream brain mask with Dice 0.995.
3. **Alignment.** The head is rotated to the midline. The upstream platform leaves this to the user; here the angle that makes the brain outline most left-right symmetric is used (17° on the demo case, where upstream used 17° by hand), or it can be set by hand. The network is not very sensitive to it (SAH volume within 1.5% between 0° and 25° on the demo case).
4. **Network input** (`tri_hybrid_unet_wrapper.predict`). Crop to the brain, resize to 352 × 288, keep slices with ≥ 20% brain, split into five HU windows (0–100, 10–90, 20–80, 30–80, 40–80) with z-score normalisation. The network takes 18 slices; upstream only analyses the first 18, this port adds an overlapping window so slices above them are analysed too (the first window is identical to upstream).
5. **Network.** The Hybrid 2D/3D UNet (inference branch only; the auxiliary decoders are unused) was rewritten with 2D operations only, slices as the batch axis: a 3×3×3 convolution becomes three 2D convolutions summed over neighbouring slices, 3D InstanceNorm becomes a mean/variance over slices and pixels, BatchNorm is folded in. It is exported as nine ONNX parts (`models/`, float16 weights, float32 arithmetic). The full-resolution parts run a few slices at a time with a one-slice halo; each 3D InstanceNorm there gets whole-volume statistics from per-slice sums, so peak memory stays within a browser tab. The part at half resolution and below runs on the whole window.
6. **Back to the scan and post-processing** (`utils/postprocess.py`, "connect" mode): labels are resized and rotated back per class, gaps closed (disk 3), fragments < 100 px removed and mixed regions reconciled (`rectify_blood_type`, twice). Volume = voxels × voxel size.

### Validation against the PyTorch original (demo case)

| check | result |
|---|---|
| 2D-only network vs original `TriHybridUNet` (same input) | max logit difference 2e-5, 0 voxels differ |
| ONNX parts (float16 weights) via ONNX Runtime vs PyTorch | 14 of 1.8 M voxels differ |
| network input built in JavaScript vs Python (same mask, 17°) | identical crop, slices and values (max difference < 0.001) |
| full JavaScript pipeline vs `demo.run(..., post_mode='connect')` (same mask, 17°, float input) | 6 of 7.3 M voxels differ; SAH Dice 1.000, IVH Dice 0.9998 |
| fully automatic (own brain mask, estimated angle) | SAH 77.2 mL, IVH 3.4 mL vs 77.5 / 2.7 mL with the upstream mask |

Upstream `demo.py` casts the rotated scan back to int16; with that truncation the upstream result differs from the float path by about 2.5% in SAH volume (Dice 0.976). The tools in `tools/` reproduce the conversion (`export_2d.py`, `parts.py`, `to_fp16.py`).

## CLAIM aSAH multiclass nnU-Net (Charité Berlin) — `claim.html`

The nnU-Net v2 **2d** configuration from [claim-berlin/aSAH-multiclass-segmentation](https://github.com/claim-berlin/aSAH-multiclass-segmentation) (weights: [Zenodo 12579491](https://doi.org/10.5281/zenodo.12579491)): PlainConvUNet, 8 stages, 32→512 features, 46 M parameters, 7 outputs (0 background, 1 SAH, 2 IVH, 3 ventricles, 4 ICH, 5 aneurysm, 6 SDH). The authors report similar performance for their 2D and 3D models and used the 2D model for external validation; the 2D model is the practical one for a browser.

`lib/claim.js` follows `nnUNetv2_predict` for the 2D configuration: reorient; clip to −3…88 HU, then resample in plane to the plans spacing 0.4453 mm with a cubic B-spline (nnU-Net order 3; slice spacing kept, and slices thinner than 4 mm are averaged to 5 mm, a documented deviation, since training scans were 5.5–6.5 mm); z-score with mean 32.47 and SD 27.41; 512 × 512 tiles with nnU-Net's sliding window (step 0.5, Gaussian importance map); argmax. The NIfTI storage order nnU-Net sees (L-A-S for dcm2niix-style files) is reproduced by flipping rows. **Exact** (default) is the full sliding window; **Fast** restricts it to the skull plus 8 mm (about half the passes, small differences at tile edges). Optional mirroring over both in-plane axes, nnU-Net's default test-time augmentation.

Validation with the published weights (`2d_checkpoint_best.pth`, Zenodo 12579491), on the three BHSD demo cases:

| check | result |
|---|---|
| this checkpoint in nnUNetv2_predict (CPU) vs the authors' `segmentations_2d` | 373 of 8.4 M voxels differ (case 1) |
| ONNX fp32 / fp16 vs PyTorch (random input) | max logit difference 4e-4 / 7e-2; argmax 0 / 10 of 262 k pixels differ |
| preprocessing in JavaScript vs nnU-Net (case 1) | identical to 2e-7 (z-scored units) except one border row |
| network + sliding window in JavaScript (fp16, exact, no mirroring) vs nnU-Net, in nnU-Net's own grid (case 1) | **31 of 10.1 M voxels differ**; SAH Dice 0.9995, IVH 1.000, ventricles 0.9999 |
| any-blood Dice vs the expert labels, JavaScript / nnU-Net (exact, case 1) | 0.444 / 0.446 |
| any-blood Dice vs the expert labels, fast mode / nnU-Net, cases 1–3 | 0.448 / 0.446, 0.814 / 0.819, 0.799 / 0.798 |
| fast vs exact mode (case 1) | 4 683 voxels differ; SAH Dice 0.93, IVH 0.95, ventricles 0.99 |

Demo: three BHSD cases shipped with the CLAIM repository, with their reference labels (BHSD scheme 1 EDH, 2 IPH, 3 IVH, 4 SAH, 5 SDH, mapped onto CLAIM's classes for Dice). Only the 2D model is ported; the 3D full-resolution checkpoint is not used.

## BLAST-CT (traumatic brain injury lesions) — `blast.html`

The pre-trained DeepMedic models of [biomedia-mira/blast-ct](https://github.com/biomedia-mira/blast-ct) (Monteiro, Newcombe et al., *Lancet Digit Health* 2020; University of Cambridge Division of Anaesthesia / NeuroICU and BioMedIA, Imperial College London): three pathways (input scales 1, 3 and 5), 5 outputs (background, IPH, extra-axial haemorrhage, perilesional oedema, IVH). It is a different task from SAHVAI and CLAIM: it was trained on traumatic brain injury (CENTER-TBI and others), and its extra-axial class combines traumatic SAH, SDH and EDH. Useful for SAH after a fall, or for traumatic ICH.

`lib/blast.js` follows blast-ct: reorient, resample to 1 mm isotropic (SimpleITK-style linear, corner-aligned), window −15…100 HU scaled to −1…1, tile with non-overlapping 60³ output blocks (140³ edge-padded input), softmax, sum over models, argmax. **Single model** is the tool's default (`model_1`); **Ensemble of 12** matches `--ensemble`.

Validation on `scan_0` from the blast-ct repository: the static ONNX re-implementation matches PyTorch exactly (max difference 0) and ONNX Runtime within 2.3e-6; the JavaScript pipeline (fp32 model) gives the same label map as the `blast-ct` command line in all but **2 of 6.0 M voxels** (IPH 48.0 mL, EAH 23.7 mL, oedema 62.7 mL, IVH 0.3 mL). The fp16 files used on the site change 0.01% of voxels on test tiles. 60³ output tiles give identical labels to the tool's 30³ tiles.

## DeepBleed ICH (MISTIE investigators, Johns Hopkins) — `deepbleed.html`

[msharrock/deepbleed](https://github.com/msharrock/deepbleed) (Sharrock et al., *Neuroinformatics* 2021): a 3D V-Net (22.5 M parameters; each block's 5×5×5 convolution is one layer applied 1–3 times) trained on 100 MISTIE II scans and validated on 500 MISTIE III scans (Dice 0.91), segmenting ICH and IVH as one class. External validation in 1,040 patients: median Dice 0.84 (J Clin Med 2023). Licence: published CC BY-NC-ND 4.0; this browser version is used under CC BY-NC 4.0 with Dr Matthew Sharrock's permission (November 2026). The weights are converted to ONNX in full precision without change (`tools/deepbleed_export.py`; the 2×2×2 transposed convolutions are written as 1×1×1 convolutions plus an interleave, identical arithmetic, because ONNX Runtime's WebGPU backend has no 3D transposed convolution); see LICENSE.md.

`predict.py` does: FSL BET brain extraction → ANTs rigid registration (Mattes MI) of the brain to a 1.5 mm, 128³ CT template → V-Net → mask > 0.5 → inverse transform (genericLabel) to the scan. `lib/deepbleed.js` reproduces the geometry; brain extraction is SAHVAI's (`lib/brain.js`), since FSL cannot run in a browser (for a skull-stripped scan, its own non-zero voxels). One quirk is kept on purpose: `predict.py`'s `nii2ants()` passes the NIfTI qform (RAS numbers) to ANTs, which reads them as LPS, so the brain reaches the registration turned 180° about the vertical axis, and ANTs, starting from the centres of mass with no rotation, keeps it so. The network was trained on inputs made this way, and it predicts well only on them. Registration here: centres of mass (as ANTs' initialiser), then a mutual-information refinement (64 bins, rotations within ±0.1 rad, shrink 4 and 2), optional.

Validation (MISTIE demo = JHU case 01; references run with `ants.registration` exactly as `predict.py`, with a given brain mask since FSL is not installable here):

| check | result |
|---|---|
| ONNX vs a TensorFlow reference from the same checkpoint (128³ test volume) | max probability difference 4e-5; 0 voxels differ |
| resampling with ANTs' own transform vs ANTs' `warpedmovout` | correlation 0.99994; inverse mapping vs `apply_transforms` Dice 1.000 |
| browser registration vs ANTs, same brain mask, native-space Dice | JHU-01 0.93 (15.2 vs 15.1 mL); BLAST scan_0 0.94; BHSD SAH_002 0.95 (Python prototype of the same method) |
| JHU-01, full browser pipeline (SAHVAI brain extraction; mask Dice 0.989 vs the manual brain mask) | 23.9 mL: the additional 8.7 mL is intraventricular blood higher in the lateral ventricle, beside the drain, that the reference run did not label |
| BHSD SAH_002, full browser pipeline, vs expert ICH + IVH | 94.0 vs 98.0 mL, Dice 0.870 |

DeepBleed runs on the CPU (WebAssembly) whatever the Compute setting: on WebGPU (tested: Chrome, AMD Radeon Pro Vega) its 128³ V-Net returned an all-zero output and took 97 s, against a correct result in 24 s on 8 CPU threads.

DeepBleed is sensitive to small changes in its input (brain mask, placement in the template): expect volume differences of several mL between runs of the original and this port on the same scan.

## ICH expansion signs — `hesigns.html`

Reads the CT signs that predict hematoma expansion in spontaneous ICH (`app-hesigns.js`, `lib/hesigns.js`). The scan is segmented by the SAHVAI worker (`worker.js`, asked with `wantRaw` for the labels before small fragments are removed); the hematoma is the largest 26-connected IPH component (optionally with IVH).

- **Signs.** Each published visual definition (Morotti et al., *Ann Neurol* 2019 consensus) is turned into measurable rules on the 0.5 × 0.5 × 5 mm grid: hypodensity, black hole (≥ 28 HU), swirl, blend (≥ 18 HU, open to the edge, sharp interface), fluid level, heterogeneous density (≥ 3 foci, Barras), irregular shape (≥ 2 margin irregularities, Barras), island (Li 2017) and satellite (Shimoda 2017). The thresholds are in `CRITERIA` and listed on the page. On a contrast-enhanced scan (≥ 2 mL of 130–600 HU in the brain) the NCCT signs are not read and ≥ 120 HU foci inside the clot are spot-sign candidates with an approximate spot sign score. Leakage and iodine signs need a delayed phase or dual-energy maps and are reader-entered. Sections with dense metal (coils, clips) are not read; catheters near the clot are flagged.
- **Reader.** Every sign card has a *Reader* control (agree / present / absent); the scores use the reader's call when given.
- **Scores.** BAT (Morotti et al., *Stroke* 2018), BRAIN (Wang et al., *Stroke* 2015) and ultra-early hematoma growth, from the signs and the onset-to-CT time, warfarin, recurrent ICH and IVH entered on the page.
- **DeepMedX.** The ROI step of [DeepMedX/sICH_HE](https://github.com/DeepMedX/sICH_HE) (`crop2DROI.ipynb`: 3.5× upsampling, slice of largest area and its neighbours, bounding box of the largest component, window 45/90 HU; then the ViT's resize 256 / centre crop 224) is re-implemented, and the three patches shown and downloadable. Their U-Mamba, Diffusion-UKAN and ViT weights are not public, so the expansion probability of their pipeline is not computed. The paper describes a 40/80 HU window and ±1 slice at 5 mm; the code (followed here) uses 45/90 HU and neighbours 1/3.5 of a slice apart.
- **Yale expansion model** (`worker-yale.js`, `lib/yalehe.js`). Tran AT, et al., Payabvash S. *npj Digit Med* 2024;7:26 ([anhtrnyaleedu/HE](https://github.com/anhtrnyaleedu/HE), checkpoints in the repository): MONAI SegResNet hematoma segmentation, then a two-channel 3D DenseNet121 (CT, and CT within 40 mm of the clot) with Monte-Carlo dropout, trained on ATACH-2 + Yale (n = 634) for expansion ≥ 6 mL (test AUC 0.81 in high-confidence cases). `tools/yale_export.py` converts the checkpoints to ONNX (weights unchanged, float16 storage); `Dropout3d` becomes an in-graph random channel mask switched by an `mc` input, so the Monte-Carlo passes run in the browser. The notebook's preprocessing (brain window, skull strip, pad to 512 × 512 × 48, 50 % sliding window, 1 mm resampling, ± 16 mm slab, 40 mm dilation, area resize, centre crop) is re-implemented; NumPy's round-half-even is reproduced. Checks: ONNX vs PyTorch classifier logits identical, segmentation 7 / 14,473 and 11 / 62,198 voxels different; JS classifier input identical to Python's (mean |Δ| < 1e-8); JS brain window 0.3–0.4 % of voxels different (the skull-strip edge band); browser on BHSD SAH_002: deterministic p 0.175 vs 0.174 in Python, mask 80.0 vs 80.0 mL. The scan reaches the model through the SAHVAI grid flipped to the LAS layout the authors' NIfTI files use. Option: the SAHVAI IPH mask in place of SegResNet (faster; not the published pipeline). **Licence: BSD 3-Clause, © 2024 Yale University; the converted weights are hosted in `models/yale/` with the authors’ permission (S. Payabvash, November 2026).** Copies without the files show a notice instead of the result.
- **Follow-up CT.** A second drop zone runs SAHVAI on the follow-up scan and classifies expansion: conventional HE (ICH growth > 6 mL or > 33 %; Dowlatshahi 2011) and revised HE (rHE: ≥ 6 mL or ≥ 33 %, IVH growth ≥ 1 mL, or new IVH; Yogendrakumar et al., *Stroke* 2020; Ning, Yu et al., *Sci Rep* 2025), plus ≥ 3 / 6 / 9 / 12.5 mL. New IVH from AI volumes means < 0.5 mL at baseline and ≥ 0.5 mL at follow-up. Volumes can be typed over.
- **Spot-sign mimics** (Gazzola et al., *Stroke* 2008;39:1177–1183). On NCCT, calcification (≥ 150 HU, below catheter density) in or within 5 mm of the clot is reported as a would-be CTA mimic; on CTA, every candidate carries the mimic warning (calcification; micro-AVM, aneurysm, moyamoya vessels entering from the edge).
- **Checks.** `node tools/hesigns_selftest.mjs`: each sign injected into a synthetic hematoma is found and a plain hematoma has none (9 phantoms). In headless Chromium the page gave the same signs as Node on BHSD SAH_002 (blend, irregular, island, satellite; the island and satellite counts include subarachnoid blood, which the page warns about) and MISTIE/CLEAR 01 (one satellite focus; no density sign). No expert-read sign data set has been used: sensitivity and specificity are unknown.

## SAH Swin UNETR (Río Hortega University Hospital, Valladolid) — `sahswin.html`

[smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction](https://github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction) (García-García S, Cepeda S, et al., *World Neurosurg* 2024, doi:10.1016/j.wneu.2024.07.216; *Brain Sci* 2024;14:10): the blood-segmentation stage of the group's SAH mortality-prediction pipeline, a MONAI Swin UNETR (feature size 48, 63.1 M parameters, 2 classes) whose checkpoint `SWIN_UNETR_48_100_best_metric_model.pth` is linked from the README (Google Drive). Not CereBleed: the group's newer nnU-Net multiclass tool ([geibac.uva.es/cerebleed](https://geibac.uva.es/cerebleed); Cepeda et al., *Neurosurgery* 2026) is available only as a hosted service. Licence: the repository's Academic Non-Commercial Source Code License Agreement (© 2024 Hospital Río Hortega and University of Augsburg; the README also says CC BY-NC). Hosted with the permission of Dr Santiago Cepeda (1 November 2026), with attribution as published; see LICENSE.md.

Original pipeline (`SAH_mortality_prediction.py` + `inference_2.py`): dcm2niix → FSL brain extraction (as DeepBleed) → ANTs `SyN` registration (affine + deformable, Mattes MI) of the brain to the repository's 1 mm MNI CT template (193 × 229 × 193) → MONAI `Spacingd` 1.5 × 1.5 × 2 mm (bilinear) → `NormalizeIntensityd(nonzero=True)` → `CropForegroundd` (voxels > 0 after normalisation) → `sliding_window_inference` (128 × 128 × 64, overlap 0.25, constant blending) → `Invertd` (bilinear logits) → argmax, in template space. `lib/sahswin.js` reproduces the MONAI steps exactly, including which voxels count as "non-zero" after `Spacingd` (torch's float64 grid leaves values of ~1e-14 at the brain's edge, and the network's volume moves by up to 10% if they are left out). A scan already in the template space (the RHUH preprocessed scans) goes straight to the network; any other scan has its brain extracted (SAHVAI's method) and is registered to a 2 mm version of the template (rotation, translation and per-axis scale, mutual information, 4 then 2 mm) instead of SyN; the logits are mapped back to the scan and volumes are measured there (the original's template-space volumes scale with head size: 184 vs 101–107 mL on BHSD SAH_002).

ONNX (`tools/sahswin_export.py`): weights unchanged in fp32; transposed convolutions as 1×1×1 convolutions plus interleave; window attention in chunks of 64 windows (the first stage's 500 windows × 3 heads × 343² attention matrices are 706 MB each at once; chunked, the session peaks at 2.2 GB of WebAssembly memory instead of hitting the 4 GB limit); shifted-window masks computed in the graph from region ids instead of 280 MB of constants. Runs on the CPU (about 30 s per window on 8 threads, two windows for a typical brain).

| check | result |
|---|---|
| ONNX vs PyTorch (MONAI 1.6), one 128 × 128 × 64 window | max logit difference 1.4e-5; 0 voxels differ |
| browser pipeline vs the original Python pipeline, RHUH cases 001–003 (template space) | Dice 1.000, 1.000, 1.000 (63.0, 12.5, 37.2 mL) |
| the same, vs the RHUH expert masks | Dice 0.772, 0.696, 0.822 (expert 60.8, 13.4, 36.9 mL); the model comes from the same hospital, so not an independent test |
| original pipeline, ANTs SyN vs rigid registration, BHSD SAH_001–003 (native space) | Dice 0.79, 0.93, 0.87 |
| browser registration vs the original with SyN, same brain mask | Dice 0.77, 0.94, 0.90 (19.4 vs 27.6, 106.8 vs 101.0, 23.2 vs 19.7 mL) |
| BHSD SAH_002 in the page, vs expert all-blood labels | 106.8 vs 105.5 mL, Dice 0.832 (original with SyN: 0.829) |

## RHUH SAH dataset (Cepeda & García-García) — `rhuh.html`

[Zenodo 8228847](https://doi.org/10.5281/zenodo.8228847): non-contrast CT scans of aneurysmal SAH from Río Hortega University Hospital (Valladolid, Spain) with expert manual bleeding masks, NIfTI, CC BY 4.0, distributed as one 649 MB RAR archive. Browsers cannot read RAR, so the user extracts it and drops (or chooses) the CT and mask of one case; the page runs the chosen model and reports any-blood Dice and volumes against the expert mask, with the mask outlined in white. No RHUH data are hosted here.

Format (checked on the downloaded archive, MD5 a67bf358ebb326f156071864c318ab42): 100 cases `sah_rhuh_NNN/sah_rhuh_NNN_ct.nii.gz` + `_seg.nii.gz` (binary expert mask), registered to MNI space (193 × 229 × 193, 1 mm, RAS), skull-stripped with background exactly 0 and intensities close to HU. `lib/stripped.js` detects such volumes (many exact zeros, no air, no bone), sets the background to −1000 HU and, for SAHVAI, uses the non-background voxels as the brain mask. Cases 001–003 are hosted in `demo/rhuh/` (CC BY 4.0, unchanged).

Any-blood Dice against the expert masks (browser pipelines, run in Node):

| case | expert mL | SAHVAI (BrainHemoAI) | CLAIM (exact) | BLAST-CT (1 model) |
|---|---|---|---|---|
| 001 | 62.0 | 0.654 (41.1 mL) | 0.115 (18.2 mL) | 0.216 (8.8 mL) |
| 002 | 13.3 | 0.267 (4.9 mL) | 0.017 (13.1 mL) | – |
| 003 | 36.7 | 0.619 (48.4 mL) | 0.372 (30.0 mL) | – |

CLAIM and BLAST-CT degrade badly on these skull-stripped, template-registered scans (CLAIM without the background change: 0.114 on case 001); SAHVAI with the scan's own brain mask holds up best, so it is the page default. These are three cases only.

## SAHVAI 3D-4D — `3d.html`

3D view and volume-over-time chart built on the SAHVAI (BrainHemoAI) pipeline; `app-3d.js`, `style-3d.css`, `lib/mesh.js`, `vendor/three/`. It uses `worker.js` in its multi-scan mode (`{multi: true}`), so the segmentation and volumes are identical to the SAHVAI page.

- **Input.** One scan, or several at once or one after another (*Add to timeline*). DICOM uploads are split into studies by StudyInstanceUID (falling back to study date and time) and the head series of each study is chosen as on the SAHVAI page; each NIfTI / NRRD file is one scan. Scans are ordered by acquisition date (DICOM AcquisitionDate/Time, else Series, else Study), or by a date in the file name (e.g. `2026-04-12`), and every date can be edited in the table. Patient IDs are compared only to warn when scans seem to come from different patients; they are never shown or stored.
- **3D.** Brain = the intracranial mask the pipeline already computes (skull, scalp and table removed). Each label is block-averaged to about 1 mm (blood) or 2 mm (brain), Gaussian-smoothed, interpolated from 5 mm to about 1.25 mm slices and meshed by marching cubes (outward normals from the field gradient), then drawn with three.js: SAH, IPH and IVH opaque, brain as translucent glass, an approximate ventricle layer (CSF density > 12 mm inside the brain outline plus IVH, ≥ 1 mL components near the midline; HU-based context only, not a model output). Rotate, zoom, pan, view presets in radiological orientation (axial seen from below, coronal from the front, sagittal from the left), orientation triad, slice planes, auto-rotate, PNG export.
- **Slices.** Axial, coronal and sagittal views of the 0.5 × 0.5 × 5 mm working grid with the overlay and a linked crosshair.
- **4D.** SAH (SAHV), IPH, IVH and total blood (mL) against scan date, with Δ total, first-to-last change, mean SAH change per day, peak total and, with three or more dated scans over at least a day, an exploratory log-linear SAH half-life. *Play* steps the 3D view through the scans with the same camera; scans are centred on the brain, not registered. Exports: CSV, JSON, chart PNG.
- Volumes always come from the voxel labels, never from the surfaces. SAH is all subarachnoid blood found by the model, not the 5-cistern ABC/2 SAHV used by eSAH.

## Research exports and the Research Dataset — `lib/research.js`, `research.html`

Every tool (SAHVAI, 3D-4D, CLAIM, BLAST-CT, DeepBleed, SAH Swin UNETR, RHUH, HEADS-UP, ICH Expansion Signs) and the manual ABC/2 calculator has a **Research export** box under its result, so each site can build its own de-identified hemorrhage volumetric dataset.

- **Add to site dataset** stores one record (one scan, one tool) in this browser (IndexedDB, origin-wide, shared by every page of the site). **Download this scan (.zip)** gives `record.csv`, `data_dictionary.csv`, `provenance.json`, `fhir_bundle.json`, the label map (`labels.nii.gz`, no patient metadata) and a README.
- **Research Dataset** (`research.html`, the calculator's *Research Dataset* tab, `#research`) lists the records, takes outcomes when known (DCI, mRS at 90 days, in-hospital death), and exports the dataset as CSV + data dictionary (NIH ScHARe, R, Python), Parquet (typed; Terra/BigQuery, pandas, DuckDB), REDCap data dictionary + records, FHIR R4 collection bundle (Patient = pseudonym, Device = model, Observations; local code system), OMOP-style MEASUREMENT rows (concept ids mapped by the site; 8587 = millilitre), JSON Lines (e.g. `npx convex import`), a re-importable backup, or all of them in one zip. Import merges files from another computer at the site.
- **Schema** `heads-up-research-1.0`: 67 variables in groups (record and site, imaging, analysis, AI volumes, manual ABC/2, reference comparison, HEADS-UP detection, ICH expansion signs, optional clinical, outcomes). Clinical and outcome variables follow the NINDS UIA & SAH common data elements (Suarez et al., *Neurocrit Care* 2019;30(Suppl 1):4–19); the dictionary names each variable's CDE concept. Race and ethnicity use the US OMB SPD 15 (2024) categories and are optional.
- **De-identification at the source.** The study ID is turned into a pseudonym on the computer (site code + HMAC-SHA-256 with a random site key kept in the browser; the same key on every computer at a site links serial scans); dates become hours from onset; age is whole years capped at 90; DICOM age, sex, scanner and dates are read only to pre-fill the form and are kept out of the JSON reports; file names, series descriptions and exact timestamps are scrubbed from provenance; no images, names, MRNs or free text are exported. Sites collect and share under their own approvals.
- Parquet is written in the browser with [hyparquet-writer](https://github.com/hyparam/hyparquet-writer) 0.18 (MIT; `vendor/hyparquet-writer/`); the zip writer, CSV, REDCap, FHIR and OMOP writers are in `lib/research.js`.
- Tested (headless Chromium): records from BLAST-CT, SAHVAI, the ABC/2 calculator, HEADS-UP, ICH Expansion Signs and 3D-4D; all exports open (zip integrity, Parquet read back with hyparquet, FHIR JSON, REDCap dictionary columns), outcome edits persist, and the `#research` link opens the tab.

## Requirements

Recent desktop Chrome or Edge (tested in Chromium); Firefox and Safari may work on the CPU. 8 GB RAM or more. Measured on the demo case (2 windows of 18 slices), Intel Mac with 8 threads: **43 s on WebGPU, 118 s on the CPU**, identical volumes. Multi-threaded CPU needs the page to be cross-origin isolated, which `coi-serviceworker.js` provides when the page is opened on its own; embedded as a tab of the calculator it runs single-threaded on the CPU (WebGPU is unaffected).

## Running locally and deploying

`python3 serve.py` in this folder, then open `http://localhost:8000/` (any static server works; `serve.py` adds the headers for multi-threaded CPU). For hosting inside an institution, PACS options and governance notes see **[DEPLOY.md](DEPLOY.md)**.

## Licence and attribution

See [LICENSE.md](LICENSE.md): app code CC BY-NC-SA 4.0 (adapted from radar-web); BrainHemoAI code and weights MIT, used with the permission of Dr Ping Hu and the BrainHemoAI group, to whom we are grateful; CLAIM weights and demo data CC BY-NC 4.0; BLAST-CT Apache-2.0 (Copyright 2020 Imperial College London); DeepBleed CC BY-NC 4.0 for this browser version, with the permission of Dr Matthew Sharrock (published CC BY-NC-ND 4.0; weights unchanged) and its MISTIE/CLEAR demo CT CC BY-NC 4.0; SAH Swin UNETR academic non-commercial licence, hosted with the permission of Dr Santiago Cepeda (weights unchanged); RHUH dataset CC BY 4.0 (cases 001–003 hosted as demos). Please cite:

- Hu P, Zhou H, Yan T, et al. Deep learning-assisted identification and quantification of aneurysmal subarachnoid hemorrhage in non-contrast CT scans: development and external validation of Hybrid 2D/3D UNet. *NeuroImage* 2023;279:120321. doi:10.1016/j.neuroimage.2023.120321
- Kiewitz J, Aydin OU, Hilbert A, Gultom M, Nouri A, Khalil AA, Vajkoczy P, Tanioka S, Ishida F, Dengler NF, Frey D. Deep learning-based multiclass segmentation in aneurysmal subarachnoid hemorrhage. *Front Neurol* 2024;15:1490216. doi:10.3389/fneur.2024.1490216
- Monteiro M, Newcombe VFJ, Mathieu F, Adatia K, Kamnitsas K, Ferrante E, Das T, Whitehouse D, Rueckert D, Menon DK, Glocker B. Multiclass semantic segmentation and quantification of traumatic brain injury lesions on head CT using deep learning: an algorithm development and multicentre validation study. *Lancet Digit Health* 2020;2(6):e314–e322. doi:10.1016/S2589-7500(20)30085-6
- Sharrock MF, Mould WA, Ali H, Hildreth M, Awad IA, Hanley DF, Muschelli J. 3D deep neural network segmentation of intracerebral hemorrhage: development and validation for clinical trials. *Neuroinformatics* 2021;19(3):403–415. doi:10.1007/s12021-020-09493-5
- García-García S, Cepeda S, Arrese I, Sarabia R. A fully automated pipeline using Swin transformers for deep learning-based blood segmentation on head computed tomography scans after aneurysmal subarachnoid hemorrhage. World Neurosurg 2024. doi:10.1016/j.wneu.2024.07.216 (SAH Swin UNETR)
- García-García S, Cepeda S, Müller D, et al. Mortality prediction of patients with subarachnoid hemorrhage using a deep learning model based on an initial brain CT scan. Brain Sci 2024;14(1):10. doi:10.3390/brainsci14010010
- Hanley D, Mould WA, Ullman N, Muschelli J. Head CT scans associated with "Validated automatic brain extraction of head CT images". Johns Hopkins Research Data Repository; 2019. doi:10.7281/T1/CZDPSX (DeepBleed demo case)
- Cepeda S, García-García S. Non-contrast CT scans dataset for subarachnoid hemorrhage: bleeding segmentation. Zenodo; 2023. doi:10.5281/zenodo.8228847
- Wu B, et al. BHSD: a 3D multi-class brain hemorrhage segmentation dataset. MLMI 2023, LNCS 14348. doi:10.1007/978-3-031-45673-2_15 (CLAIM demo cases)
- Zhang Q, Zhang J, …, Liang T. An expert-level generalist AI for abdominal CT diagnosis. *Science* 2026;393(6817):eaec6129. doi:10.1126/science.aec6129 (RADAR; the browser architecture of this port follows radar-web)
