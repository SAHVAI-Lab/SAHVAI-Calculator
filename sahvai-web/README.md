# SAHVAI Web: hemorrhage segmentation models in the browser

**https://sahvai-lab.github.io/SAHVAI-Calculator/sahvai-web/** (also the *SAHVAI Web (AI)* tab of the [SAHVAI Calculator](https://sahvai-lab.github.io/SAHVAI-Calculator/))

Six pages, each its own tab in the SAHVAI Calculator, all running entirely in the browser with ONNX Runtime Web (WebGPU, or the CPU via multi-threaded WebAssembly). Each has drag-and-drop for DICOM / NIfTI / NRRD, a demo case, an axial viewer with per-slice volumes, downloads, and its citation, licence and disclaimer. The SAHVAI page (`index.html`, `app.js`, `worker.js`) is the original BrainHemoAI tool and loads only that model; the SAHVAI 3D-4D page (`3d.html`, `app-3d.js`) reuses its `worker.js`; the other four (`claim.html`, `blast.html`, `deepbleed.html`, `rhuh.html`) share `app-models.js`, `worker-models.js` and `style-models.css`, and each loads only its own model.

| page | model / data | from | classes | licence |
|---|---|---|---|---|
| SAHVAI (`index.html`) | Hybrid 2D/3D UNet, BrainHemoAI weights (Hu et al., *NeuroImage* 2023) | BrainHemoAI | SAH, IPH, IVH | MIT (with permission) |
| CLAIM aSAH (`claim.html`) | nnU-Net v2 2D multiclass aSAH model (Kiewitz et al., *Front Neurol* 2024) | CLAIM, Charité Berlin | SAH, IVH, ventricles, ICH, aneurysm, SDH | CC BY-NC 4.0 |
| BLAST-CT (`blast.html`) | DeepMedic, 1 or 12 models (Monteiro et al., *Lancet Digit Health* 2020) | Cambridge / Imperial College London | IPH, extra-axial haemorrhage, oedema, IVH (traumatic brain injury) | Apache-2.0 |
| DeepBleed ICH (`deepbleed.html`) | 3D V-Net, MISTIE-2 weights (Sharrock et al., *Neuroinformatics* 2021) | MISTIE investigators, Johns Hopkins | ICH + IVH (one class) | CC BY-NC-ND 4.0 |
| SAHVAI 3D-4D (`3d.html`) | the SAHVAI model above, plus 3D surfaces (three.js) and volumes over serial scans | SAHVAI Lab | SAH, IPH, IVH in 3D; volume vs date | MIT (with permission) |
| RHUH dataset (`rhuh.html`) | open aSAH NCCT scans with expert bleeding masks (Cepeda & García-García, Zenodo 2023) | Río Hortega University Hospital, Valladolid | expert mask vs any of the three models (Dice, volumes) | CC BY 4.0 (data not hosted here) |

The design follows [RADAR in the browser](https://github.com/jarrelscy/radar-web), the browser port of Alibaba DAMO Academy's RADAR abdominal CT model (Zhang, Ding et al., *Science* 2026).

Images never leave the computer: files are read and processed in the browser tab and nothing is uploaded. Models are served from this site and cached by the browser (SAHVAI 26.5 MB, BLAST-CT 2.1 MB per model, CLAIM 92.7 MB and DeepBleed 90.2 MB, each in two parts).

> **Research and education use only.** This is not a medical device. None of these models has been validated or approved for clinical use in this form; do not use them for diagnosis, triage or treatment decisions. Always review the segmentation against the images.

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

[msharrock/deepbleed](https://github.com/msharrock/deepbleed) (Sharrock et al., *Neuroinformatics* 2021): a 3D V-Net (22.5 M parameters; each block's 5×5×5 convolution is one layer applied 1–3 times) trained on 100 MISTIE II scans and validated on 500 MISTIE III scans (Dice 0.91), segmenting ICH and IVH as one class. External validation in 1,040 patients: median Dice 0.84 (J Clin Med 2023). Licence CC BY-NC-ND 4.0: the weights are converted to ONNX in full precision without change (`tools/deepbleed_export.py`; the 2×2×2 transposed convolutions are written as 1×1×1 convolutions plus an interleave, identical arithmetic, because ONNX Runtime's WebGPU backend has no 3D transposed convolution); see LICENSE.md.

`predict.py` does: FSL BET brain extraction → ANTs rigid registration (Mattes MI) of the brain to a 1.5 mm, 128³ CT template → V-Net → mask > 0.5 → inverse transform (genericLabel) to the scan. `lib/deepbleed.js` reproduces the geometry; brain extraction is SAHVAI's (`lib/brain.js`), since FSL cannot run in a browser (for a skull-stripped scan, its own non-zero voxels). One quirk is kept on purpose: `predict.py`'s `nii2ants()` passes the NIfTI qform (RAS numbers) to ANTs, which reads them as LPS, so the brain reaches the registration turned 180° about the vertical axis, and ANTs, starting from the centres of mass with no rotation, keeps it so. The network was trained on inputs made this way, and it predicts well only on them. Registration here: centres of mass (as ANTs' initialiser), then a mutual-information refinement (64 bins, rotations within ±0.1 rad, shrink 4 and 2), optional.

Validation (MISTIE demo = JHU case 01; references run with `ants.registration` exactly as `predict.py`, with a given brain mask since FSL is not installable here):

| check | result |
|---|---|
| ONNX vs a TensorFlow reference from the same checkpoint (128³ test volume) | max probability difference 4e-5; 0 voxels differ |
| resampling with ANTs' own transform vs ANTs' `warpedmovout` | correlation 0.99994; inverse mapping vs `apply_transforms` Dice 1.000 |
| browser registration vs ANTs, same brain mask, native-space Dice | JHU-01 0.93 (15.2 vs 15.1 mL); BLAST scan_0 0.94; BHSD SAH_002 0.95 (Python prototype of the same method) |
| JHU-01, full browser pipeline (SAHVAI brain extraction; mask Dice 0.989 vs the manual brain mask) | 23.9 mL: the additional 8.7 mL is intraventricular blood higher in the lateral ventricle, beside the drain, that the reference run did not label |
| BHSD SAH_002, full browser pipeline, vs expert ICH + IVH | 94.0 vs 98.0 mL, Dice 0.870 |

DeepBleed is sensitive to small changes in its input (brain mask, placement in the template): expect volume differences of several mL between runs of the original and this port on the same scan.

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

## Requirements

Recent desktop Chrome or Edge (tested in Chromium); Firefox and Safari may work on the CPU. 8 GB RAM or more. Measured on the demo case (2 windows of 18 slices), Intel Mac with 8 threads: **43 s on WebGPU, 118 s on the CPU**, identical volumes. Multi-threaded CPU needs the page to be cross-origin isolated, which `coi-serviceworker.js` provides when the page is opened on its own; embedded as a tab of the calculator it runs single-threaded on the CPU (WebGPU is unaffected).

## Running locally and deploying

`python3 serve.py` in this folder, then open `http://localhost:8000/` (any static server works; `serve.py` adds the headers for multi-threaded CPU). For hosting inside an institution, PACS options and governance notes see **[DEPLOY.md](DEPLOY.md)**.

## Licence and attribution

See [LICENSE.md](LICENSE.md): app code CC BY-NC-SA 4.0 (adapted from radar-web); BrainHemoAI code and weights MIT, used with the permission of Dr Ping Hu and the BrainHemoAI group, to whom we are grateful; CLAIM weights and demo data CC BY-NC 4.0; BLAST-CT Apache-2.0 (Copyright 2020 Imperial College London); DeepBleed CC BY-NC-ND 4.0 (weights unchanged) and its MISTIE/CLEAR demo CT CC BY-NC 4.0; RHUH dataset CC BY 4.0 (not redistributed). Please cite:

- Hu P, Zhou H, Yan T, et al. Deep learning-assisted identification and quantification of aneurysmal subarachnoid hemorrhage in non-contrast CT scans: development and external validation of Hybrid 2D/3D UNet. *NeuroImage* 2023;279:120321. doi:10.1016/j.neuroimage.2023.120321
- Kiewitz J, Aydin OU, Hilbert A, Gultom M, Nouri A, Khalil AA, Vajkoczy P, Tanioka S, Ishida F, Dengler NF, Frey D. Deep learning-based multiclass segmentation in aneurysmal subarachnoid hemorrhage. *Front Neurol* 2024;15:1490216. doi:10.3389/fneur.2024.1490216
- Monteiro M, Newcombe VFJ, Mathieu F, Adatia K, Kamnitsas K, Ferrante E, Das T, Whitehouse D, Rueckert D, Menon DK, Glocker B. Multiclass semantic segmentation and quantification of traumatic brain injury lesions on head CT using deep learning: an algorithm development and multicentre validation study. *Lancet Digit Health* 2020;2(6):e314–e322. doi:10.1016/S2589-7500(20)30085-6
- Sharrock MF, Mould WA, Ali H, Hildreth M, Awad IA, Hanley DF, Muschelli J. 3D deep neural network segmentation of intracerebral hemorrhage: development and validation for clinical trials. *Neuroinformatics* 2021;19(3):403–415. doi:10.1007/s12021-020-09493-5
- Hanley D, Mould WA, Ullman N, Muschelli J. Head CT scans associated with "Validated automatic brain extraction of head CT images". Johns Hopkins Research Data Repository; 2019. doi:10.7281/T1/CZDPSX (DeepBleed demo case)
- Cepeda S, García-García S. Non-contrast CT scans dataset for subarachnoid hemorrhage: bleeding segmentation. Zenodo; 2023. doi:10.5281/zenodo.8228847
- Wu B, et al. BHSD: a 3D multi-class brain hemorrhage segmentation dataset. MLMI 2023, LNCS 14348. doi:10.1007/978-3-031-45673-2_15 (CLAIM demo cases)
- Zhang Q, Zhang J, …, Liang T. An expert-level generalist AI for abdominal CT diagnosis. *Science* 2026;393(6817):eaec6129. doi:10.1126/science.aec6129 (RADAR; the browser architecture of this port follows radar-web)
