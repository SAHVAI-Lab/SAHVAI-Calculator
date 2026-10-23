# Licences

**Research and education use only. Not a medical device.** Provided as is, without warranty of any kind.

SAHVAI Web runs three independently licensed models and links to one open dataset. Each keeps its own licence, summarised below; the non-commercial terms of CLAIM (CC BY-NC 4.0) and of this app's code (CC BY-NC-SA 4.0) mean the page as a whole is for non-commercial research and education.

## Code in this folder

Parts of this web app are adapted from [radar-web](https://github.com/jarrelscy/radar-web) (`lib/source.js`, `lib/dicom.js`, `lib/nifti.js`, `lib/series.js`, `coi-serviceworker.js` changes, and the structure of `app.js` / `worker.js`), which is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence](https://creativecommons.org/licenses/by-nc-sa/4.0/). As required by that licence, the code in this folder (`sahvai-web/`) is released under **CC BY-NC-SA 4.0** as well. The rest of the SAHVAI Calculator repository is not affected.

The pre- and post-processing in `lib/hemo.js` is a port of the Python code in [BrainHemoAI](https://github.com/BrainHemo/BrainHemoAI) (MIT License, see below).

## Model weights (`models/`) and ported processing code

**Panel 1, SAHVAI.** The ONNX files in `models/` (top level) are a format conversion of `ml/pth/tri_hybrid_unet.pth` from [BrainHemoAI](https://github.com/BrainHemo/BrainHemoAI) (Hu P, Zhou H, Yan T, et al., *NeuroImage* 2023;279:120321), and `lib/hemo.js` ports its pre- and post-processing. Dr Ping Hu of the BrainHemoAI group has given permission for this use under the MIT License below (October 2026).

### BrainHemoAI MIT License

MIT License

Copyright (c) 2023 the BrainHemoAI authors (Ping Hu, Haizhu Zhou, Tengfeng Yan, et al.)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


### CLAIM aSAH multiclass segmentation

**Panel 2.** `models/claim/` is a format conversion to ONNX (fp16, split into parts for hosting) of the nnU-Net v2 "2d" checkpoint `2d_checkpoint_best.pth` published by the CLAIM group (Charité Lab for Artificial Intelligence in Medicine, Charité – Universitätsmedizin Berlin) at [Zenodo 10.5281/zenodo.12579491](https://doi.org/10.5281/zenodo.12579491), with the plans from [claim-berlin/aSAH-multiclass-segmentation](https://github.com/claim-berlin/aSAH-multiclass-segmentation). The weights are not retrained or otherwise changed. `lib/claim.js` re-implements nnU-Net's inference steps for this configuration.

Licence: [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) (full text in [licenses/CC-BY-NC-4.0-claim.md](licenses/CC-BY-NC-4.0-claim.md)). Attribution: Kiewitz J, Aydin OU, Hilbert A, Gultom M, Nouri A, Khalil AA, Vajkoczy P, Tanioka S, Ishida F, Dengler NF, Frey D. Deep learning-based multiclass segmentation in aneurysmal subarachnoid hemorrhage. *Front Neurol* 2024;15:1490216. doi:10.3389/fneur.2024.1490216. Changes: converted from PyTorch to ONNX and to half precision. Non-commercial use only.

`demo/claim/` holds the three example NCCT volumes and reference labels distributed in the CLAIM repository (same licence), which come from the BHSD dataset: Wu B, et al. BHSD: a 3D multi-class brain hemorrhage segmentation dataset. MLMI 2023, LNCS 14348, Springer 2024. doi:10.1007/978-3-031-45673-2_15. The reference files are renamed `SAH_00k_gt.nii.gz`; their labels follow BHSD (1 EDH, 2 IPH, 3 IVH, 4 SAH, 5 SDH).

nnU-Net itself (Isensee F, et al. *Nat Methods* 2021;18:203–211) is Apache-2.0; no nnU-Net code is bundled.

### BLAST-CT (Apache-2.0)

**Panel 3.** `models/blast/blast_m1.onnx` … `blast_m12.onnx` are format conversions (ONNX, fp16) of the 12 pre-trained DeepMedic models `blast_ct/data/saved_models/model_1.torch_model` … `model_12.torch_model` (the 12 that `blast-ct --ensemble` uses; the single-model default is model 1) from [biomedia-mira/blast-ct](https://github.com/biomedia-mira/blast-ct) (commit 613acbc), and `lib/blast.js` re-implements its inference (resampling, windowing, tiling, ensembling). `demo/blast/` holds `scan_0_image.nii.gz` and `scan_0_segmentation.nii.gz` from the same repository.

Copyright 2020 Imperial College London. Licensed under the Apache License, Version 2.0 (full text in [licenses/Apache-2.0-blast-ct.txt](licenses/Apache-2.0-blast-ct.txt)); you may not use these files except in compliance with the License. Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND. **Modifications:** the PyTorch models were re-expressed with static shapes and exported to ONNX, then converted to half precision (outputs checked against the original: probabilities within 2.3e-6 in fp32; label maps within 2 voxels of the blast-ct CLI on `scan_0`). Citation: Monteiro M, Newcombe VFJ, Mathieu F, et al. *Lancet Digit Health* 2020;2(6):e314–e322. doi:10.1016/S2589-7500(20)30085-6. The blast-ct authors state that the software is not intended for clinical use.

### RHUH dataset

**Panel 4** does not host any RHUH data: the user downloads it from [Zenodo 10.5281/zenodo.8228847](https://doi.org/10.5281/zenodo.8228847) and opens it locally. The dataset is © Santiago Cepeda and Sergio García-García (Río Hortega University Hospital, Valladolid) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); cite: Cepeda S, García-García S. Non-contrast CT scans dataset for subarachnoid hemorrhage: bleeding segmentation. Zenodo; 2023. doi:10.5281/zenodo.8228847. The authors' own segmentation and mortality code ([smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction](https://github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction)) is under an Academic Non-Commercial Source Code License and is not included or run here.

## Demo cases (`demo/`)

`brainhemoai-demo-head.nrrd` is the example scan (`nrrd/head.nrrd`) distributed with BrainHemoAI.

## Bundled libraries (`vendor/`)

ONNX Runtime Web (MIT), jpeg-lossless-decoder-js (MIT), Cornerstone codecs CharLS / OpenJPEG / OpenJPH (MIT), coi-serviceworker (MIT). See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
