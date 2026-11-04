# Licences

**Research and education use only. Not a medical device.** Provided as is, without warranty of any kind.

SAHVAI Web runs three independently licensed models and links to one open dataset. Each keeps its own licence, summarised below; the non-commercial terms of CLAIM (CC BY-NC 4.0) and of this app's code (CC BY-NC-SA 4.0) mean the page as a whole is for non-commercial research and education.

## Code in this folder

Parts of this web app are adapted from [radar-web](https://github.com/jarrelscy/radar-web) (`lib/source.js`, `lib/dicom.js`, `lib/nifti.js`, `lib/series.js`, `coi-serviceworker.js` changes, and the structure of `app.js` / `worker.js`), which is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence](https://creativecommons.org/licenses/by-nc-sa/4.0/). As required by that licence, the code in this folder (`sahvai-web/`) is released under **CC BY-NC-SA 4.0** as well. The rest of the SAHVAI Calculator repository is not affected.

The pre- and post-processing in `lib/hemo.js` is a port of the Python code in [BrainHemoAI](https://github.com/BrainHemo/BrainHemoAI) (MIT License, see below).

## Model weights (`models/`) and ported processing code

**SAHVAI page.** The ONNX files in `models/` (top level) are a format conversion of `ml/pth/tri_hybrid_unet.pth` from [BrainHemoAI](https://github.com/BrainHemo/BrainHemoAI) (Hu P, Zhou H, Yan T, et al., *NeuroImage* 2023;279:120321), and `lib/hemo.js` ports its pre- and post-processing. Dr Ping Hu of the BrainHemoAI group has given permission for this use under the MIT License below (October 2026).

### BrainHemoAI MIT License

MIT License

Copyright (c) 2023 the BrainHemoAI authors (Ping Hu, Haizhu Zhou, Tengfeng Yan, et al.)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


### CLAIM aSAH multiclass segmentation

**CLAIM page.** `models/claim/` is a format conversion to ONNX (fp16, split into parts for hosting) of the nnU-Net v2 "2d" checkpoint `2d_checkpoint_best.pth` published by the CLAIM group (Charité Lab for Artificial Intelligence in Medicine, Charité – Universitätsmedizin Berlin) at [Zenodo 10.5281/zenodo.12579491](https://doi.org/10.5281/zenodo.12579491), with the plans from [claim-berlin/aSAH-multiclass-segmentation](https://github.com/claim-berlin/aSAH-multiclass-segmentation). The weights are not retrained or otherwise changed. `lib/claim.js` re-implements nnU-Net's inference steps for this configuration.

Licence: [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) (full text in [licenses/CC-BY-NC-4.0-claim.md](licenses/CC-BY-NC-4.0-claim.md)). Attribution: Kiewitz J, Aydin OU, Hilbert A, Gultom M, Nouri A, Khalil AA, Vajkoczy P, Tanioka S, Ishida F, Dengler NF, Frey D. Deep learning-based multiclass segmentation in aneurysmal subarachnoid hemorrhage. *Front Neurol* 2024;15:1490216. doi:10.3389/fneur.2024.1490216. Changes: converted from PyTorch to ONNX and to half precision. Non-commercial use only.

`demo/claim/` holds the three example NCCT volumes and reference labels distributed in the CLAIM repository (same licence), which come from the BHSD dataset: Wu B, et al. BHSD: a 3D multi-class brain hemorrhage segmentation dataset. MLMI 2023, LNCS 14348, Springer 2024. doi:10.1007/978-3-031-45673-2_15. The reference files are renamed `SAH_00k_gt.nii.gz`; their labels follow BHSD (1 EDH, 2 IPH, 3 IVH, 4 SAH, 5 SDH).

nnU-Net itself (Isensee F, et al. *Nat Methods* 2021;18:203–211) is Apache-2.0; no nnU-Net code is bundled.

### BLAST-CT (Apache-2.0)

**BLAST-CT page.** `models/blast/blast_m1.onnx` … `blast_m12.onnx` are format conversions (ONNX, fp16) of the 12 pre-trained DeepMedic models `blast_ct/data/saved_models/model_1.torch_model` … `model_12.torch_model` (the 12 that `blast-ct --ensemble` uses; the single-model default is model 1) from [biomedia-mira/blast-ct](https://github.com/biomedia-mira/blast-ct) (commit 613acbc), and `lib/blast.js` re-implements its inference (resampling, windowing, tiling, ensembling). `demo/blast/` holds `scan_0_image.nii.gz` and `scan_0_segmentation.nii.gz` from the same repository.

Copyright 2020 Imperial College London. Licensed under the Apache License, Version 2.0 (full text in [licenses/Apache-2.0-blast-ct.txt](licenses/Apache-2.0-blast-ct.txt)); you may not use these files except in compliance with the License. Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND. **Modifications:** the PyTorch models were re-expressed with static shapes and exported to ONNX, then converted to half precision (outputs checked against the original: probabilities within 2.3e-6 in fp32; label maps within 2 voxels of the blast-ct CLI on `scan_0`). Citation: Monteiro M, Newcombe VFJ, Mathieu F, et al. *Lancet Digit Health* 2020;2(6):e314–e322. doi:10.1016/S2589-7500(20)30085-6. The blast-ct authors state that the software is not intended for clinical use.

### DeepBleed (CC BY-NC 4.0)

**DeepBleed page.** `models/deepbleed/deepbleed.onnx.0` and `.1` are the MISTIE-2 weights of DeepBleed (TensorFlow checkpoint linked from the README of [msharrock/deepbleed](https://github.com/msharrock/deepbleed), commit 4467eaa) in ONNX format, full precision, split into two parts for hosting. The network is re-expressed operation for operation and the weights are copied unchanged: the ONNX model reproduces a TensorFlow reference built from the same checkpoint (maximum probability difference 4e-5, 0 voxels different on a 128³ test volume). `models/deepbleed/scct_unsmooth_SS_0.01_128x128x128.nii.gz` is DeepBleed's CT template, unchanged. `lib/deepbleed.js` re-implements the geometry of its `predict.py` (registration, resampling, inverse mapping); no DeepBleed code is included.

Licence: DeepBleed is published under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (full text, including the authors' note "THIS SOFTWARE IS NOT INTENDED FOR CLINICAL USE", in [licenses/CC-BY-NC-ND-4.0-deepbleed.md](licenses/CC-BY-NC-ND-4.0-deepbleed.md)). For this browser version, **Dr Matthew Sharrock gave written permission (email to Dr W. David Freeman, November 2026) to host it and to drop the no-derivatives condition**, now that it runs in the browser and has been compared with the original; it is therefore used and shared here under [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) (full text in [licenses/CC-BY-NC-4.0-deepbleed.md](licenses/CC-BY-NC-4.0-deepbleed.md)). Non-commercial use only; not intended for clinical use. The weights themselves are unchanged (ONNX format, full precision). Attribution: Sharrock MF, Mould WA, Ali H, Hildreth M, Awad IA, Hanley DF, Muschelli J. 3D deep neural network segmentation of intracerebral hemorrhage: development and validation for clinical trials. *Neuroinformatics* 2021;19(3):403–415. doi:10.1007/s12021-020-09493-5.

`demo/deepbleed/jhu_mistie_01_ct.nii.gz` is `01/BRAIN_1_Anonymized.nii.gz`, unchanged, from Hanley D, Mould WA, Ullman N, Muschelli J. Head CT scans associated with "Validated automatic brain extraction of head CT images". Johns Hopkins Research Data Repository, V1, 2019. doi:10.7281/T1/CZDPSX (MISTIE II/III and CLEAR III), [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

### ICH expansion signs page

`hesigns.html`, `app-hesigns.js` and `lib/hesigns.js` are SAHVAI Lab code under the app licence above and use no third-party model weights (the segmentation is the SAHVAI / BrainHemoAI model). The region-of-interest step of the DeepMedX sICH_HE pipeline (Yu Q, et al. *npj Digit Med* 2026;9:39; [github.com/DeepMedX/sICH_HE](https://github.com/DeepMedX/sICH_HE), non-commercial research licence) is re-implemented from its description and notebook, with attribution; no DeepMedX code or weights are copied. The sign schematics were drawn for this page.

**Yale expansion model (not distributed).** `worker-yale.js`, `lib/yalehe.js` and `tools/yale_export.py` load the ONNX conversion of the checkpoints published at [github.com/anhtrnyaleedu/HE](https://github.com/anhtrnyaleedu/HE) (Tran et al., *npj Digit Med* 2024). That repository has no licence file, so the converted weights (`models/yale/`) are not committed or hosted until the authors give written permission; `MyDenseNet.py` there is derived from MONAI (Apache-2.0).

### SAH Swin UNETR (Valladolid)

**SAH Swin UNETR page.** `models/sahswin/sahswin.onnx.0`–`.2` are the weights of `SWIN_UNETR_48_100_best_metric_model.pth`, linked from the README of [smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction](https://github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction), in ONNX format, full precision and unchanged (ONNX output matches PyTorch to 1.4e-5, 0 voxels different), split into three parts for hosting. `models/sahswin/ct_template2mni_2mm.nii.gz` is the repository's `ct_template2mni.nii.gz`, smoothed and subsampled to 2 mm for registration. `lib/sahswin.js` re-implements the steps of its `inference_2.py` (MONAI) and the geometry of its preprocessing; no code from the repository is included.

Licence: the repository's **Academic Non-Commercial Source Code License Agreement**, © 2024 Hospital Río Hortega and The University of Augsburg (full text in [licenses/Academic-NonCommercial-sah-swin-unetr.txt](licenses/Academic-NonCommercial-sah-swin-unetr.txt)); the repository README also states CC BY-NC. Academic, non-commercial research use only; not for clinical decision-making; no commercial use by or for any commercial entity. The converted weights and template are hosted here with the written permission of Dr Santiago Cepeda (email to Dr W. David Freeman, 1 November 2026), on condition of attribution as published. Attribution: García-García S, Cepeda S, Arrese I, Sarabia R. *World Neurosurg* 2024, doi:10.1016/j.wneu.2024.07.216; García-García S, Cepeda S, Müller D, et al. *Brain Sci* 2024;14(1):10, doi:10.3390/brainsci14010010.

### HEADS-UP

**HEADS-UP page.** `models/headsup/headsup_dn121.onnx`, `headsup_dn169.onnx` and `headsup_sx101.onnx.0`–`.1` are the fold-0 checkpoints of the three 2D CNNs (DenseNet121 at 512 px, DenseNet169 at 256 px, SE-ResNeXt101 at 256 px) of the 1st-place solution of the RSNA 2019 Intracranial Hemorrhage Detection challenge, published by Tao Shen and colleagues at [SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection) (checkpoints linked from its README). They are converted to ONNX with the weights unchanged and stored as 16-bit numbers (computed in 32-bit); a second output, the class activation map (the final linear layer applied before pooling, which averages exactly to the original logits), is added for localisation. `lib/headsup.js` re-implements the solution's `prepare_data.py` / `predict.py` preprocessing; no code from the repository is included. `tools/headsup_export.py` rebuilds the files.

Licence: the repository carries no licence file. The converted weights are hosted here with the **written permission of Tao Shen** (email to Dr W. David Freeman, 1 November 2026), which covers deploying and hosting the ONNX-converted fold-0 DenseNet121 512, DenseNet169 256 and SE-ResNeXt101 256 models in this open, non-commercial tool suite, with attribution to the paper and repository. Research and education use only. Attribution: Wang X, Shen T, Yang S, Lan J, Xu Y, Wang M, Zhang J, Han X. A deep learning algorithm for automatic detection and classification of acute intracranial hemorrhages in head CT scans. *NeuroImage Clin* 2021;32:102785, doi:10.1016/j.nicl.2021.102785.

The networks were trained on the **RSNA Brain CT Hemorrhage Dataset** (Copyright RSNA, 2019), which RSNA licenses for non-commercial purposes only, including academic research and education; cite Flanders AE, et al. *Radiol Artif Intell* 2020;2(3):e190211, doi:10.1148/ryai.2020190211. HEADS-UP itself: Salman S, Gu Q, Dherin B, et al. *Mayo Clin Proc Digit Health* 2023;1(4):547–556, doi:10.1016/j.mcpdig.2023.08.009.

### RHUH dataset

**The RHUH dataset page.** `demo/rhuh/` holds three cases (sah_rhuh_001–003: CT and expert mask, unchanged) from the RHUH dataset; the other 97 are downloaded by the user from [Zenodo 10.5281/zenodo.8228847](https://doi.org/10.5281/zenodo.8228847) and opened locally. The dataset is © Santiago Cepeda and Sergio García-García (Río Hortega University Hospital, Valladolid) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); cite: Cepeda S, García-García S. Non-contrast CT scans dataset for subarachnoid hemorrhage: bleeding segmentation. Zenodo; 2023. doi:10.5281/zenodo.8228847. The authors' own segmentation and mortality code ([smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction](https://github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction)) is under an Academic Non-Commercial Source Code License; its Swin UNETR segmentation weights run on the SAH Swin UNETR page (above), and no code from it is included.

## Demo cases (`demo/`)

`brainhemoai-demo-head.nrrd` is the example scan (`nrrd/head.nrrd`) distributed with BrainHemoAI.

## Bundled libraries (`vendor/`)

ONNX Runtime Web (MIT), jpeg-lossless-decoder-js (MIT), Cornerstone codecs CharLS / OpenJPEG / OpenJPH (MIT), coi-serviceworker (MIT), three.js (MIT; 3D-4D page), hyparquet-writer and hyparquet (MIT; Parquet export of the Research Dataset). See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
