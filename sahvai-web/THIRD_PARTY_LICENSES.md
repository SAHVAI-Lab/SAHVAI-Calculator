# Third-Party Licenses

This folder bundles the following open-source software (in `vendor/`), taken from
[radar-web](https://github.com/jarrelscy/radar-web). Their notices are reproduced below.

---

## Bundled with this web port (vendor/)

- ONNX Runtime Web 1.30.0 (vendor/ort), MIT License, https://github.com/microsoft/onnxruntime
- hyparquet-writer 0.18.0 with hyparquet 1.32.0 (vendor/hyparquet-writer, bundled with esbuild), MIT License, https://github.com/hyparam/hyparquet-writer (licence texts in vendor/hyparquet-writer/)
- jpeg-lossless-decoder-js 2.1.2 (vendor/codecs/lossless.js), MIT License, https://github.com/rii-mango/JPEGLosslessDecoderJS
- @cornerstonejs/codec-charls 1.2.7, codec-openjpeg 1.3.6, codec-openjph 2.4.11 (vendor/codecs), MIT License, https://github.com/cornerstonejs
- coi-serviceworker (coi-serviceworker.js), MIT License, https://github.com/gzuidhof/coi-serviceworker
- three.js r186 (vendor/three/three-sahvai.min.js: core, WebGLRenderer and OrbitControls bundled and minified; marching-cubes tables in lib/mc-tables.js), MIT License, Copyright 2010-2026 three.js authors, https://github.com/mrdoob/three.js (vendor/three/LICENSE)

## Models and demo data (not software libraries)

- BrainHemoAI Hybrid 2D/3D UNet weights and demo scan (models/, demo/brainhemoai-demo-head.nrrd), MIT License, https://github.com/BrainHemo/BrainHemoAI
- BLAST-CT DeepMedic weights and scan_0 (models/blast/, demo/blast/), Apache License 2.0, Copyright 2020 Imperial College London, https://github.com/biomedia-mira/blast-ct (licenses/Apache-2.0-blast-ct.txt)
- CLAIM aSAH multiclass nnU-Net weights and BHSD demo cases (models/claim/, demo/claim/), CC BY-NC 4.0, https://github.com/claim-berlin/aSAH-multiclass-segmentation, https://doi.org/10.5281/zenodo.12579491 (licenses/CC-BY-NC-4.0-claim.md)
- SAH Swin UNETR weights and 2 mm registration template (models/sahswin/), Academic Non-Commercial Source Code License, © 2024 Hospital Río Hortega and University of Augsburg, https://github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction (licenses/Academic-NonCommercial-sah-swin-unetr.txt); hosted with the permission of Dr Santiago Cepeda, 1 November 2026
- DeepBleed V-Net weights and CT template (models/deepbleed/), CC BY-NC 4.0 for this browser version with the permission of Dr Matthew Sharrock, November 2026 (licenses/CC-BY-NC-4.0-deepbleed.md); published CC BY-NC-ND 4.0, https://github.com/msharrock/deepbleed (licenses/CC-BY-NC-ND-4.0-deepbleed.md); MISTIE/CLEAR demo CT (demo/deepbleed/), CC BY-NC 4.0, https://doi.org/10.7281/T1/CZDPSX
- HEADS-UP: RSNA 2019 Intracranial Hemorrhage Detection 1st-place 2D CNN weights, fold 0 (models/headsup/), https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection (no licence file; hosted for non-commercial research with attribution, with the written permission of Tao Shen, 1 November 2026); trained on the RSNA Brain CT Hemorrhage Dataset, Copyright RSNA 2019, non-commercial use only

See LICENSE.md for attribution and the changes made.
