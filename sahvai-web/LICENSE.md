# Licences

**Research use only. Not a medical device.** Provided as is, without warranty of any kind.

## Code in this folder

Parts of this web app are adapted from [radar-web](https://github.com/jarrelscy/radar-web) (`lib/source.js`, `lib/dicom.js`, `lib/nifti.js`, `lib/series.js`, `coi-serviceworker.js` changes, and the structure of `app.js` / `worker.js`), which is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence](https://creativecommons.org/licenses/by-nc-sa/4.0/). As required by that licence, the code in this folder (`sahvai-web/`) is released under **CC BY-NC-SA 4.0** as well. The rest of the SAHVAI Calculator repository is not affected.

The pre- and post-processing in `lib/hemo.js` is a port of the Python code in [BrainHemoAI](https://github.com/BrainHemo/BrainHemoAI).

## Model weights (`models/`)

The ONNX files are a format conversion of `ml/pth/tri_hybrid_unet.pth` from [BrainHemoAI](https://github.com/BrainHemo/BrainHemoAI) (Hu P, Zhou H, Yan T, et al., *NeuroImage* 2023;279:120321). The weights remain the work of their authors; this conversion does not relicense them. At the time of conversion the BrainHemoAI repository did not state a licence, so please check with the authors before any use beyond academic research.

## Demo case (`demo/`)

`brainhemoai-demo-head.nrrd` is the example scan (`nrrd/head.nrrd`) distributed with BrainHemoAI.

## Bundled libraries (`vendor/`)

ONNX Runtime Web (MIT), jpeg-lossless-decoder-js (MIT), Cornerstone codecs CharLS / OpenJPEG / OpenJPH (MIT), coi-serviceworker (MIT). See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
