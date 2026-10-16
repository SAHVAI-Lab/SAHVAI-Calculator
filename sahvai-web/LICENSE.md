# Licences

**Research use only. Not a medical device.** Provided as is, without warranty of any kind.

## Code in this folder

Parts of this web app are adapted from [radar-web](https://github.com/jarrelscy/radar-web) (`lib/source.js`, `lib/dicom.js`, `lib/nifti.js`, `lib/series.js`, `coi-serviceworker.js` changes, and the structure of `app.js` / `worker.js`), which is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence](https://creativecommons.org/licenses/by-nc-sa/4.0/). As required by that licence, the code in this folder (`sahvai-web/`) is released under **CC BY-NC-SA 4.0** as well. The rest of the SAHVAI Calculator repository is not affected.

The pre- and post-processing in `lib/hemo.js` is a port of the Python code in [BrainHemoAI](https://github.com/BrainHemo/BrainHemoAI) (MIT License, see below).

## Model weights (`models/`) and ported processing code

The ONNX files are a format conversion of `ml/pth/tri_hybrid_unet.pth` from [BrainHemoAI](https://github.com/BrainHemo/BrainHemoAI) (Hu P, Zhou H, Yan T, et al., *NeuroImage* 2023;279:120321), and `lib/hemo.js` ports its pre- and post-processing. Dr Ping Hu of the BrainHemoAI group has given permission for this use under the MIT License below (October 2026).

### BrainHemoAI MIT License

MIT License

Copyright (c) 2023 the BrainHemoAI authors (Ping Hu, Haizhu Zhou, Tengfeng Yan, et al.)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Demo case (`demo/`)

`brainhemoai-demo-head.nrrd` is the example scan (`nrrd/head.nrrd`) distributed with BrainHemoAI.

## Bundled libraries (`vendor/`)

ONNX Runtime Web (MIT), jpeg-lossless-decoder-js (MIT), Cornerstone codecs CharLS / OpenJPEG / OpenJPH (MIT), coi-serviceworker (MIT). See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
