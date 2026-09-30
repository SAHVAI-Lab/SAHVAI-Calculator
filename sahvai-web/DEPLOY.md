# Deploying SAHVAI Web inside an institution (e.g. Mayo Clinic)

Version 1.0 · 30 September 2026 · source: `sahvai-web/` in github.com/sahvai-lab/SAHVAI-Calculator

> **Research use only.** Not a medical device, not FDA-cleared, and validated on a single public case so far. Deploy for research under your institution's AI-governance and IRB processes; do not use results for patient care.

## 1. What is in the package

| path | contents |
|---|---|
| `index.html`, `app.js`, `worker.js`, `style.css` | the web app (static files, no build step) |
| `lib/` | DICOM / NIfTI / NRRD readers, resampling, brain extraction, UNet orchestration, post-processing |
| `models/` | the Hybrid 2D/3D UNet as nine ONNX parts (26.5 MB, float16 weights) |
| `vendor/` | ONNX Runtime Web 1.30 and DICOM image codecs (all local) |
| `coi-serviceworker.js` | enables multi-threading when the server cannot send the headers below |
| `demo/` | the public BrainHemoAI example scan |
| `tools/` | Python scripts that rebuild the ONNX parts from the original PyTorch weights |
| `serve.py` | a standard-library Python server with the right headers |

The app makes **no external network requests**: the model, runtime and codecs are all served from the same folder, so it works on an isolated intranet. Images are processed in the browser's memory and never leave the workstation.

## 2. Try it on one workstation

```
cd sahvai-web
python3 serve.py            # then open http://localhost:8000/ in Chrome or Edge
```

Click **Try the demo case**; expected result: SAH ≈ 77.6 mL, IPH 0 mL, IVH ≈ 2.7 mL.

## 3. Host it on an internal web server

It is a static site: copy the folder to any web server. Four requirements:

1. **HTTPS** (or `localhost`). WebGPU, the service worker and the model cache need a secure context.
2. **Cross-origin isolation headers** on every response, for multi-threaded CPU inference:
   `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.
   (Without them `coi-serviceworker.js` adds them on first load; with them it is not needed.)
3. **MIME types**: `.wasm` → `application/wasm`, `.mjs` / `.js` → `text/javascript`.
4. Allow files up to ~30 MB (`models/pm.onnx`, the runtime `.wasm`).

nginx:

```nginx
location /sahvai-web/ {
    alias /srv/sahvai-web/;
    add_header Cross-Origin-Opener-Policy same-origin always;
    add_header Cross-Origin-Embedder-Policy require-corp always;
    add_header Cross-Origin-Resource-Policy same-origin always;
    types { application/wasm wasm; text/javascript mjs js; }
}
```

Apache (`.htaccess` or vhost):

```apache
Header always set Cross-Origin-Opener-Policy "same-origin"
Header always set Cross-Origin-Embedder-Policy "require-corp"
Header always set Cross-Origin-Resource-Policy "same-origin"
AddType application/wasm .wasm
AddType text/javascript .mjs
```

Client requirements: desktop Chrome or Edge (current), 8 GB RAM or more. Measured on the demo case on an 8-thread Intel Mac: **43 s with WebGPU, 118 s on the CPU**. Managed browsers need WebGPU and service workers left enabled (or the headers above).

## 4. Getting studies from PACS

**Works today (no integration):** export the non-contrast head CT from PACS or the viewer as DICOM (a zip, a folder or loose files) and drag it onto the page. From a whole study the app picks the axial, non-contrast, soft-tissue head series nearest 5 mm and skips bone-kernel, contrast, CTA, reformats and scouts. Compressed transfer syntaxes (JPEG lossless, JPEG-LS, JPEG 2000, HTJ2K, RLE, deflate) are supported; enhanced multi-frame DICOM is not.

**Deeper integration options (not built yet), from least to most work:**

1. **DICOMweb pull in the browser.** Add a "load study" option that fetches a series by StudyInstanceUID from the PACS DICOMweb (WADO-RS) endpoint. Needs the endpoint to allow CORS from the app's origin and to accept the user's existing auth token. The rest of the pipeline is unchanged.
2. **Viewer plug-in.** Wrap the pipeline (`lib/pipeline.js` is independent of the page) as an OHIF viewer extension or a similar plug-in to your enterprise viewer, overlaying the labels on the study the radiologist has open.
3. **Automated server-side node.** Receive studies by DICOM C-STORE from a PACS routing rule (e.g. an Orthanc or MONAI Deploy application), run the same ONNX parts with Python/C++ ONNX Runtime (the orchestration is `run_parts()` in `tools/parts.py`; pre/post-processing is in BrainHemoAI or ports straight from `lib/hemo.js`), and send back a **DICOM SEG** (the masks) plus an **SR or secondary capture** (the volumes) to PACS as a new series. This is what batch processing of a research cohort needs.

## 5. Privacy notes

- Pixel data and DICOM headers stay in the browser tab's memory; nothing is uploaded, logged remotely or stored, apart from the model files in the browser cache.
- Downloads: the NIfTI label map and CT contain no DICOM tags. **The JSON report includes the series description and, for NIfTI/NRRD inputs, the file name**; remove those if file names carry identifiers.

## 6. Before research use at Mayo

- **Governance:** register through the institutional AI review and IRB processes for research use on patient images; keep outputs out of the clinical record.
- **Model weights:** converted from BrainHemoAI (Hu et al., *NeuroImage* 2023). That repository states no licence; confirm permission for institutional use with the authors (corresponding author Xingen Zhu, Second Affiliated Hospital of Nanchang University).
- **Code licence:** the app code is CC BY-NC-SA 4.0 (derived from radar-web): **non-commercial**, share-alike. Commercial use (e.g. inside a product) would need a clean-room reimplementation of the derived files or a licence from their authors.
- **Local validation plan (suggested):** run a SAHVAI cohort sample (e.g. 50–100 aneurysmal SAH scans across scanners) against manual segmentations and ABC/2 volumes; report Dice and ICC per compartment; review the brain-outline QC on every case; note failure modes (post-craniotomy or craniectomy, clip or coil artifact, EVD tracts, motion, contrast given). The AI "SAH" volume includes all subarachnoid blood, not only the five cisterns of the ABC/2 SAHV used by the eSAH score.

## 7. Verifying the build

The conversion and its checks are in `tools/` (needs PyTorch, ONNX, ONNX Runtime and the BrainHemoAI repository). On the demo case the browser pipeline reproduces the original PyTorch `demo.run(..., post_mode='connect')` to within 6 of 7.3 million voxels (SAH Dice 1.000, IVH 0.9998) given the same brain mask and rotation; the fully automatic pipeline gives SAH 77.6 mL and IVH 2.7 mL on both WebGPU and CPU.

## Citation

Hu P, Zhou H, Yan T, et al. Deep learning-assisted identification and quantification of aneurysmal subarachnoid hemorrhage in non-contrast CT scans: development and external validation of Hybrid 2D/3D UNet. *NeuroImage* 2023;279:120321.
