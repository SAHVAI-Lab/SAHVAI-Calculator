# Deploying SAHVAI Web inside an institution (e.g. Mayo Clinic)

Version 2.1 · 25 October 2026 (four pages: SAHVAI / BrainHemoAI, CLAIM, BLAST-CT, RHUH dataset) · source: `sahvai-web/` in github.com/sahvai-lab/SAHVAI-Calculator

> **Research and education use only.** Not a medical device, not FDA-cleared, and checked against the original implementations on public demo cases only. Deploy for research under your institution's AI-governance and IRB processes; do not use results for patient care.

## 1. What is in the package

| path | contents |
|---|---|
| `index.html`, `app.js`, `worker.js`, `style.css` | the SAHVAI (BrainHemoAI) page (static files, no build step) |
| `claim.html`, `blast.html`, `rhuh.html`, `app-models.js`, `worker-models.js`, `style-models.css` | the CLAIM, BLAST-CT and RHUH dataset pages (`tools/make_pages.py` regenerates the HTML) |
| `lib/` | DICOM / NIfTI / NRRD readers, resampling, brain extraction, UNet orchestration, post-processing |
| `models/` | SAHVAI: the Hybrid 2D/3D UNet as nine ONNX parts (26.5 MB, float16 weights) |
| `models/blast/` | BLAST-CT: the 12 BLAST-CT DeepMedic models (2.1 MB each, float16; Apache-2.0) |
| `models/claim/` | CLAIM: the CLAIM nnU-Net 2D model in two parts (92.7 MB, float16; CC BY-NC 4.0) |
| `models/deepbleed/` | DeepBleed: the V-Net in two parts (90.2 MB, float32, unchanged weights; CC BY-NC-ND 4.0) and its 1.5 mm CT template |
| `vendor/` | ONNX Runtime Web 1.30 and DICOM image codecs (all local) |
| `coi-serviceworker.js` | enables multi-threading when the server cannot send the headers below |
| `demo/` | public demo cases: BrainHemoAI `head.nrrd`, three BHSD cases from the CLAIM repository with labels, BLAST-CT `scan_0` with labels, one MISTIE/CLEAR ICH scan (JHU, CC BY-NC 4.0) |
| `licenses/` | full licence texts for BLAST-CT (Apache-2.0) and CLAIM (CC BY-NC 4.0) |
| `tools/` | Python scripts that rebuild the ONNX parts from the original PyTorch weights |
| `serve.py` | a standard-library Python server with the right headers |

The app makes **no external network requests**: the model, runtime and codecs are all served from the same folder, so it works on an isolated intranet. Images are processed in the browser's memory and never leave the workstation.

## 2. Try it on one workstation

```
cd sahvai-web
python3 serve.py            # then open http://localhost:8000/ in Chrome or Edge
```

Expected demo results: SAHVAI page **Try the demo case** SAH ≈ 77.6 mL, IPH 0 mL, IVH ≈ 2.7 mL; BLAST-CT page **Try the demo case (TBI)** with the single model IPH ≈ 48.0 mL, EAH ≈ 23.7 mL, oedema ≈ 62.7 mL, IVH ≈ 0.3 mL (as the `blast-ct` command line); CLAIM page **Demo: BHSD case 1** (exact mode) SAH ≈ 20.6 mL, IVH ≈ 0.9 mL, ventricles ≈ 96 mL, any-blood Dice 0.44 against the BHSD labels (nnU-Net itself: 0.45).

## 3. Host it on an internal web server

It is a static site: copy the folder to any web server. Four requirements:

1. **HTTPS** (or `localhost`). WebGPU, the service worker and the model cache need a secure context.
2. **Cross-origin isolation headers** on every response, for multi-threaded CPU inference:
   `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.
   (Without them `coi-serviceworker.js` adds them on first load; with them it is not needed.)
3. **MIME types**: `.wasm` → `application/wasm`, `.mjs` / `.js` → `text/javascript`.
4. Allow files up to ~50 MB (`models/claim/claim2d.onnx.0` and `.1`, `models/deepbleed/deepbleed.onnx.0` and `.1`, `models/pm.onnx`, the runtime `.wasm`).

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
- **Model weights:** SAHVAI converted from BrainHemoAI (Hu et al., *NeuroImage* 2023), MIT License with the permission of Dr Ping Hu and the BrainHemoAI group;  CLAIM (Kiewitz et al., *Front Neurol* 2024), **CC BY-NC 4.0, non-commercial only**; BLAST-CT (Monteiro et al., *Lancet Digit Health* 2020), Apache-2.0. See `LICENSE.md`. BLAST-CT was trained on traumatic brain injury and CLAIM on aneurysmal SAH at one centre; neither has been validated at Mayo.
- **RHUH dataset page:** CC BY 4.0, downloaded by the user from Zenodo; not redistributed in this package.
- **Code licence:** the app code is CC BY-NC-SA 4.0 (derived from radar-web): **non-commercial**, share-alike. Commercial use (e.g. inside a product) would need a clean-room reimplementation of the derived files or a licence from their authors.
- **Local validation plan (suggested):** run a SAHVAI cohort sample (e.g. 50–100 aneurysmal SAH scans across scanners) against manual segmentations and ABC/2 volumes; report Dice and ICC per compartment; review the brain-outline QC on every case; note failure modes (post-craniotomy or craniectomy, clip or coil artifact, EVD tracts, motion, contrast given). The AI "SAH" volume includes all subarachnoid blood, not only the five cisterns of the ABC/2 SAHV used by the eSAH score.

## 7. Verifying the build

The conversion and its checks are in `tools/` (needs PyTorch, ONNX, ONNX Runtime and the BrainHemoAI repository). On the demo case the browser pipeline reproduces the original PyTorch `demo.run(..., post_mode='connect')` to within 6 of 7.3 million voxels (SAH Dice 1.000, IVH 0.9998) given the same brain mask and rotation; the fully automatic pipeline gives SAH 77.6 mL and IVH 2.7 mL on both WebGPU and CPU.

## Citation

- Hu P, Zhou H, Yan T, et al. Deep learning-assisted identification and quantification of aneurysmal subarachnoid hemorrhage in non-contrast CT scans: development and external validation of Hybrid 2D/3D UNet. *NeuroImage* 2023;279:120321.
- Kiewitz J, Aydin OU, Hilbert A, et al. Deep learning-based multiclass segmentation in aneurysmal subarachnoid hemorrhage. *Front Neurol* 2024;15:1490216.
- Monteiro M, Newcombe VFJ, Mathieu F, et al. Multiclass semantic segmentation and quantification of traumatic brain injury lesions on head CT using deep learning. *Lancet Digit Health* 2020;2(6):e314–e322.
- Cepeda S, García-García S. Non-contrast CT scans dataset for subarachnoid hemorrhage: bleeding segmentation. Zenodo; 2023. doi:10.5281/zenodo.8228847.
