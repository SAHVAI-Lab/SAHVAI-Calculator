# Generates claim.html, blast.html and rhuh.html (one panel each; index.html is the SAHVAI / BrainHemoAI page).
import html, os, re

def drop(pid, demos, gt=True, hint=''):
    btns = ''.join(f'<button class="btn ghost" type="button" data-demo="{i}">{html.escape(d)}</button>' for i, d in enumerate(demos))
    gtrow = f'''<div class="gtrow"><label class="btn small ghost">Optional: reference labels (.nii.gz)<input class="f-gt" type="file" accept=".nii,.gz,.nrrd"></label>
      <span class="small">reference: <b class="gtname">none</b> · must share the scan's geometry; Dice is reported per class</span></div>''' if gt else ''
    return f'''<div class="drop" tabindex="0">
      <p class="big"><b>Drop a head CT here</b></p>
      <p class="hint">A zip of a DICOM study, a DICOM folder, loose DICOM files, a .nii / .nii.gz or a .nrrd volume.{hint}</p>
      <div class="buttons">
        <label class="btn">Choose file(s)<input class="f-files" type="file" multiple accept=".zip,.nii,.gz,.nrrd,.dcm,application/zip,application/dicom,*/*"></label>
        <label class="btn">Choose folder<input class="f-folder" type="file" webkitdirectory></label>
        {btns}
      </div>
      {gtrow}
    </div>'''

STATUS = '''<div class="status" hidden><div class="stage"><span class="stage-name">Starting</span><span class="pct"></span></div><div class="bar"><i></i></div></div>'''
RESULTS = '''<div class="results" hidden>
      <div class="vols"></div>
      <div class="warnbox" hidden></div>
      <div class="cmp" hidden></div>
      <div class="viewerhost viewer"></div>
      <div class="details"></div>
    </div>'''

def panel(pid, num, title, sub, origin, body, cite, extra_cls=''):
    return f'''
  <section class="panel {extra_cls}" id="panel-{pid}" aria-labelledby="h-{pid}">
    <header class="phead">
      <span class="num">{num}</span>
      <div><h2 id="h-{pid}">{title}</h2><p class="sub">{sub}</p></div>
      <span class="origin">{origin}</span>
    </header>
    {body}
    {STATUS}
    {RESULTS}
    <div class="cite">{cite}</div>
  </section>'''

DISCLAIMER = '<p class="disc"><b>Disclaimer.</b> For research and education only. Not a medical device; not cleared or approved by the FDA, CE or any regulator; not validated for clinical use; must not be used for diagnosis, triage, treatment or any other clinical decision. Outputs can be wrong and must be reviewed against the images by a qualified reader. Provided "as is" without warranty of any kind. Images are processed locally in this browser and are not uploaded.</p>'

# ---------- 1 SAHVAI ----------
p1 = panel('sahvai', '1', 'SAHVAI <span>BrainHemoAI Hybrid 2D/3D UNet</span>',
  'Subarachnoid (SAH), intraparenchymal (IPH) and intraventricular (IVH) hemorrhage on non-contrast head CT, with volumes.',
  'Hu et al. · Shanghai / BrainHemoAI',
  f'''<div class="opts">
      <div class="group"><span class="lbl">Head alignment</span>
        <label><input type="radio" name="align" value="auto" checked> Auto</label>
        <label><input type="radio" name="align" value="manual"> Set</label>
        <input class="angle" type="number" min="-45" max="45" step="1" value="0" disabled aria-label="Rotation in degrees"> °</div>
      <span class="small">Model download about 26 MB, cached after the first visit.</span>
    </div>
    {drop('sahvai', ['Try the demo case'], gt=True, hint=' From a study, the axial non-contrast head series nearest 5 mm is used (bone-kernel, contrast and reformatted series are skipped). Thin slices are averaged to 5 mm.')}''',
  f'''<h3>Citation, licence and disclaimer</h3>
    <p><b>Model.</b> Hu P, Zhou H, Yan T, et al. Deep learning-assisted identification and quantification of aneurysmal subarachnoid hemorrhage in non-contrast CT scans: development and external validation of Hybrid 2D/3D UNet. <i>NeuroImage</i> 2023;279:120321. <a href="https://doi.org/10.1016/j.neuroimage.2023.120321">doi:10.1016/j.neuroimage.2023.120321</a></p>
    <p><b>Code and weights.</b> <a href="https://github.com/BrainHemo/BrainHemoAI">BrainHemoAI</a> (Dr Ping Hu and colleagues), converted to ONNX for the browser. Used with the permission of Dr Ping Hu under the <a href="LICENSE.md#brainhemoai-mit-license">MIT License</a>. Demo case: <code>head.nrrd</code> from the BrainHemoAI repository.</p>
    <p><b>How it runs here.</b> Reorientation and resampling to 0.5 × 0.5 × 5 mm, brain extraction, midline alignment, five HU windows, 18-slice windows through the network (split into nine ONNX parts so it fits in a browser tab), post-processing and voxel-count volumes, as in the original code. On the BrainHemoAI demo case the port matches the PyTorch pipeline to within 6 voxels.</p>
    {DISCLAIMER}''')

# ---------- 2 CLAIM ----------
p2 = panel('claim', 'C', 'CLAIM aSAH <span>Charité Berlin · nnU-Net multiclass</span>',
  'Six classes on non-contrast CT after aneurysmal SAH: SAH, intraventricular hemorrhage, ventricles, intracerebral hemorrhage, aneurysm and subdural hematoma.',
  'CLAIM · Charité Lab for AI in Medicine, Berlin',
  f'''<div class="pending note warn" hidden><b>CLAIM model files not found.</b> This copy of SAHVAI Web has no <code>models/claim/</code> folder; convert the CC BY-NC 4.0 weights from Zenodo with <code>tools/claim_export.py</code> to enable this panel.</div>
    <div class="opts">
      <div class="group" role="radiogroup" aria-label="CLAIM inference mode"><span class="lbl">Inference</span>
        <label title="The nnU-Net sliding window restricted to the skull plus 8 mm: about half the network passes, small differences at tile edges"><input type="radio" name="claim-mode" value="fast"> Fast (head region only, approximate)</label>
        <label title="nnU-Net sliding window with Gaussian weighting, as nnUNetv2_predict"><input type="radio" name="claim-mode" value="exact" checked> Exact nnU-Net sliding window (default)</label></div>
      <label><input class="mirror" type="checkbox"> Mirroring (test-time augmentation, 4× slower)</label>
      <span class="small">2D model, 92.7 MB download, cached after the first run.</span>
    </div>
    {drop('claim', ['Demo: BHSD case 1', 'Demo: BHSD case 2', 'Demo: BHSD case 3'], gt=True, hint=' The 2D model works slice by slice at 0.445 mm in plane; it was trained on 5.5 to 6.5 mm slices, so thin slices are averaged to 5 mm. The demo cases include the expert labels, so Dice is shown.')}''',
  f'''<h3>Citation, licence and disclaimer</h3>
    <p><b>Model.</b> Kiewitz J, Aydin OU, Hilbert A, Gultom M, Nouri A, Khalil AA, Vajkoczy P, Tanioka S, Ishida F, Dengler NF, Frey D. Deep learning-based multiclass segmentation in aneurysmal subarachnoid hemorrhage. <i>Front Neurol</i> 2024;15:1490216. <a href="https://doi.org/10.3389/fneur.2024.1490216">doi:10.3389/fneur.2024.1490216</a> (preprint: medRxiv <a href="https://doi.org/10.1101/2024.06.24.24309431">2024.06.24.24309431</a>)</p>
    <p><b>Code and weights.</b> <a href="https://claim.charite.de/en/projects/subarachnoid_hemorrhage_segmentation">CLAIM project page</a> · <a href="https://github.com/claim-berlin/aSAH-multiclass-segmentation">github.com/claim-berlin/aSAH-multiclass-segmentation</a> · weights on Zenodo <a href="https://doi.org/10.5281/zenodo.12579491">doi:10.5281/zenodo.12579491</a>. Licensed <a href="https://creativecommons.org/licenses/by-nc/4.0/">CC BY-NC 4.0</a> (attribution, non-commercial use only); the nnU-Net 2D network is converted to ONNX without changes to the weights. nnU-Net: Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. <i>Nat Methods</i> 2021;18:203–211 (Apache-2.0).</p>
    <p><b>Demo data.</b> Three cases from the BHSD dataset as distributed in the CLAIM repository: Wu B, Xie Y, Zhang Z, et al. BHSD: a 3D multi-class brain hemorrhage segmentation dataset. In: <i>Machine Learning in Medical Imaging (MLMI 2023)</i>, LNCS 14348. Springer; 2024. <a href="https://doi.org/10.1007/978-3-031-45673-2_15">doi:10.1007/978-3-031-45673-2_15</a>. The reference labels shown with the demo are those supplied with the CLAIM demo data.</p>
    <p><b>How it runs here.</b> As nnUNetv2_predict for the 2D configuration: clip to −3…88 HU, cubic-spline resampling in plane to 0.4453 mm, z-score (mean 32.47, SD 27.41), 512 × 512 tiles with Gaussian weighting, argmax. Checked against nnUNetv2_predict with the published weights on the three demo cases (see the README). The authors report similar accuracy for their 2D and 3D models and used the 2D model for external validation.</p>
    {DISCLAIMER}''')

# ---------- 3 BLAST-CT ----------
p3 = panel('blast', 'B', 'BLAST-CT <span>Cambridge &amp; Imperial · traumatic brain injury lesions</span>',
  'Four lesion classes from the CENTER-TBI work: intraparenchymal haemorrhage, extra-axial haemorrhage, perilesional oedema and intraventricular haemorrhage. Use it for SAH after a fall (traumatic SAH is part of the extra-axial class) or for traumatic ICH.',
  'Menon, Newcombe, Glocker et al. · University of Cambridge NeuroICU · Imperial College London',
  f'''<div class="opts">
      <div class="group" role="radiogroup" aria-label="BLAST-CT models"><span class="lbl">Models</span>
        <label><input type="radio" name="blast-ens" value="1" checked> Single model (blast-ct default)</label>
        <label><input type="radio" name="blast-ens" value="12"> Ensemble of 12 (as <code>--ensemble</code>, 12× slower)</label></div>
      <span class="small">About 2 MB per model, cached.</span>
    </div>
    <div class="note small-note">Not the same task as SAHVAI and CLAIM: <b>EAH</b> combines subarachnoid, subdural and extradural blood, and the model was trained on traumatic brain injury, not aneurysmal SAH.</div>
    {drop('blast', ['Try the demo case (TBI)'], gt=True, hint=' The scan is resampled to 1 mm isotropic, as in the blast-ct tool.')}''',
  f'''<h3>Citation, licence and disclaimer</h3>
    <p><b>Model.</b> Monteiro M, Newcombe VFJ, Mathieu F, Adatia K, Kamnitsas K, Ferrante E, Das T, Whitehouse D, Rueckert D, Menon DK, Glocker B. Multiclass semantic segmentation and quantification of traumatic brain injury lesions on head CT using deep learning: an algorithm development and multicentre validation study. <i>Lancet Digit Health</i> 2020;2(6):e314–e322. <a href="https://doi.org/10.1016/S2589-7500(20)30085-6">doi:10.1016/S2589-7500(20)30085-6</a></p>
    <p><b>Code and weights.</b> <a href="https://github.com/biomedia-mira/blast-ct">github.com/biomedia-mira/blast-ct</a> (BioMedIA, Imperial College London, with the University of Cambridge Division of Anaesthesia), <a href="https://www.apache.org/licenses/LICENSE-2.0">Apache License 2.0</a>, Copyright 2020 Imperial College London. The 12 pre-trained DeepMedic models are converted to ONNX (fp16) without retraining. Architecture: Kamnitsas K, Ledig C, Newcombe VFJ, et al. <i>Med Image Anal</i> 2017;36:61–78. The software "is not intended for clinical use" (blast-ct README). Demo: <code>scan_0</code> and its label map from the blast-ct repository.</p>
    <p><b>How it runs here.</b> As blast-ct: reorientation, 1 mm isotropic linear resampling (SimpleITK-style), window −15 to 100 HU scaled to −1…1, three-pathway DeepMedic over tiles, softmax averaged over the chosen models, argmax. Output tiles of 60³ give the same labels as the tool's 30³ tiles (checked: 0 voxels differ).</p>
    {DISCLAIMER}''')

# ---------- 4 RHUH dataset ----------
p4 = panel('rhuh', 'R', 'RHUH SAH dataset <span>Cepeda &amp; García-García · Valladolid, Spain</span>',
  'Open non-contrast CT scans of aneurysmal SAH with expert manual bleeding masks. Load one case and its mask, run any of the three models on it and compare with the expert segmentation.',
  'Río Hortega University Hospital · Zenodo',
  f'''<ol class="steps">
      <li>Download <code>subarachnoid_hemorrhage_rhuh.rar</code> (649 MB) from <a href="https://zenodo.org/records/8228847">Zenodo record 8228847</a> and extract it (browsers cannot open RAR archives).</li>
      <li>Drop the two NIfTI files of one case here (the CT and its mask), or choose them below.</li>
      <li>Pick a model and press <b>Run</b>. The expert mask is drawn in white and Dice and volumes are compared as "any blood".</li>
    </ol>
    <div class="drop" tabindex="0">
      <p class="big"><b>Drop one RHUH case here</b> <span class="small">(CT + expert mask, .nii / .nii.gz)</span></p>
      <div class="buttons">
        <label class="btn ghost">CT volume<input class="f-ct" type="file" accept=".nii,.gz,.nrrd"></label>
        <label class="btn ghost">Expert mask<input class="f-mask" type="file" accept=".nii,.gz,.nrrd"></label>
      </div>
      <p class="small">CT: <b class="ctname">none</b> · mask: <b class="maskname">none</b></p>
      <div class="buttons">
        <label class="lbl" for="rhuh-model">Model</label>
        <select class="model" id="rhuh-model">
          <option value="sahvai">1 · SAHVAI (BrainHemoAI)</option>
          <option value="claim">2 · CLAIM nnU-Net</option>
          <option value="blast">3 · BLAST-CT</option>
        </select>
        <button class="btn run" type="button">Run</button>
      </div>
    </div>
    <div class="note small-note">The dataset description lists anonymisation, resampling, skull stripping and intensity normalisation among its preprocessing. The three models expect raw Hounsfield units and the full head, so the page checks the intensity range and warns when a volume does not look like HU; skull-stripped input can also affect the brain extraction of SAHVAI (BrainHemoAI).</div>''',
  f'''<h3>Citation, licence and disclaimer</h3>
    <p><b>Dataset.</b> Cepeda S, García-García S. Non-contrast CT scans dataset for subarachnoid hemorrhage: bleeding segmentation [data set]. Zenodo; 2023. Version 1. <a href="https://doi.org/10.5281/zenodo.8228847">doi:10.5281/zenodo.8228847</a>. Licensed <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Río Hortega University Hospital, Valladolid, Spain.</p>
    <p><b>Related work by the dataset authors.</b> García-García S, Cepeda S, Müller D, Mosteiro A, Torné R, Agudo S, de la Torre N, Arrese I, Sarabia R. Mortality prediction of patients with subarachnoid hemorrhage using a deep learning model based on an initial brain CT scan. <i>Brain Sci</i> 2024;14(1):10. <a href="https://doi.org/10.3390/brainsci14010010">doi:10.3390/brainsci14010010</a> · García-García S, Cepeda S, Arrese I, Sarabia R. A fully automated pipeline using Swin transformers for deep learning-based blood segmentation on head computed tomography scans after aneurysmal subarachnoid hemorrhage. <i>World Neurosurg</i> 2024. <a href="https://doi.org/10.1016/j.wneu.2024.07.216">doi:10.1016/j.wneu.2024.07.216</a> · Code: <a href="https://github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction">smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction</a> (Academic Non-Commercial Source Code License; their Swin UNETR weights are not distributed and are not run here).</p>
    <p>Data you drop here stays in this browser. The comparison is between a public research dataset and research models; it is not a validation of any model for clinical use.</p>
    {DISCLAIMER}''', 'dataset')


PAGES = [
  ('claim', 'claim.html', 'CLAIM aSAH segmentation in the browser', 'CLAIM aSAH <span>in the browser</span>',
   'The multiclass aneurysmal SAH model of the CLAIM group at Charité Berlin (Kiewitz et al., <i>Front Neurol</i> 2024): SAH, intraventricular hemorrhage, ventricles, intracerebral hemorrhage, aneurysm and subdural hematoma on non-contrast head CT. Everything runs on this computer.', p2,
   'We thank the <a href="https://claim.charite.de/">CLAIM</a> group at Charité – Universitätsmedizin Berlin for publishing the aSAH multiclass nnU-Net weights (<a href="LICENSE.md#claim-asah-multiclass-segmentation">CC BY-NC 4.0</a>), and the BHSD authors for the demo cases.'),
  ('blast', 'blast.html', 'BLAST-CT TBI lesion segmentation in the browser', 'BLAST-CT <span>in the browser</span>',
   'The traumatic brain injury lesion model of the University of Cambridge and Imperial College London (Monteiro et al., <i>Lancet Digit Health</i> 2020): intraparenchymal and extra-axial haemorrhage, perilesional oedema and intraventricular haemorrhage on non-contrast head CT. Use it for SAH after a fall or for traumatic ICH. Everything runs on this computer.', p3,
   'We thank the BLAST-CT authors at the University of Cambridge and Imperial College London for publishing their code and models (<a href="LICENSE.md#blast-ct-apache-20">Apache-2.0</a>).'),
  ('rhuh', 'rhuh.html', 'RHUH SAH dataset in the browser', 'RHUH SAH dataset <span>in the browser</span>',
   'Open non-contrast CT scans of aneurysmal SAH with expert bleeding masks from Río Hortega University Hospital, Valladolid (Cepeda &amp; García-García, Zenodo 2023). Load a case and its mask, run SAHVAI (BrainHemoAI), CLAIM or BLAST-CT on it and compare with the expert. Everything runs on this computer.', p4,
   'We thank Dr Santiago Cepeda and Dr Sergio García-García for the RHUH dataset (<a href="LICENSE.md#rhuh-dataset">CC BY 4.0</a>), and the BrainHemoAI, CLAIM and BLAST-CT groups for the models it can be compared with.'),
]
NAV = [('index.html', 'SAHVAI (BrainHemoAI)'), ('claim.html', 'CLAIM aSAH'), ('blast.html', 'BLAST-CT'), ('rhuh.html', 'RHUH dataset')]
for pid, fname, title, h1, lede, body, ack in PAGES:
    cur = ' aria-current="page"'
    nav = ''.join(f'<a href="{f}"{cur if f == fname else ""}>{html.escape(t)}</a>' for f, t in NAV)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{html.escape(re.sub('<[^>]+>', '', lede))} Research and education use only.">
<script src="coi-serviceworker.js"></script>
<link rel="stylesheet" href="style-models.css">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E%3Ccircle cx=%2716%27 cy=%2716%27 r=%2714%27 fill=%27%23003da5%27/%3E%3Ccircle cx=%2716%27 cy=%2716%27 r=%278%27 fill=%27none%27 stroke=%27%23ffb000%27 stroke-width=%274%27/%3E%3C/svg%3E">
</head>
<body data-page="{pid}">
<header class="top">
  <div class="wrap">
    <h1>{h1}</h1>
    <p>{lede}</p>
    <nav class="jump" aria-label="SAHVAI Web tools">{nav}</nav>
  </div>
</header>

<main class="wrap">
  <section class="notice" aria-label="Before you start">
    <div class="note warn">
      <h2>Research and education use only</h2>
      <p>A research demonstration, not a medical device. This model is not validated or approved for clinical use here; do not use it for diagnosis, treatment or any other clinical decision. Segmentations can be wrong; always review them against the images.</p>
    </div>
    <div class="note">
      <h2>Your images stay on this computer</h2>
      <p>Files are read and processed inside this browser tab on your GPU or CPU. No image data or results are uploaded. Only the page, the model files and the demo cases are downloaded, and the model is cached for later visits.</p>
    </div>
    <div class="note">
      <h2>What you need</h2>
      <ul>
        <li>A recent desktop Chrome or Edge (Firefox and Safari may work on the CPU).</li>
        <li>8 GB RAM or more. WebGPU is fastest; on the CPU a scan takes one to several minutes.</li>
        <li>A <b>non-contrast axial head CT</b> in Hounsfield units.</li>
      </ul>
    </div>
  </section>

  <section class="controls" aria-label="Settings">
    <div class="group" role="radiogroup" aria-label="Compute device">
      <span class="lbl">Compute</span>
      <label><input type="radio" name="device" value="auto" checked> Auto</label>
      <label><input type="radio" name="device" value="webgpu"> WebGPU</label>
      <label><input type="radio" name="device" value="wasm"> CPU</label>
    </div>
    <div id="caps" class="caps"></div>
  </section>
{body}

  <details id="logbox">
    <summary>Log</summary>
    <pre id="log"></pre>
  </details>
</main>

<footer class="wrap">
  <p><b>Research and education use only. Not a medical device.</b> Images are processed locally and never uploaded.</p>
  <p><b>Acknowledgements.</b> {ack} Inclusion here does not imply endorsement of this site by these groups.</p>
  <p>Browser design adapted from <a href="https://github.com/jarrelscy/radar-web">RADAR in the browser</a> (CC BY-NC-SA 4.0), which runs <a href="https://doi.org/10.1126/science.aec6129">RADAR</a> (Zhang, Ding et al., Science 2026). Runs on <a href="https://onnxruntime.ai/">ONNX Runtime Web</a>. See <a href="LICENSE.md">licences</a>.</p>
  <p>Part of the <a href="../">SAHVAI Calculator</a> · SAHVAI Lab.</p>
</footer>
<script type="module" src="app-models.js"></script>
</body>
</html>
"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', fname)
    open(path, 'w').write(page); print('wrote', path, len(page))
