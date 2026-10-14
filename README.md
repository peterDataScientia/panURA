# PanURA

Interactive QSAR screening application for *Helicobacter pylori* urease inhibitors, using the original final LightGBM regression model.

## Start locally

Use **Python 3.12** for reproducible deployment. Run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Application structure

- `app.py` — Streamlit pages, automatic prediction workflow and batch screening.
- `panura_inputs.py` — CSV, TXT and XLSX upload parsing.
- `panura_engine.py` — independent RDKit/PubChem fingerprint preprocessing and QSAR inference.
- `panura_visuals.py` — ID-labelled interactive activity profiles and adaptive distribution plots.
- `panura_navigation.py` — accessible top-right hamburger menu using Streamlit Components v2 and editor width measurement.
- `styles.css` — responsive design and motion, with reduced-motion support.
- `tests/test_engine.py` — preprocessing/inference contract tests.\n- `tests/test_ad.py` — original training AD reproduction and manifest checks.
- `tests/test_inputs.py` — sample-file integrity and CSV/TXT parsing tests.
- `tests/test_visuals.py` — molecule-level plot and histogram integrity checks.
- `tests/test_browser.py` — Playwright navigation checks at phone, tablet and desktop viewport sizes.
- `.github/workflows/ci.yml` — automatic Python and browser tests.

## Original model artifacts

The app loads **only** `models/Final_Model.pkl.b64` (base64 encoding of the original `Final_Model.pkl`) and `models/Feature_Selection_Manifest.json.b64`. The manifest contains 142 selected `PubChem_<index>` descriptors; the original 881-bit fingerprints and feature ordering are retained. The app **does not use** `Auto_Model_Optimized_Pipeline.pkl` and does not refit the model.

**Applicability domain:** `models/ad_reference.json` contains only the original 280 training molecules' 142 selected fingerprint bits (compressed; no compound IDs or measured endpoints). PanURA reproduces the manuscript's intercept-augmented Moore–Penrose leverage, using the training P95 threshold (approximately 0.98752, reported as 0.988). Prospective screening uses structural leverage only; experimental residual criteria cannot be evaluated for unsynthesized molecules. Maximum Tanimoto on the selected 142 fingerprints is presented as supplementary proximity information, not a reliability probability.

Optional `models/train_fingerprints.npy` must contain 881-bit rows for the **original training compounds**. This file is currently not present in the repository, so maximum Tanimoto similarity cannot be computed. The app omits the column instead of exporting empty values. If genuine matching training fingerprints are provided, maximum Tanimoto similarity is calculated using the 881-bit representation; it is not a formal applicability-domain determination.

Reported research metrics: 351 compounds; 280/71 Kennard–Stone training/test partition; test R² ≈ 0.782 and RMSE ≈ 0.493 pIC₅₀ units. These research metrics are not a claim of a newly validated web deployment.

## Usage

**Molecular Prediction / design:** Paste a SMILES to load an editable JSME structure, or draw a new molecule from scratch in the same workspace. Valid structural edits update the pretrained model's pIC₅₀ and IC₅₀ predictions without a Predict button. Save designs to compare predicted potency against the first saved compound; restore any saved structure for further editing and export the comparison to .csv. The editor uses stable session-state seeds to avoid resetting on drawing updates.

**Batch Screening:** Download a ready-to-use example from `examples/PanURA_example.csv` or `examples/PanURA_example.txt`, or upload your own CSV, TXT or XLSX file. Example CSV/TXT files contain **ID** and **SMILES** columns. In TXT, these columns are tab-separated; a one-SMILES-per-line TXT file is also supported and receives generated IDs. CSV/XLSX files require a SMILES column; optional user-supplied IDs or Identifiers are preserved verbatim (including leading zeroes). If no identifier is provided, PanURA generates stable `Compound_0001`-style IDs. The identifier remains associated with each molecule through the preview, screening results and downloaded prediction CSV. For up to 40 valid predictions, a ranked molecular dot plot shows every compound ID and its pIC₅₀ with SMILES and IC₅₀ on hover. For larger datasets, an adaptive frequency distribution becomes the main visualization; the per-molecule chart remains available in an expander. Example molecules are for input-format demonstration, not measured inhibition data.

**Model & Methods:** Summary of descriptors, model and reported external evaluation.

Use the top-right ☰ Menu to change screens. It closes on selection, Escape or a pointer event outside; the logo stays left and no sidebar is needed.

## Tests

```bash
python -m pytest tests/test_engine.py
python -m playwright install chromium
python -m pytest tests/test_browser.py
```

The GitHub Actions workflow checks syntax, unit tests and Playwright interactions. Browser-mode behavior on Streamlit Cloud still needs real deployment verification after a successful build.
