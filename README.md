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
- `panura_navigation.py` — accessible top-right hamburger menu using Streamlit Components v2 and editor width measurement.
- `styles.css` — responsive design and motion, with reduced-motion support.
- `tests/test_engine.py` — preprocessing/inference contract tests.
- `tests/test_inputs.py` — sample-file integrity and CSV/TXT parsing tests.
- `tests/test_browser.py` — Playwright navigation checks at phone, tablet and desktop viewport sizes.
- `.github/workflows/ci.yml` — automatic Python and browser tests.

## Original model artifacts

The app loads **only** `models/Final_Model.pkl.b64` (base64 encoding of the original `Final_Model.pkl`) and `models/Feature_Selection_Manifest.json.b64`. The manifest contains 142 selected `PubChem_<index>` descriptors; the original 881-bit fingerprints and feature ordering are retained. The app **does not use** `Auto_Model_Optimized_Pipeline.pkl` and does not refit the model.

Optional `models/train_fingerprints.npy` must contain 881-bit rows for the training compounds. If present, the app reports maximum Tanimoto similarity; this is not a formal applicability-domain determination.

Reported research metrics: 351 compounds; 280/71 Kennard–Stone training/test partition; test R² ≈ 0.782 and RMSE ≈ 0.493 pIC₅₀ units. These research metrics are not a claim of a newly validated web deployment.

## Usage

**Molecular Prediction:** Choose Draw molecule (StreamJSME) or Enter SMILES. JSME emits SMILES on its drawing-change callback; Python computes predictions on valid structures without an Apply/Predict button. The component measures editor space to choose a width that fits mobile and desktop. pIC₅₀ and estimated IC₅₀ (µM) update automatically when the input changes.

**Batch Screening:** Download a ready-to-use example from `examples/PanURA_example.csv` or `examples/PanURA_example.txt`, or upload your own CSV, TXT or XLSX file. Example CSV/TXT files contain **ID** and **SMILES** columns. In TXT, these columns are tab-separated; a one-SMILES-per-line TXT file is also supported and receives generated IDs. CSV/XLSX files require a SMILES column; optional user-supplied IDs or Identifiers are preserved verbatim (including leading zeroes). If no identifier is provided, PanURA generates stable `Compound_0001`-style IDs. The identifier remains associated with each molecule through the preview, screening results and downloaded prediction CSV. Example molecules are for input-format demonstration, not measured inhibition data.

**Model & Methods:** Summary of descriptors, model and reported external evaluation.

Use the top-right ☰ Menu to change screens. It closes on selection, Escape or a pointer event outside; the logo stays left and no sidebar is needed.

## Tests

```bash
python -m pytest tests/test_engine.py
python -m playwright install chromium
python -m pytest tests/test_browser.py
```

The GitHub Actions workflow checks syntax, unit tests and Playwright interactions. Browser-mode behavior on Streamlit Cloud still needs real deployment verification after a successful build.
