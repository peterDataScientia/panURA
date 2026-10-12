# PanURA
**An Interactive and Explainable QSAR Platform for Urease Inhibitor Discovery**

PanURA is a Streamlit application for exploring chemical structures and predicting *H. pylori* urease inhibitor activity using a separately trained QSAR model.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model artifacts
The application starts without a model and shows a clear **Model not configured** state. No synthetic predictions are generated.

Place compatible artifacts in `models/`:
- `model.joblib`: trained regressor with a `predict` method
- `feature_bits.json`: JSON array of the selected **zero-based PubChem fingerprint bit indices**, in training order (142 indices for the current study)
- `scaler.joblib` (optional): fitted transform applied before model prediction
- `train_fingerprints.npy` (optional): full 881-bit training fingerprints for Tanimoto similarity
- `ad_reference.json` (optional): contains an independently validated applicability-domain specification; leverage is **not** calculated from a single query molecule

**Important:** prediction preprocessing, bit ordering, fingerprint version, scaling, target units, and applicability-domain rules must match the validated training notebook. The model from HematoAI targets a different protein and is not compatible.

## Features
- SMILES input and optional JSME sketcher
- RDKit molecular structure rendering and descriptors
- Single-molecule prediction and CSV/XLSX batch screening
- Interactive Plotly distribution and downloadable CSV
- Clear model status and scientific scope
- Custom responsive CSS

This is an initial application foundation, not a substitute for validation of deployed model artifacts.
