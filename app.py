"""PanURA: urease inhibitor QSAR screening platform."""
from __future__ import annotations

import io
import base64
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from skfp.fingerprints import PubChemFingerprint

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
st.set_page_config(page_title="PanURA | Urease QSAR", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
css_path = ROOT / "styles.css"
if css_path.exists():
    st.markdown("<style>" + css_path.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    model_path = MODEL_DIR / "Final_Model.pkl.b64"
    manifest_path = MODEL_DIR / "Feature_Selection_Manifest.json.b64"
    if not model_path.exists() or not manifest_path.exists():
        return None, "Original model or feature manifest is missing."
    try:
        model = joblib.load(io.BytesIO(base64.b64decode(model_path.read_text(encoding="ascii"))))
        manifest = json.loads(base64.b64decode(manifest_path.read_text(encoding="ascii")))
        names = manifest["selected_features"]
        if len(names) != 142 or len(set(names)) != 142 or any(not name.startswith("PubChem_") for name in names):
            raise ValueError("Invalid 142-feature training manifest")
        bits = [int(name.split("_")[1]) for name in names]
        if any(i < 0 or i >= 881 for i in bits):
            raise ValueError("PubChem feature index out of range")
        if getattr(model, "n_features_in_", 142) != 142:
            raise ValueError("Final model does not expect 142 descriptors")
        train_path = MODEL_DIR / "train_fingerprints.npy"
        train = np.load(train_path, allow_pickle=False) if train_path.exists() else None
        if train is not None and (train.ndim != 2 or train.shape[1] != 881):
            raise ValueError("Training fingerprints must have 881 columns")
        return {"model": model, "bits": bits, "train": train, "fp": PubChemFingerprint()}, None
    except Exception as exc:
        return None, f"Could not load model artifacts: {exc}"

def parse_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles).strip()) if pd.notna(smiles) else None
    return mol

def predict(smiles, assets):
    mol = parse_smiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    fp = np.asarray(assets["fp"].transform([Chem.MolToSmiles(mol)]), dtype=np.float64)
    if fp.shape != (1, 881):
        raise ValueError(f"Unexpected fingerprint shape: {fp.shape}")
    x = fp[:, assets["bits"]]
    value = float(assets["model"].predict(x)[0])
    if not np.isfinite(value):
        raise ValueError("Non-finite prediction")
    similarity = None
    if assets["train"] is not None:
        query = DataStructs.CreateFromBitString("".join(map(str, fp[0].astype(int))))
        refs = [DataStructs.CreateFromBitString("".join(map(str, row.astype(int)))) for row in assets["train"]]
        similarity = max(DataStructs.BulkTanimotoSimilarity(query, refs)) if refs else None
    return value, similarity

st.image(str(ROOT / "assets" / "panura-logo.svg"), width=340)
st.markdown("""<div class="hero"><div class="eyebrow">PAN SCIENTIFIC APPLICATIONS · CHEMINFORMATICS</div>
<h1>PanURA</h1><p>Interactive and Explainable QSAR for <i>H. pylori</i> Urease Inhibitor Discovery</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.image(str(ROOT / "assets" / "panura-logo.svg"), use_container_width=True)
    page = st.radio("Workspace", ["Molecular Prediction", "Batch Screening", "Model & Methods"], label_visibility="collapsed")
    st.divider()
    st.caption("PubChem fingerprints · QSAR regression · Structure-aware screening")
    st.caption("Research use only · Predictions require a validated model.")

assets, model_error = load_predictor()
if model_error:
    st.warning("Model not configured — " + model_error)

if page == "Molecular Prediction":
    st.markdown('<div class="panel-heading">Molecular prediction workspace</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")
    with left:
        mode = st.radio("Molecule input", ["SMILES", "Draw molecule"], horizontal=True)
        smiles = ""
        if mode == "SMILES":
            smiles = st.text_input("Canonical or isomeric SMILES", placeholder="Paste a valid SMILES")
        else:
            try:
                from streamlit_ketcher import st_ketcher
                smiles = st_ketcher(value=st.session_state.get("panura_drawing", ""), key="panura_ketcher") or ""
                st.session_state["panura_drawing"] = smiles
            except Exception as exc:
                st.info(f"Sketcher unavailable ({exc}). Paste SMILES instead.")
                smiles = st.text_input("SMILES", key="fallback_smiles")
        mol = parse_smiles(smiles) if smiles else None
        if smiles and mol is None:
            st.error("SMILES could not be parsed.")
        if mol is not None:
            try:
                from rdkit.Chem import Draw
                st.image(Draw.MolToImage(mol, size=(520, 300)), caption="RDKit 2D molecular structure")
            except (ImportError, OSError, RuntimeError) as drawing_error:
                st.warning("2D rendering is unavailable in this environment; molecular prediction remains accessible.")
                st.caption(str(drawing_error))
            a, b, c = st.columns(3)
            a.metric("Molecular weight", f"{Descriptors.MolWt(mol):.1f}")
            b.metric("cLogP", f"{Descriptors.MolLogP(mol):.2f}")
            c.metric("H-bond donors", str(Descriptors.NumHDonors(mol)))
    with right:
        st.markdown('<div class="panel-heading">Prediction console</div>', unsafe_allow_html=True)
        st.caption("Predicted pIC₅₀ is displayed only when validated artifacts are installed.")
        if st.button("Predict activity", type="primary", disabled=assets is None or mol is None, use_container_width=True):
            try:
                value, similarity = predict(smiles, assets)
                st.metric("Predicted pIC₅₀", f"{value:.3f}")
                st.metric("Estimated IC₅₀", f"{10 ** (6-value):.3g} µM")
                if similarity is not None:
                    st.metric("Maximum training-set Tanimoto", f"{similarity:.3f}")
                st.info("Similarity alone is not a validated applicability-domain decision.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        st.divider()
        st.markdown("**Interpretability**")
        st.info("Compound-specific SHAP will be enabled after the trained model and its explainer/background data are verified.")

elif page == "Batch Screening":
    st.markdown('<div class="panel-heading">Batch screening</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload CSV or XLSX containing a SMILES column", type=["csv", "xlsx"])
    if upload:
        try:
            df = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
            st.dataframe(df.head(20), use_container_width=True)
            columns = [c for c in df.columns if "smiles" in str(c).lower()]
            selected = st.selectbox("SMILES column", list(df.columns), index=list(df.columns).index(columns[0]) if columns else 0)
            if st.button("Run batch predictions", type="primary", disabled=assets is None):
                results = []
                progress = st.progress(0)
                for i, smi in enumerate(df[selected]):
                    try:
                        value, sim = predict(smi, assets)
                        results.append({"pIC50_pred": value, "IC50_uM_pred": 10 ** (6-value), "max_tanimoto": sim, "status": "OK"})
                    except Exception as exc:
                        results.append({"pIC50_pred": np.nan, "IC50_uM_pred": np.nan, "max_tanimoto": np.nan, "status": str(exc)})
                    progress.progress((i+1)/len(df))
                output = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
                st.session_state["batch_output"] = output
            if "batch_output" in st.session_state:
                output = st.session_state["batch_output"]
                st.dataframe(output, use_container_width=True)
                valid = output.dropna(subset=["pIC50_pred"])
                if not valid.empty:
                    st.plotly_chart(px.histogram(valid, x="pIC50_pred", nbins=25, title="Predicted pIC₅₀ distribution"), use_container_width=True)
                st.download_button("Download predictions (CSV)", output.to_csv(index=False).encode("utf-8"), file_name="PanURA_predictions.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not process uploaded data: {exc}")
else:
    st.subheader("Model and methods")
    a, b, c, d = st.columns(4)
    a.metric("Curated compounds", "351")
    b.metric("Selected PubChem bits", "142")
    c.metric("Reported test R²", "0.782")
    d.metric("Reported test RMSE", "0.493")
    st.markdown("""**Target:** *Helicobacter pylori* urease inhibitor activity

**Method:** PubChem fingerprint descriptors (881 bits), feature selection, and a LightGBM regression model.

**Reported validation:** Kennard–Stone split (280 training / 71 test); test R² ≈ 0.782, test RMSE ≈ 0.493 pIC₅₀ units.

**Deployment status:** The original CV-selected Final_Model.pkl and 142-bit training feature manifest are installed. No optimized model is used. Deployment predictions still require independent verification against the saved test predictions. These validation metrics describe the research model, not a newly validated deployment.

**Applicability domain:** Similarity is an exploratory diagnostic; leverage requires the fitted training feature matrix or a validated AD reference. No query-only leverage formula is used.

**SHAP:** Planned interactive explanations must be computed from the trained model, not a static image reused for every compound.
""")
