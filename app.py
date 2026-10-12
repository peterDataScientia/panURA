"""PanURA · live urease-inhibitor QSAR screening workspace."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit.Chem import Descriptors

from panura_engine import load_assets, parse_smiles, predict_smiles
from panura_inputs import read_screening_file
from panura_visuals import activity_distribution_plot, molecule_activity_plot, prepare_activity_data
from panura_navigation import available_editor_width, render_navigation

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

st.set_page_config(
    page_title="PanURA | Urease QSAR",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    "<style>" + (ROOT / "styles.css").read_text(encoding="utf-8") + "</style>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def predictor_assets():
    return load_assets(MODELS)


@st.cache_data(show_spinner=False, max_entries=750)
def cached_prediction(smiles: str):
    # Cache repeated drawings without changing the original trained model.
    return predict_smiles(smiles, predictor_assets())


def intro(section: str, title: str, subtitle: str):
    # All strings below are fixed in code; do not render user input as HTML.
    st.markdown(
        f'<section class="panura-intro"><div class="eyebrow">{section}</div>'
        f'<h2>{title}</h2><p>{subtitle}</p></section>',
        unsafe_allow_html=True,
    )


def section(number: str, title: str, tag: str):
    st.markdown(
        f'<div class="panura-section"><span class="num">{number}</span>'
        f'<strong>{title}</strong><span class="tag">{tag}</span></div>',
        unsafe_allow_html=True,
    )


st.markdown('<div class="panura-topline"></div>', unsafe_allow_html=True)
page = render_navigation(ROOT / "assets" / "panura-logo.svg")

try:
    assets = predictor_assets()
    engine_error = None
except Exception as exc:
    assets, engine_error = None, str(exc)
    st.error("Unable to start the prediction engine: " + engine_error)

if page == "Molecular Prediction":
    editor_col, results_col = st.columns([1.18, 1], gap="large", vertical_alignment="top")

    with editor_col:
        with st.container(key="panura_editor_column"):
            section("01", "Molecular editor", "LIVE INPUT")
            input_mode = st.segmented_control(
                "Molecule input",
                options=["Draw molecule", "Enter SMILES"],
                default="Draw molecule",
                key="panura_input_mode",
            )

            # Keep the JSME initialization seed stable during drawing callbacks.
            previous_mode = st.session_state.get("panura_previous_input_mode")
            if previous_mode != input_mode and input_mode == "Draw molecule":
                st.session_state["panura_editor_seed"] = st.session_state.get("panura_last_valid_smiles", "")
                st.session_state["panura_editor_generation"] = st.session_state.get("panura_editor_generation", 0) + 1
            st.session_state["panura_previous_input_mode"] = input_mode

            smiles = ""
            if input_mode == "Enter SMILES":
                smiles = st.text_input(
                    "SMILES",
                    value=st.session_state.get("panura_last_valid_smiles", ""),
                    placeholder="Enter a valid canonical or isomeric SMILES",
                    key="panura_manual_input",
                ).strip()
            else:
                try:
                    from StreamJSME import StreamJSME
                    width = available_editor_width()
                    editor_height = max(360, min(470, round(width * 1.05)))
                    draw_value = StreamJSME(
                        smiles=st.session_state.get("panura_editor_seed", ""),
                        width=width, height=editor_height, margin=0,
                        key=f'panura_live_editor_{st.session_state.get("panura_editor_generation", 0)}',
                    )
                    smiles = draw_value.strip() if isinstance(draw_value, str) else ""
                except (ImportError, OSError) as exc:
                    st.error(f"Sketcher could not load: {exc}")
                    smiles = st.text_input("SMILES", key="panura_editor_fallback").strip()

            mol = parse_smiles(smiles)
            if mol is not None:
                st.session_state["panura_last_valid_smiles"] = smiles
            elif smiles:
                st.caption("Complete the molecular structure to calculate activity.")

    with results_col:
        section("02", "Prediction console", "AUTO UPDATE")
        st.markdown(
            '<p class="panura-helper">Urease inhibition · QSAR regression outputs</p>',
            unsafe_allow_html=True,
        )
        if mol is None:
            st.markdown(
                '<div class="panura-empty"><span class="symbol">⌬</span>'
                '<strong>Ready for a molecule</strong>'
                '<p>Draw a structure or enter SMILES. Predictions update without an Apply or Predict button.</p></div>',
                unsafe_allow_html=True,
            )
        elif assets is not None:
            try:
                result = cached_prediction(smiles)
                with st.container(key="panura_prediction_metrics"):
                    c1, c2 = st.columns(2, gap="small")
                    c1.metric("Predicted pIC₅₀", f'{result["pic50"]:.3f}')
                    c2.metric("Estimated IC₅₀", f'{result["ic50_um"]:.3g} µM')
                    if result["max_tanimoto"] is not None:
                        st.metric("Max. training-set Tanimoto", f'{result["max_tanimoto"]:.3f}')
            except Exception as exc:
                st.error(f"Prediction error: {exc}")

    # Keep the live output directly after the mobile editor.
    # Secondary descriptors and duplicate 2D preview are collapsed by default,
    # so phone users do not scroll past them to reach pIC50 and IC50.
    if mol is not None:
        with st.container(key="panura_structure_details"):
            with st.expander("Molecular properties and 2D structure", expanded=False):
                with st.container(key="panura_descriptor_metrics"):
                    m1, m2, m3 = st.columns(3, gap="small")
                    m1.metric("Molecular weight", f"{Descriptors.MolWt(mol):.1f}")
                    m2.metric("cLogP", f"{Descriptors.MolLogP(mol):.2f}")
                    m3.metric("H-bond donors", f"{Descriptors.NumHDonors(mol)}")
                try:
                    from rdkit.Chem import Draw
                    st.image(
                        Draw.MolToImage(mol, size=(620, 360)),
                        caption="2D molecular structure",
                        use_container_width=True,
                    )
                except (ImportError, OSError, RuntimeError):
                    st.caption("2D structure preview unavailable.")

elif page == "Batch Screening":
    # Tiny sample files show both accepted formats without occupying vertical space.
    sample_csv, sample_txt = st.columns(2, gap="small")
    with sample_csv:
        st.download_button(
            "Download example .CSV",
            data=(ROOT / "examples" / "PanURA_example.csv").read_bytes(),
            file_name="PanURA_example.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with sample_txt:
        st.download_button(
            "Download example .TXT",
            data=(ROOT / "examples" / "PanURA_example.txt").read_bytes(),
            file_name="PanURA_example.txt",
            mime="text/plain",
            use_container_width=True,
        )
    uploaded = st.file_uploader(
        "Upload CSV, TXT or Excel file containing SMILES",
        type=["csv", "txt", "xlsx"],
        help="TXT: ID and SMILES separated by tab, or one SMILES per line. CSV/XLSX: SMILES column, optional ID.",
    )
    if uploaded is not None:
        try:
            source_id = hashlib.sha256(uploaded.getvalue()).hexdigest()
            df = read_screening_file(uploaded.name, uploaded.getvalue())
            if df.empty or not len(df.columns):
                st.info("The uploaded dataset is empty.")
            else:
                # The identifier is metadata, never the QSAR input column.
                id_headers = {"id", "identifier", "compound_id", "molecule_id", "sample_id", "compoundid"}
                molecule_columns = [
                    c for c in df.columns
                    if str(c).strip().casefold() not in id_headers
                ]
                candidates = [
                    c for c in molecule_columns
                    if "smiles" in str(c).strip().casefold()
                ]
                if len(candidates) == 1 or len(molecule_columns) == 1:
                    selected = candidates[0] if candidates else molecule_columns[0]
                    st.caption(f"SMILES column: {selected}")
                else:
                    selected = st.selectbox(
                        "SMILES column",
                        molecule_columns,
                        index=molecule_columns.index(candidates[0]) if candidates else 0,
                    )
                st.dataframe(df.head(15), use_container_width=True, hide_index=True)
                id_column = df.columns[0]  # The importer keeps the original identifier first.
                similarity_available = bool(
                    assets is not None
                    and assets.training_fingerprints is not None
                    and len(assets.training_fingerprints) > 0
                )
                # Schema-version token invalidates old results with an empty
                # max_tanimoto column after an application upgrade.
                current_id = (source_id, selected, similarity_available, "activity-v2")
                if st.button("Run Screening", type="primary", disabled=assets is None):
                    rows = []
                    progress = st.progress(0)
                    for i, smi in enumerate(df[selected]):
                        try:
                            result = cached_prediction(str(smi))
                            record = {
                                "pIC50_pred": result["pic50"],
                                "IC50_uM_pred": result["ic50_um"],
                                "status": "OK",
                            }
                            if similarity_available:
                                record["max_tanimoto"] = result["max_tanimoto"]
                            rows.append(record)
                        except Exception as exc:
                            record = {
                                "pIC50_pred": float("nan"),
                                "IC50_uM_pred": float("nan"),
                                "status": str(exc),
                            }
                            if similarity_available:
                                record["max_tanimoto"] = float("nan")
                            rows.append(record)
                        progress.progress((i + 1) / len(df))
                    st.session_state["panura_batch"] = (
                        current_id,
                        pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1),
                    )
                cached_batch = st.session_state.get("panura_batch")
                if cached_batch is not None and cached_batch[0] == current_id:
                    output = cached_batch[1]
                    st.dataframe(output, use_container_width=True, hide_index=True)
                    if not similarity_available:
                        st.caption("Maximum Tanimoto is not shown: the original training fingerprints have not been supplied.")
                    successful = prepare_activity_data(output)
                    if not successful.empty:
                        st.caption(
                            f"Predicted activity · {len(successful):,} molecules · "
                            f"median pIC₅₀ {successful['pIC50_pred'].median():.2f}"
                        )
                        plot_config = {
                            "displaylogo": False,
                            "responsive": True,
                            "toImageButtonOptions": {
                                "filename": "PanURA_predicted_activity",
                                "format": "png",
                                "scale": 3,
                            },
                        }
                        if len(successful) <= 40:
                            st.plotly_chart(
                                molecule_activity_plot(successful, id_column, selected),
                                use_container_width=True,
                                config=plot_config,
                                key="panura_molecule_profile",
                            )
                            if len(successful) >= 20:
                                with st.expander("Activity distribution"):
                                    st.plotly_chart(
                                        activity_distribution_plot(successful),
                                        use_container_width=True,
                                        config=plot_config,
                                        key="panura_distribution_optional",
                                    )
                        else:
                            st.plotly_chart(
                                activity_distribution_plot(successful),
                                use_container_width=True,
                                config=plot_config,
                                key="panura_distribution_primary",
                            )
                            with st.expander("Molecule-by-molecule activity"):
                                st.plotly_chart(
                                    molecule_activity_plot(successful, id_column, selected),
                                    use_container_width=True,
                                    config=plot_config,
                                    key="panura_molecule_profile_optional",
                                )
                    st.download_button(
                        "Download predictions (CSV)",
                        data=output.to_csv(index=False).encode("utf-8"),
                        file_name="PanURA_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Could not read dataset: {exc}")

else:
    intro(
        "03 / MODEL INFORMATION",
        "Model & Methods",
        "Dataset characteristics, fingerprints and independent evaluation.",
    )
    a, b, c, d = st.columns(4)
    a.metric("Curated compounds", "351")
    b.metric("Selected fingerprints", "142")
    c.metric("Reported test R²", "0.782")
    d.metric("Reported test RMSE", "0.493")
    st.markdown(
        "**Target:** *Helicobacter pylori* urease inhibition.\n\n"
        "**Descriptors:** 881-bit PubChem fingerprints; 142 training-selected bits.\n\n"
        "**Model:** Original cross-validation-selected LightGBM regressor "
        "(`Final_Model.pkl`), without re-training or post-selection optimization.\n\n"
        "**Reported evaluation:** Kennard–Stone 280/71 development/test split; "
        "test R² ≈ 0.782 and test RMSE ≈ 0.493 pIC₅₀ units.\n\n"
        "**Predictions:** pIC₅₀ and IC₅₀ in micromolar; optional maximum "
        "training-set Tanimoto similarity when training fingerprints are available."
    )
