"""PanURA · live urease-inhibitor QSAR screening workspace."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
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
    # One synchronized workspace: paste SMILES, draw/edit, and predict live.
    # A new editor generation is used only when the user deliberately loads a
    # different molecule. Ordinary drawing callbacks never remount the editor.
    for key, default in (
        ("panura_editor_seed", ""), ("panura_active_smiles", ""),
        ("panura_manual_smiles", ""), ("panura_editor_generation", 0),
        ("panura_editor_initialized", False), ("panura_editor_last_emitted", ""),
        ("panura_design_history", []),
    ):
        if key not in st.session_state:
            st.session_state[key] = default

    if "panura_pending_smiles_sync" in st.session_state:
        st.session_state["panura_manual_smiles"] = st.session_state.pop("panura_pending_smiles_sync")

    def load_structure_into_editor(value: str):
        smi = value.strip()
        st.session_state["panura_editor_seed"] = smi
        st.session_state["panura_active_smiles"] = smi
        st.session_state["panura_manual_smiles"] = smi
        st.session_state["panura_editor_last_emitted"] = smi
        st.session_state["panura_editor_initialized"] = False
        st.session_state["panura_editor_generation"] += 1

    def accept_pasted_smiles():
        smi = st.session_state["panura_manual_smiles"].strip()
        if not smi or parse_smiles(smi) is not None:
            load_structure_into_editor(smi)

    editor_col, results_col = st.columns([1.18, 1], gap="large", vertical_alignment="top")
    with editor_col:
        with st.container(key="panura_editor_column"):
            section("01", "Molecular editor", "LIVE DESIGN")
            st.text_input(
                "Paste SMILES to display and edit its molecular structure",
                key="panura_manual_smiles",
                placeholder="Paste an existing structure, or draw a new molecule below",
                on_change=accept_pasted_smiles,
            )
            if st.session_state["panura_manual_smiles"] and parse_smiles(st.session_state["panura_manual_smiles"]) is None:
                st.warning("The SMILES is incomplete or invalid; the previous valid structure is retained.")
            try:
                from StreamJSME import StreamJSME
                width = available_editor_width()
                editor_height = max(360, min(470, round(width * 1.05)))
                draw_value = StreamJSME(
                    smiles=st.session_state["panura_editor_seed"],
                    width=width, height=editor_height, margin=0,
                    key=f'panura_design_editor_{st.session_state["panura_editor_generation"]}',
                )
                if isinstance(draw_value, str):
                    drawn = draw_value.strip()
                    # Some JSME builds return an empty value before a seeded
                    # canvas has initialized. Do not erase the pasted molecule.
                    initial_empty = bool(
                        not drawn and st.session_state["panura_editor_seed"]
                        and not st.session_state["panura_editor_initialized"]
                    )
                    if not initial_empty:
                        st.session_state["panura_editor_initialized"] = True
                        if drawn != st.session_state["panura_editor_last_emitted"]:
                            st.session_state["panura_editor_last_emitted"] = drawn
                            st.session_state["panura_active_smiles"] = drawn
                            if drawn != st.session_state["panura_manual_smiles"]:
                                # Sync the editable SMILES field on the next natural
                                # event; avoid an extra rerun that remounts JSME.
                                st.session_state["panura_pending_smiles_sync"] = drawn
            except (ImportError, OSError) as exc:
                st.warning(f"Interactive editor unavailable: {exc}")
            st.button("Clear structure", key="panura_clear_structure", on_click=load_structure_into_editor, args=("",))
            smiles = st.session_state["panura_active_smiles"]
            mol = parse_smiles(smiles)
            if mol is not None:
                st.caption(f"Current structure SMILES: {Chem.MolToSmiles(mol, isomericSmiles=True)}")
            if smiles and mol is None:
                st.caption("Complete the molecular structure to calculate activity.")

    result = None
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
                '<p>Paste SMILES or draw a structure. Predictions update after every valid change.</p></div>',
                unsafe_allow_html=True,
            )
        elif assets is not None:
            try:
                result = cached_prediction(smiles)
                with st.container(key="panura_prediction_metrics"):
                    c1, c2 = st.columns(2, gap="small")
                    c1.metric("Predicted pIC₅₀", f'{result["pic50"]:.3f}')
                    c2.metric("Estimated IC₅₀", f'{result["ic50_um"]:.3g} µM')
                    if result.get("ad_inside") is not None:
                        status = "Inside" if result["ad_inside"] else "Outside"
                        a, b = st.columns(2)
                        a.metric("Training-space AD", status)
                        b.metric("Leverage", f'{result["leverage"]:.3f}')
                        st.caption(
                            f'Leverage P95 cutoff: {result["ad_cutoff"]:.3f}. '
                            'Prospective structural domain only; no experimental residual is assumed.'
                        )
                        if not result["ad_inside"]:
                            st.warning(
                                "Outside the training leverage domain. Treat potency estimates "
                                "as exploratory and prioritize experimental verification."
                            )
                        if result.get("selected_tanimoto") is not None:
                            st.caption(
                                f'Maximum Tanimoto on 142 selected fingerprints: '
                                f'{result["selected_tanimoto"]:.3f} (supplementary similarity, not AD).'
                            )
                    else:
                        st.caption("Training-space applicability domain is unavailable.")
                    if result.get("max_tanimoto") is not None:
                        st.caption(f'Maximum 881-bit training Tanimoto: {result["max_tanimoto"]:.3f}')
            except Exception as exc:
                st.error(f"Prediction error: {exc}")

    if mol is not None:
        with st.container(key="panura_structure_details"):
            with st.expander("Molecular properties and 2D structure", expanded=False):
                m1, m2, m3 = st.columns(3, gap="small")
                m1.metric("Molecular weight", f"{Descriptors.MolWt(mol):.1f}")
                m2.metric("cLogP", f"{Descriptors.MolLogP(mol):.2f}")
                m3.metric("H-bond donors", f"{Descriptors.NumHDonors(mol)}")
                try:
                    from rdkit.Chem import Draw
                    st.image(Draw.MolToImage(mol, size=(620, 360)),
                             caption="2D molecular structure", use_container_width=True)
                except (ImportError, OSError, RuntimeError):
                    st.caption("2D structure preview unavailable.")

    st.divider()
    section("03", "Design history and comparison", "SAVE VARIANTS")
    st.caption(
        "Modify the drawing, observe updated predictions, and save candidates for comparison. "
        "Predicted improvements are hypotheses, not confirmed experimental optimization."
    )
    history = st.session_state["panura_design_history"]

    def save_current_design():
        # A widget callback runs before Streamlit reconstructs the page.
        # Recalculate from the current structure so saved values never refer
        # to an outdated screen or an earlier asynchronous editor event.
        current = st.session_state.get("panura_active_smiles", "").strip()
        current_mol = parse_smiles(current)
        if current_mol is None or assets is None:
            st.session_state["panura_design_notice"] = "No valid molecule to save."
            return
        actual = cached_prediction(current)
        canonical = Chem.MolToSmiles(current_mol, isomericSmiles=True)
        records = st.session_state["panura_design_history"]
        if any(item["Canonical_SMILES"] == canonical for item in records):
            st.session_state["panura_design_notice"] = "This design is already saved."
            return
        records.append({
            "Design": f"Design {len(records) + 1}",
            "SMILES": current,
            "Canonical_SMILES": canonical,
            "Predicted_pIC50": round(actual["pic50"], 4),
            "Estimated_IC50_uM": round(actual["ic50_um"], 5),
            "AD": ("Inside" if actual["ad_inside"] else "Outside")
                  if actual.get("ad_inside") is not None else "Unavailable",
            "Leverage": round(actual["leverage"], 5)
                        if actual.get("leverage") is not None else None,
        })
        st.session_state["panura_design_notice"] = "Design saved."

    st.button("Save current design", disabled=result is None,
              key="panura_save_design", on_click=save_current_design)
    if st.session_state.get("panura_design_notice"):
        st.caption(st.session_state["panura_design_notice"])
    if history:
        base = history[0]["Predicted_pIC50"]
        comparison = pd.DataFrame(history)
        comparison["Delta_pIC50_vs_first"] = (
            comparison["Predicted_pIC50"] - base
        ).round(4)
        st.dataframe(
            comparison.drop(columns=["Canonical_SMILES"]),
            use_container_width=True, hide_index=True,
        )
        labels = [f'{item["Design"]} · {item["Predicted_pIC50"]:.3f}' for item in history]
        selected_design = st.selectbox("Restore a saved molecular structure", labels, key="panura_restore_choice")
        r1, r2 = st.columns(2)
        with r1:
            st.button("Load selected design into editor", key="panura_restore_design",
                      on_click=load_structure_into_editor,
                      args=(history[labels.index(selected_design)]["SMILES"],))
        with r2:
            st.download_button(
                "Download design comparison (.csv)",
                comparison.to_csv(index=False).encode("utf-8"),
                file_name="PanURA_design_comparison.csv", mime="text/csv",
                on_click="ignore", key="panura_design_export",
            )

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
                current_id = (source_id, selected, similarity_available, "activity-v3-ad")
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
                                "AD_status": ("Inside" if result["ad_inside"] else "Outside") if result.get("ad_inside") is not None else "Unavailable",
                                "leverage": result.get("leverage"),
                                "selected_bit_tanimoto": result.get("selected_tanimoto"),
                            }
                            if similarity_available:
                                record["max_tanimoto"] = result["max_tanimoto"]
                            rows.append(record)
                        except Exception as exc:
                            record = {
                                "pIC50_pred": float("nan"),
                                "IC50_uM_pred": float("nan"),
                                "status": str(exc),
                                "AD_status": "Unavailable",
                                "leverage": float("nan"),
                                "selected_bit_tanimoto": float("nan"),
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
                    st.caption("AD uses original training-set P95 leverage. Selected-bit Tanimoto is a supplementary proximity score, not an AD probability.")
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
        "**Predictions:** pIC₅₀, IC₅₀ in micromolar, and original training-set "
        "P95 leverage AD, with selected-fingerprint Tanimoto reported separately. "
        "A new molecule has no experimental residual, so residual-based AD is not applicable."
    )
