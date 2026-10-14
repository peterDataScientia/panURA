"""Pure PanURA QSAR inference functions; no Streamlit UI state here."""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from rdkit import Chem
from skfp.fingerprints import PubChemFingerprint
from panura_ad import LeverageDomain, load_leverage_domain, assess_leverage, max_selected_tanimoto

PUBCHEM_BITS = 881
SELECTED_BITS = 142


@dataclass
class PredictorAssets:
    model: Any
    indices: np.ndarray
    fingerprints: PubChemFingerprint
    training_fingerprints: np.ndarray | None = None
    ad_domain: LeverageDomain | None = None


def parse_smiles(value: object):
    """Return an RDKit molecule or None (including incomplete drawing states)."""
    if not isinstance(value, str) or not value.strip():
        return None
    return Chem.MolFromSmiles(value.strip())


def load_assets(model_dir: Path) -> PredictorAssets:
    """Load only the original final model and its feature manifest."""
    model_dir = Path(model_dir)
    model_path = model_dir / "Final_Model.pkl.b64"
    manifest_path = model_dir / "Feature_Selection_Manifest.json.b64"
    model = joblib.load(io.BytesIO(base64.b64decode(model_path.read_text(encoding="ascii"))))
    manifest = json.loads(base64.b64decode(manifest_path.read_text(encoding="ascii")))
    names = manifest["selected_features"]
    if (
        not isinstance(names, list)
        or len(names) != SELECTED_BITS
        or len(set(names)) != SELECTED_BITS
        or any(not isinstance(n, str) or not n.startswith("PubChem_") for n in names)
    ):
        raise ValueError("The final training manifest must specify 142 distinct PubChem bits.")
    # Preserve the original model's zero-based descriptor indexing convention.
    indices = np.asarray([int(n.rsplit("_", 1)[1]) for n in names], dtype=np.intp)
    if np.any(indices < 0) or np.any(indices >= PUBCHEM_BITS):
        raise ValueError("PubChem descriptor index outside 0..880.")
    if getattr(model, "n_features_in_", SELECTED_BITS) != SELECTED_BITS:
        raise ValueError("Loaded final model has a different feature count.")
    training_path = model_dir / "train_fingerprints.npy"
    training = None
    if training_path.exists():
        training = np.load(training_path, allow_pickle=False)
        if training.ndim != 2 or training.shape[1] != PUBCHEM_BITS:
            raise ValueError("Training fingerprints must have 881 columns.")
        training = np.asarray(training, dtype=bool)
    ad_path = model_dir / "ad_reference.json"
    ad = load_leverage_domain(ad_path, names) if ad_path.exists() else None
    return PredictorAssets(model, indices, PubChemFingerprint(), training, ad)


def predict_smiles(value: str, assets: PredictorAssets) -> dict[str, float | None]:
    """Calculate predictions from the final model; never refit or optimize it."""
    if parse_smiles(value) is None:
        raise ValueError("Invalid molecular structure.")
    full_fp = np.asarray(assets.fingerprints.transform([value.strip()]), dtype=np.float64)
    if full_fp.shape != (1, PUBCHEM_BITS):
        raise ValueError(f"Unexpected PubChem fingerprint shape: {full_fp.shape}.")
    selected = full_fp[:, assets.indices]
    pic50 = float(assets.model.predict(selected)[0])
    if not np.isfinite(pic50):
        raise ValueError("Model returned a non-finite prediction.")
    similarity = None
    if assets.training_fingerprints is not None and len(assets.training_fingerprints):
        probe = full_fp[0].astype(bool)
        ref = assets.training_fingerprints
        intersection = np.count_nonzero(ref & probe, axis=1)
        union = np.count_nonzero(ref | probe, axis=1)
        similarities = np.divide(
            intersection, union, out=np.zeros_like(intersection, dtype=float),
            where=union != 0,
        )
        similarity = float(np.max(similarities))
    ad_result = assess_leverage(selected[0], assets.ad_domain) if assets.ad_domain is not None else {}
    selected_similarity = max_selected_tanimoto(selected[0], assets.ad_domain) if assets.ad_domain is not None else None
    return {
        "pic50": pic50, "ic50_um": 10 ** (6 - pic50), "max_tanimoto": similarity,
        "selected_tanimoto": selected_similarity,
        "leverage": ad_result.get("leverage"),
        "ad_cutoff": ad_result.get("ad_cutoff"),
        "ad_inside": ad_result.get("ad_inside"),
    }
