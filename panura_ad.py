"""Prospective leverage applicability domain reproduced from the original training split."""
from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LeverageDomain:
    selected_training: np.ndarray
    training_pseudoinverse: np.ndarray
    cutoff_p95: float
    cutoff_p99: float
    conventional_cutoff: float


def load_leverage_domain(path: Path, selected_features: list[str]) -> LeverageDomain:
    """Use the original intercept-augmented Moore-Penrose leverage calculation."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    n, p = payload["shape"]
    if n != 280 or p != 142 or payload["feature_names"] != selected_features:
        raise ValueError("AD reference does not match the original LightGBM feature manifest.")
    compressed = base64.b64decode(payload["matrix_zlib_base64"], validate=True)
    packed = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8)
    bytes_per_row = (p + 7) // 8
    if packed.size != n * bytes_per_row:
        raise ValueError("Invalid AD reference matrix size.")
    train = np.unpackbits(packed.reshape(n, bytes_per_row), axis=1)[:, :p].astype(float)
    augmented = np.column_stack([np.ones(n), train])
    pseudo = np.linalg.pinv(augmented)
    h_train = np.einsum("ij,ji->i", augmented, pseudo)
    return LeverageDomain(
        selected_training=train,
        training_pseudoinverse=pseudo,
        cutoff_p95=float(np.quantile(h_train, 0.95)),
        cutoff_p99=float(np.quantile(h_train, 0.99)),
        conventional_cutoff=float(3 * (p + 1) / n),
    )


def assess_leverage(selected_bits: np.ndarray, domain: LeverageDomain) -> dict[str, float | bool]:
    """Compute structural leverage only; no unknown experimental residual is used."""
    row = np.asarray(selected_bits, dtype=float).reshape(1, -1)
    if row.shape[1] != domain.selected_training.shape[1]:
        raise ValueError("AD descriptor dimension mismatch.")
    projection = np.column_stack((np.ones(1), row)) @ domain.training_pseudoinverse
    h = float(np.sum(projection ** 2))
    outside = h > domain.cutoff_p95 and not np.isclose(
        h, domain.cutoff_p95, atol=1e-10, rtol=1e-9
    )
    return {"leverage": h, "ad_inside": not outside, "ad_cutoff": domain.cutoff_p95}


def max_selected_tanimoto(selected_bits: np.ndarray, domain: LeverageDomain) -> float:
    """Optional similarity on selected 142 bits only, NOT full 881-bit Tanimoto."""
    query = np.asarray(selected_bits, dtype=bool).reshape(-1)
    reference = domain.selected_training.astype(bool)
    both = np.count_nonzero(reference & query, axis=1)
    either = np.count_nonzero(reference | query, axis=1)
    scores = np.divide(both, either, out=np.zeros_like(both, dtype=float), where=either > 0)
    return float(scores.max())
