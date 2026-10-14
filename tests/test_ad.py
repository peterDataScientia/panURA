"""Leverage AD regression tests against the original 280-compound reference."""
from pathlib import Path
import json
import numpy as np
from panura_ad import load_leverage_domain, assess_leverage, max_selected_tanimoto

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "ad_reference.json"

def test_original_training_ad_reproduces_manuscript():
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    domain = load_leverage_domain(SOURCE, raw["feature_names"])
    assert domain.selected_training.shape == (280, 142)
    assert abs(domain.cutoff_p95 - 0.98752174387856) < 1e-7
    assert abs(domain.conventional_cutoff - (3 * 143 / 280)) < 1e-12
    results = [assess_leverage(row, domain) for row in domain.selected_training]
    assert sum(not item["ad_inside"] for item in results) == 9
    assert all(np.isfinite(item["leverage"]) for item in results)

def test_identical_training_vector_has_perfect_selected_similarity():
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    domain = load_leverage_domain(SOURCE, raw["feature_names"])
    assert max_selected_tanimoto(domain.selected_training[0], domain) == 1.0

def test_manifest_order_is_checked():
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    names = raw["feature_names"].copy()
    names[0], names[1] = names[1], names[0]
    import pytest
    with pytest.raises(ValueError, match="manifest"):
        load_leverage_domain(SOURCE, names)
