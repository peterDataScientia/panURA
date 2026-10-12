"""Regression tests for the unchanged PanURA fingerprint-to-model contract."""
import numpy as np
import pytest

from panura_engine import PredictorAssets, parse_smiles, predict_smiles


class FakeFingerprint:
    def transform(self, values):
        assert len(values) == 1
        row = np.zeros((1, 881), dtype=float)
        row[0, 3] = 1.0
        row[0, 9] = 1.0
        return row


class FakeModel:
    def __init__(self):
        self.last_input = None

    def predict(self, x):
        self.last_input = np.asarray(x)
        return np.array([6.0])


def assets(training=None):
    return PredictorAssets(
        model=FakeModel(),
        indices=np.asarray([9, 3], dtype=np.intp),
        fingerprints=FakeFingerprint(),
        training_fingerprints=training,
    )


def test_smiles_validation_and_partial_drawings():
    assert parse_smiles("CCO") is not None
    assert parse_smiles("") is None
    assert parse_smiles(None) is None
    assert parse_smiles("not-valid") is None


def test_prediction_uses_selected_bits_in_training_order():
    ref = assets()
    result = predict_smiles("CCO", ref)
    np.testing.assert_array_equal(ref.model.last_input, [[1.0, 1.0]])
    assert result["pic50"] == pytest.approx(6.0)
    assert result["ic50_um"] == pytest.approx(1.0)
    assert result["max_tanimoto"] is None


def test_optional_similarity_matches_tanimoto_definition():
    train = np.zeros((2, 881), dtype=bool)
    train[0, 3] = True
    train[0, 9] = True
    train[1, 3] = True
    result = predict_smiles("CCO", assets(train))
    assert result["max_tanimoto"] == pytest.approx(1.0)


def test_invalid_molecule_is_rejected_before_model():
    with pytest.raises(ValueError, match="Invalid molecular"):
        predict_smiles("!!!", assets())


def test_non_finite_prediction_rejected():
    ref = assets()
    ref.model.predict = lambda x: np.array([np.nan])
    with pytest.raises(ValueError, match="non-finite"):
        predict_smiles("CCO", ref)


def test_bundled_final_model_predicts_without_refitting():
    """Smoke-test the real shipped pickle, manifest and fingerprint generator."""
    from pathlib import Path
    from panura_engine import load_assets

    model_dir = Path(__file__).resolve().parents[1] / "models"
    original = load_assets(model_dir)
    assert original.indices.shape == (142,)
    result = predict_smiles("CCO", original)
    assert np.isfinite(result["pic50"])
    assert result["ic50_um"] > 0
