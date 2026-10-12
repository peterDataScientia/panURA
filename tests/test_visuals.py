"""PanURA chart integrity: scientific axes, IDs, and adaptive distributions."""
import numpy as np
import pandas as pd
import pytest

from panura_visuals import (
    prepare_activity_data,
    molecule_activity_plot,
    activity_distribution_plot,
)


def sample_frame(n: int = 6):
    return pd.DataFrame({
        "ID": [f"URA_{k:03d}" for k in range(1, n + 1)],
        "SMILES": ["CCO"] * n,
        "pIC50_pred": np.linspace(4.2, 6.5, n),
        "IC50_uM_pred": 10 ** (6 - np.linspace(4.2, 6.5, n)),
    })


def test_molecule_plot_contains_ids_and_true_predictions():
    frame = sample_frame(6)
    fig = molecule_activity_plot(prepare_activity_data(frame), "ID", "SMILES")
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert len(fig.data[0].x) == 6
    assert list(fig.layout.yaxis.ticktext) == list(reversed(frame["ID"].tolist()))
    assert "SMILES" in fig.data[0].hovertemplate
    assert "Predicted pIC₅₀" in fig.layout.xaxis.title.text
    # IDs and predictions are paired after sort, not mixed by rank.
    for index, row in enumerate(fig.data[0].customdata):
        assert row[0] == fig.layout.yaxis.ticktext[index]
        original = frame.loc[frame["ID"] == row[0], "pIC50_pred"].iloc[0]
        assert float(fig.data[0].x[index]) == pytest.approx(original)


def test_invalid_prediction_rows_are_excluded_from_visualization():
    frame = sample_frame()
    frame.loc[1, "pIC50_pred"] = float("nan")
    frame.loc[2, "pIC50_pred"] = float("inf")
    selected = prepare_activity_data(frame)
    assert len(selected) == 4
    assert "URA_002" not in selected.ID.tolist()
    assert "URA_003" not in selected.ID.tolist()


def test_distribution_requires_sufficient_sample_size():
    with pytest.raises(ValueError, match="At least 20"):
        activity_distribution_plot(prepare_activity_data(sample_frame(6)))


def test_large_sample_uses_frequency_histogram():
    fig = activity_distribution_plot(prepare_activity_data(sample_frame(84)))
    assert len(fig.data) == 1
    assert fig.data[0].type == "histogram"
    assert len(fig.data[0].x) == 84
    assert "Number of molecules" in fig.layout.yaxis.title.text


def test_identical_predictions_still_render_distribution():
    frame = sample_frame(50)
    frame["pIC50_pred"] = 5.2
    fig = activity_distribution_plot(prepare_activity_data(frame))
    assert fig.data[0].nbinsx == 1


def test_empty_valid_predictions_rejected():
    frame = sample_frame(2)
    frame["pIC50_pred"] = np.nan
    with pytest.raises(ValueError, match="No valid"):
        molecule_activity_plot(prepare_activity_data(frame), "ID", "SMILES")
