"""Scientific batch activity charts with legible molecule-level data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

NAVY = "#17385f"
TEAL = "#158e97"
MUTED = "#607489"
GRID = "#e5edf3"
CONTINUOUS = [
    [0, "#8acfd3"],
    [0.5, "#2696a5"],
    [1, "#173e77"],
]


def prepare_activity_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only finite pIC50 predictions, preserving molecular IDs."""
    if "pIC50_pred" not in frame:
        raise ValueError("Missing pIC50_pred column.")
    data = frame.copy()
    data["pIC50_pred"] = pd.to_numeric(data["pIC50_pred"], errors="coerce")
    data = data.loc[np.isfinite(data["pIC50_pred"].to_numpy(dtype=float))].copy()
    return data


def _base_layout(fig: go.Figure, height: int, lmargin: int = 55) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        margin=dict(l=lmargin, r=22, t=23, b=58),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color=NAVY, size=12),
        showlegend=False,
        hoverlabel=dict(bgcolor="#163b63", font=dict(color="white", size=12)),
        dragmode=False,
    )
    fig.update_xaxes(showline=True, linecolor="#b6c9d9", linewidth=1,
                     gridcolor=GRID, zeroline=False, ticks="outside", ticklen=4)
    fig.update_yaxes(showline=False, gridcolor=GRID, zeroline=False)
    return fig


def molecule_activity_plot(data: pd.DataFrame, id_col: str, smiles_col: str) -> go.Figure:
    """Ranked horizontal dot plot: each point is a molecule, not a fabricated trend."""
    if data.empty:
        raise ValueError("No valid predictions for a molecule activity chart.")
    sorted_data = data.sort_values("pIC50_pred", ascending=False, kind="stable").reset_index(drop=True)
    values = sorted_data["pIC50_pred"].to_numpy(dtype=float)
    ids = sorted_data[id_col].astype(str).tolist()
    smiles = sorted_data[smiles_col].astype(str).tolist()
    ic50 = pd.to_numeric(sorted_data["IC50_uM_pred"], errors="coerce").to_numpy()
    rank = np.arange(len(values))
    custom = [[ids[i], smiles[i], float(ic50[i])] for i in range(len(values))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values,
        y=rank,
        mode="markers",
        marker=dict(
            size=13, color=values, colorscale=CONTINUOUS,
            showscale=False, line=dict(color="#ffffff", width=1.5),
        ),
        customdata=custom,
        hovertemplate="<b>%{customdata[0]}</b><br>"
                      "Predicted pIC₅₀: %{x:.3f}<br>"
                      "Estimated IC₅₀: %{customdata[2]:.3g} µM<br>"
                      "SMILES: %{customdata[1]}<extra></extra>",
    ))
    median = float(np.median(values))
    fig.add_vline(x=median, line_width=1.35, line_color="#d68a43",
                  line_dash="dash",
                  annotation_text=f"Median {median:.2f}",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="#9b662f"))
    low, high = float(values.min()), float(values.max())
    padding = max((high - low) * 0.14, 0.18)
    fig.update_xaxes(
        title=dict(text="Predicted pIC₅₀ (−log₁₀ M)", font=dict(size=12)),
        range=[low - padding, high + padding],
        tickformat=".2f",
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=rank.tolist(),
        ticktext=[value if len(value) <= 19 else value[:16] + "…" for value in ids],
        autorange="reversed",
        fixedrange=False,
        showgrid=True,
    )
    height = min(980, max(315, len(values) * 33 + 105))
    _base_layout(fig, height=height, lmargin=110)
    fig.update_yaxes(tickfont=dict(size=11))
    return fig


def activity_distribution_plot(data: pd.DataFrame) -> go.Figure:
    """Adaptive frequency histogram, only for datasets large enough to bin."""
    if data.empty:
        raise ValueError("No valid predictions for a distribution chart.")
    values = data["pIC50_pred"].to_numpy(dtype=float)
    n = len(values)
    if n < 20:
        raise ValueError("At least 20 predictions are required for a meaningful histogram.")
    spread = float(np.ptp(values))
    edges = np.histogram_bin_edges(values, bins="fd") if spread > 0 else np.array([values[0] - 0.2, values[0] + 0.2])
    bins = max(5, min(35, len(edges) - 1)) if spread > 0 else 1
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values, nbinsx=bins,
        marker=dict(color=TEAL, line=dict(color="#ffffff", width=1)),
        opacity=0.92,
        hovertemplate="Predicted pIC₅₀ bin: %{x:.2f}<br>"
                      "Molecules: %{y}<extra></extra>",
    ))
    median = float(np.median(values))
    fig.add_vline(x=median, line_width=1.4, line_dash="dash", line_color="#bd7b3c",
                  annotation_text=f"Median {median:.2f}",
                  annotation_position="top right",
                  annotation_font=dict(color="#95612d", size=10))
    fig.update_xaxes(title="Predicted pIC₅₀ (−log₁₀ M)", tickformat=".2f")
    fig.update_yaxes(title="Number of molecules", tickformat="d", rangemode="tozero")
    return _base_layout(fig, height=395, lmargin=67)
