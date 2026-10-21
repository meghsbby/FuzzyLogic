"""
Interactive Plotly visualizations for the Fuzzy Logic System.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from fuzzy_engine import LinguisticVariable, MamdaniEngine


COLOR_PALETTE = [
    "#3B82F6",  # Blue
    "#10B981",  # Emerald Green
    "#F59E0B",  # Amber
    "#EF4444",  # Red
    "#8B5CF6",  # Purple
    "#06B6D4",  # Cyan
    "#EC4899",  # Pink
]


def plot_linguistic_variable(
    var: LinguisticVariable,
    current_val: Optional[float] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Generate interactive Plotly chart of membership functions for a linguistic variable,
    with an optional vertical indicator for the current crisp value.
    """
    fig = go.Figure()
    x = var.get_universe_array()

    memberships = var.fuzzify(current_val) if current_val is not None else {}

    for idx, (term_name, mf) in enumerate(var.terms.items()):
        y = mf.evaluate(x)
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{term_name}",
                line=dict(width=2.5, color=color),
                hovertemplate=f"<b>{term_name}</b><br>{var.name}: %{{x:.1f}} {var.unit}<br>Degree μ: %{{y:.2f}}<extra></extra>",
            )
        )

        # Highlight point on curve if current_val provided
        if current_val is not None and term_name in memberships:
            deg = memberships[term_name]
            if deg > 0.01:
                fig.add_trace(
                    go.Scatter(
                        x=[current_val],
                        y=[deg],
                        mode="markers+text",
                        marker=dict(size=9, color=color, symbol="circle", line=dict(width=1.5, color="#ffffff")),
                        text=[f" {deg:.2f}"],
                        textposition="top right",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    # Crisp value indicator line
    if current_val is not None:
        fig.add_vline(
            x=current_val,
            line_width=2,
            line_dash="dash",
            line_color="#E2E8F0",
            annotation_text=f"Input: {current_val:.1f} {var.unit}",
            annotation_position="top left",
        )

    fig.update_layout(
        title=dict(text=title or f"{var.name} ({var.unit}) Membership Functions", font=dict(size=14, color="#E2E8F0")),
        xaxis=dict(
            title=f"{var.name} [{var.unit}]",
            gridcolor="#334155",
            zerolinecolor="#475569",
            color="#94A3B8",
        ),
        yaxis=dict(
            title="Membership Degree μ",
            range=[-0.05, 1.1],
            gridcolor="#334155",
            zerolinecolor="#475569",
            color="#94A3B8",
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=280,
    )

    return fig


def plot_defuzzification(
    u_grid: np.ndarray,
    aggregated_mf: np.ndarray,
    defuzz_val: float,
    var_name: str,
    unit: str,
    method: str,
) -> go.Figure:
    """
    Plot the aggregated fuzzy output set and the defuzzified crisp center point.
    """
    fig = go.Figure()

    # Shaded Area under Aggregated Fuzzy Set
    fig.add_trace(
        go.Scatter(
            x=u_grid,
            y=aggregated_mf,
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.25)",
            mode="lines",
            line=dict(color="#3B82F6", width=2.5),
            name="Aggregated Fuzzy Set",
            hovertemplate=f"<b>{var_name}</b><br>Value: %{{x:.1f}} {unit}<br>Aggregated μ: %{{y:.2f}}<extra></extra>",
        )
    )

    # Defuzzified crisp value vertical line
    fig.add_vline(
        x=defuzz_val,
        line_width=3,
        line_dash="solid",
        line_color="#10B981",
        annotation_text=f"{method.upper()}: {defuzz_val:.1f} {unit}",
        annotation_position="top right",
        annotation_font=dict(color="#10B981", size=13),
    )

    fig.update_layout(
        title=dict(text=f"{var_name} Defuzzification ({method.capitalize()})", font=dict(size=14, color="#E2E8F0")),
        xaxis=dict(title=f"{var_name} [{unit}]", gridcolor="#334155", color="#94A3B8"),
        yaxis=dict(title="Aggregated Degree μ", range=[-0.05, 1.1], gridcolor="#334155", color="#94A3B8"),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=280,
    )

    return fig


def plot_3d_control_surface(
    engine: MamdaniEngine,
    x_var_name: str,
    y_var_name: str,
    output_var_name: str,
    fixed_inputs: Dict[str, float],
    grid_size: int = 24,
) -> go.Figure:
    """
    Compute and plot an interactive 3D Control Surface showing how output varies
    continuously across combinations of two input variables.
    """
    x_var = engine.inputs[x_var_name]
    y_var = engine.inputs[y_var_name]
    out_var = engine.outputs[output_var_name]

    x_vals = np.linspace(x_var.universe_min, x_var.universe_max, grid_size)
    y_vals = np.linspace(y_var.universe_min, y_var.universe_max, grid_size)

    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(grid_size):
        for j in range(grid_size):
            inp = dict(fixed_inputs)
            inp[x_var_name] = float(X[i, j])
            inp[y_var_name] = float(Y[i, j])
            res = engine.compute(inp)
            Z[i, j] = res.crisp_outputs[output_var_name]

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="Viridis",
                colorbar=dict(title=f"{output_var_name} ({out_var.unit})"),
                hovertemplate=(
                    f"<b>{x_var_name}</b>: %{{x:.1f}} {x_var.unit}<br>"
                    f"<b>{y_var_name}</b>: %{{y:.1f}} {y_var.unit}<br>"
                    f"<b>{output_var_name}</b>: %{{z:.1f}} {out_var.unit}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"3D Fuzzy Control Surface: {output_var_name} vs ({x_var_name}, {y_var_name})",
            font=dict(size=15, color="#E2E8F0"),
        ),
        scene=dict(
            xaxis=dict(title=f"{x_var_name} ({x_var.unit})", gridcolor="#334155", color="#94A3B8"),
            yaxis=dict(title=f"{y_var_name} ({y_var.unit})", gridcolor="#334155", color="#94A3B8"),
            zaxis=dict(title=f"{output_var_name} ({out_var.unit})", gridcolor="#334155", color="#94A3B8"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=520,
    )

    return fig
