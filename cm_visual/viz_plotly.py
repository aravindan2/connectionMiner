from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import CmResult, PrepData, RawData


def run_all_visualizations(
    raw: RawData,
    prep: PrepData,
    cm: CmResult,
    cfg: dict[str, Any],
    meta_dfs: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Viz 1/6 — Raw clusters on UMAP ...")
    _viz_raw_clusters(raw, meta_dfs, output_dir)

    print("  Viz 2/6 — Cell type constraints on UMAP ...")
    _viz_cell_constraints(raw, meta_dfs, output_dir)

    print("  Viz 3/6 — Metacells on UMAP ...")
    _viz_metacells(raw, prep, meta_dfs, output_dir)

    print("  Viz 4/6 — Inferred type assignments on UMAP ...")
    _viz_inferred_types(raw, prep, cm, meta_dfs, output_dir)

    print("  Viz 5/6 — Connectome reconstruction: C vs C_hat ...")
    _viz_connectome_fit(prep, cm, meta_dfs, output_dir)

    print("  Viz 6/6 — Solver loss trajectory ...")
    _viz_loss_trajectory(cm, output_dir)

    print(f"  All visualizations saved to {output_dir}")


def _viz_raw_clusters(
    raw: RawData,
    meta_dfs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    cell_index = meta_dfs["cell_index"]
    barcodes = cell_index["cell_barcode"].values.astype(str)
    clusters = raw.raw_cluster_id
    clusters_str = np.where(pd.isna(clusters), "NaN", clusters.astype(int).astype(str))
    annotations = cell_index["MultiomeAnnotated"].values.astype(str) if "MultiomeAnnotated" in cell_index.columns else [""] * len(barcodes)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=raw.umap_xy[:, 0],
            y=raw.umap_xy[:, 1],
            mode="markers",
            marker=dict(
                size=2,
                color=clusters,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Cluster ID"),
            ),
            text=[
                f"Barcode: {b}<br>Cluster: {c}<br>Annotation: {a}"
                for b, c, a in zip(barcodes, clusters_str, annotations)
            ],
            hoverinfo="text",
            name="Cells",
        )
    )
    fig.update_layout(
        title="Raw MultiomeNN Clusters on UMAP",
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        width=900,
        height=700,
        hovermode="closest",
    )
    fig.write_html(output_dir / "viz_01_raw_clusters.html", include_plotlyjs="cdn")


def _viz_cell_constraints(
    raw: RawData,
    meta_dfs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    cell_index = meta_dfs["cell_index"]
    tiers = cell_index["cell_type"].values
    annotations = cell_index["MultiomeAnnotated"].values.astype(str)

    P = raw.P_constraints_cells
    n_allowed = np.array(P.sum(axis=0)).flatten().astype(int)

    named_mask = tiers == "named"
    numeric_mask = tiers == "numeric"
    orphan_mask = tiers == "orphan"

    fig = go.Figure()

    for mask, name, color in [
        (named_mask, "Named (hard type)", "blue"),
        (numeric_mask, "Numeric (soft ambiguous)", "orange"),
        (orphan_mask, "Orphan (unknown)", "grey"),
    ]:
        if mask.sum() == 0:
            continue
        fig.add_trace(
            go.Scattergl(
                x=raw.umap_xy[mask, 0],
                y=raw.umap_xy[mask, 1],
                mode="markers",
                marker=dict(size=2, color=color, opacity=0.7),
                text=[
                    f"Annotation: {a}<br>Tier: {t}<br># allowed types: {n}"
                    for a, t, n in zip(
                        annotations[mask],
                        tiers[mask],
                        n_allowed[mask],
                    )
                ],
                hoverinfo="text",
                name=name,
            )
        )

    fig.update_layout(
        title="Cell Type Constraints (P support) on UMAP",
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        width=900,
        height=700,
        hovermode="closest",
    )
    fig.write_html(output_dir / "viz_02_cell_constraints.html", include_plotlyjs="cdn")


def _viz_metacells(
    raw: RawData,
    prep: PrepData,
    meta_dfs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    type_index = meta_dfs["type_index"]
    cell_to_meta = prep.cell_to_metacell
    meta_ids_unique = np.unique(cell_to_meta)
    n_meta = len(meta_ids_unique)

    meta_sizes = prep.meta_sizes.astype(int)
    P_constraints_meta = prep.P_constraints_metacell
    dominant_types = np.argmax(P_constraints_meta, axis=0)
    type_names = type_index["type"].values.astype(str)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Colored by Metacell ID", "Colored by Metacell Size"),
        horizontal_spacing=0.1,
    )

    discrete_colors = _discrete_colorscale(n_meta)
    meta_colors = [discrete_colors[m % len(discrete_colors)] for m in cell_to_meta]

    fig.add_trace(
        go.Scattergl(
            x=raw.umap_xy[:, 0],
            y=raw.umap_xy[:, 1],
            mode="markers",
            marker=dict(size=2, color=meta_colors, opacity=0.7),
            text=[
                f"Metacell: {m}<br>Size: {meta_sizes[m]}<br>Dominant type: {type_names[dominant_types[m]] if m < len(dominant_types) else 'N/A'}"
                for m in cell_to_meta
            ],
            hoverinfo="text",
            showlegend=False,
        ),
        row=1, col=1,
    )

    size_colors = [meta_sizes[m] for m in cell_to_meta]
    fig.add_trace(
        go.Scattergl(
            x=raw.umap_xy[:, 0],
            y=raw.umap_xy[:, 1],
            mode="markers",
            marker=dict(
                size=2,
                color=size_colors,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Metacell Size", x=1.02),
            ),
            text=[
                f"Metacell: {m}<br>Size: {meta_sizes[m]}"
                for m in cell_to_meta
            ],
            hoverinfo="text",
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=f"Metacells on UMAP ({n_meta} metacells from {len(cell_to_meta)} cells)",
        width=1200,
        height=550,
        hovermode="closest",
    )
    fig.write_html(output_dir / "viz_03_metacells.html", include_plotlyjs="cdn")


def _viz_inferred_types(
    raw: RawData,
    prep: PrepData,
    cm: CmResult,
    meta_dfs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    type_index = meta_dfs["type_index"]
    cell_index = meta_dfs["cell_index"]
    type_names = type_index["type"].values.astype(str)

    n_meta = cm.P.shape[1]
    inferred_type_idx = np.argmax(cm.P, axis=0)
    P_max = np.max(cm.P, axis=0)

    epsilon = 1e-30
    P_norm = cm.P / np.maximum(np.sum(cm.P, axis=0, keepdims=True), epsilon)
    P_entropy = -np.sum(P_norm * np.log2(np.maximum(P_norm, epsilon)), axis=0)

    cell_to_meta = prep.cell_to_metacell
    cell_inferred_type = np.array([inferred_type_idx[m] for m in cell_to_meta])
    cell_P_max = np.array([P_max[m] for m in cell_to_meta])
    cell_entropy = np.array([P_entropy[m] for m in cell_to_meta])

    annotations = cell_index["MultiomeAnnotated"].values.astype(str)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Inferred Type (argmax)", "Assignment Entropy (uncertainty)"),
        horizontal_spacing=0.1,
    )

    unique_types = np.unique(cell_inferred_type)
    type_to_color_idx = {t: i % 30 for i, t in enumerate(unique_types)}
    discrete_cmap = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
                      "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5","#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
                      "#393b79","#637939","#8c6d31","#843c39","#7b4173","#5254a3","#6b6ecf","#9c9ede","#ce6dbd","#8c6eb5"]
    cell_colors_type = [discrete_cmap[type_to_color_idx[t] % len(discrete_cmap)] for t in cell_inferred_type]

    fig.add_trace(
        go.Scattergl(
            x=raw.umap_xy[:, 0],
            y=raw.umap_xy[:, 1],
            mode="markers",
            marker=dict(size=2, color=cell_colors_type, opacity=0.7),
            text=[
                f"Annotation: {a}<br>Metacell: {m}<br>Inferred type: {type_names[t]}<br>Confidence: {p:.3f}"
                for a, m, t, p in zip(annotations, cell_to_meta, cell_inferred_type, cell_P_max)
            ],
            hoverinfo="text",
            showlegend=False,
        ),
        row=1, col=1,
    )

    max_entropy = np.log2(cm.P.shape[0]) if cm.P.shape[0] > 1 else 1.0
    normalized_entropy = cell_entropy / max_entropy

    fig.add_trace(
        go.Scattergl(
            x=raw.umap_xy[:, 0],
            y=raw.umap_xy[:, 1],
            mode="markers",
            marker=dict(
                size=2,
                color=normalized_entropy,
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="Norm. Entropy", x=1.02),
                cmin=0,
                cmax=1,
            ),
            text=[
                f"Annotation: {a}<br>Metacell: {m}<br>Entropy: {e:.3f}<br>Max confidence: {p:.3f}"
                for a, m, e, p in zip(annotations, cell_to_meta, cell_entropy, cell_P_max)
            ],
            hoverinfo="text",
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title="Inferred Type Assignments on UMAP",
        width=1200,
        height=550,
        hovermode="closest",
    )
    fig.write_html(output_dir / "viz_04_inferred_types.html", include_plotlyjs="cdn")


def _viz_connectome_fit(
    prep: PrepData,
    cm: CmResult,
    meta_dfs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    C = cm.C
    W = cm.C_mask
    C_hat = cm.C_recon

    idx = np.where(W > 0)
    c_obs = C[idx]
    c_pred = C_hat[idx]
    resid = c_obs - c_pred

    n_masked = c_obs.size
    corr = np.corrcoef(c_obs, c_pred)[0, 1] if n_masked > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((c_obs - np.mean(c_obs)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, np.finfo(float).eps)
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    n_types = C.shape[0]
    top_k = min(100, n_types)
    C_display = C.copy()
    C_display[W == 0] = np.nan
    C_hat_display = C_hat.copy()
    C_hat_display[W == 0] = np.nan

    row_var = np.nanvar(C_display, axis=1)
    col_var = np.nanvar(C_display, axis=0)
    top_rows = np.argsort(-row_var)[:top_k]
    top_cols = np.argsort(-col_var)[:top_k]

    finite = np.concatenate([
        C_display[np.isfinite(C_display)],
        C_hat_display[np.isfinite(C_hat_display)],
    ])
    vmax = float(np.max(finite)) if finite.size else 1.0

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f"True C (top-{top_k} types by variance)",
            f"Reconstructed C_hat (top-{top_k})",
            f"C_true vs C_recon (r={corr:.3f}, R²={r2:.3f}, RMSE={rmse:.4f})",
        ),
        horizontal_spacing=0.08,
        column_widths=[0.35, 0.35, 0.30],
    )

    fig.add_trace(
        go.Heatmap(
            z=C_display[np.ix_(top_rows, top_cols)],
            colorscale="Viridis",
            zmin=0,
            zmax=vmax,
            showscale=False,
            hovertemplate="Pre-type: %{x}<br>Post-type: %{y}<br>Value: %{z}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=C_hat_display[np.ix_(top_rows, top_cols)],
            colorscale="Viridis",
            zmin=0,
            zmax=vmax,
            showscale=False,
            hovertemplate="Pre-type: %{x}<br>Post-type: %{y}<br>Value: %{z}<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.add_trace(
        go.Scattergl(
            x=c_obs,
            y=c_pred,
            mode="markers",
            marker=dict(size=3, color="blue", opacity=0.3),
            text=[
                f"True: {o:.4f}<br>Pred: {p:.4f}"
                for o, p in zip(c_obs, c_pred)
            ],
            hoverinfo="text",
            showlegend=False,
        ),
        row=1, col=3,
    )

    max_val = float(np.max(np.concatenate([c_obs, c_pred])))
    fig.add_trace(
        go.Scattergl(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="red", dash="dash", width=1),
            showlegend=False,
        ),
        row=1, col=3,
    )

    fig.update_xaxes(title_text="True C", row=1, col=3)
    fig.update_yaxes(title_text="Predicted C_hat", row=1, col=3)

    fig.update_layout(
        title=f"Connectome Reconstruction: C vs C_hat (n={n_masked} masked entries)",
        width=1300,
        height=500,
    )
    fig.write_html(output_dir / "viz_05_connectome_fit.html", include_plotlyjs="cdn")


def _viz_loss_trajectory(
    cm: CmResult,
    output_dir: Path,
) -> None:
    n_iter = len(cm.loss)
    iterations = np.arange(1, n_iter + 1)

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=iterations, y=cm.obj_beta,
        mode="lines+markers",
        name="obj_beta",
        marker=dict(size=4),
        hovertemplate="Iter %{x}: obj_beta=%{y:.6e}<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=iterations, y=cm.obj_P_fit,
        mode="lines+markers",
        name="obj_P_fit",
        marker=dict(size=4),
        hovertemplate="Iter %{x}: obj_P_fit=%{y:.6e}<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=iterations, y=cm.obj_P_ent,
        mode="lines+markers",
        name="obj_P_ent",
        marker=dict(size=4),
        hovertemplate="Iter %{x}: obj_P_ent=%{y:.6e}<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=iterations, y=cm.loss,
        mode="lines+markers",
        name="total_loss",
        line=dict(width=3, color="black"),
        marker=dict(size=5, symbol="diamond"),
        hovertemplate="Iter %{x}: total_loss=%{y:.6e}<extra></extra>",
    ))

    fig.update_layout(
        title="Solver Loss Trajectory",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        width=900,
        height=600,
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98),
    )
    fig.update_yaxes(type="linear")

    fig.write_html(output_dir / "viz_06_loss_trajectory.html", include_plotlyjs="cdn")


def _discrete_colorscale(n: int) -> list[str]:
    base = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
            "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5","#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5"]
    repeats = (n // len(base)) + 1
    return (base * repeats)[:n]
