from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import CmResult, PrepData, RawData

D30 = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2",
    "#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78","#98df8a","#ff9896",
    "#c5b0d5","#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
    "#393b79","#637939","#8c6d31","#843c39","#7b4173","#5254a3",
    "#6b6ecf","#9c9ede","#ce6dbd","#8c6eb5",
]


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

    print("  Viz 7/7 — Combined three-panel overview ...")
    _viz_combined_three_panel(raw, prep, cm, meta_dfs, output_dir)

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
    cell_colors_type = [D30[type_to_color_idx[t] % len(D30)] for t in cell_inferred_type]

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
    repeats = (n // len(D30)) + 1
    return (D30 * repeats)[:n]


def _viz_combined_three_panel(
    raw: RawData,
    prep: PrepData,
    cm: CmResult,
    meta_dfs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    cell_index = meta_dfs["cell_index"]
    type_index = meta_dfs["type_index"]
    type_names = type_index["type"].values.astype(str)
    families = type_index["family"].values.astype(str)
    subsystems = type_index["subsystem"].values.astype(str)
    categories = type_index["category"].values.astype(str)

    barcodes = cell_index["cell_barcode"].values.astype(str)
    annotations = cell_index["MultiomeAnnotated"].values.astype(str)
    tiers = cell_index["cell_type"].values
    clusters_raw = cell_index["MultiomeNN"].values.astype(str)
    cell_to_meta = prep.cell_to_metacell

    # # allowed types per cell
    n_allowed = np.array(raw.P_constraints_cells.sum(axis=0)).flatten().astype(int)

    # --- Panel 1: Original Cell Type Annotations ---
    unique_ann = np.unique(annotations)
    ann_to_idx = {a: i for i, a in enumerate(unique_ann)}
    ann_colors = [
        D30[ann_to_idx[a] % len(D30)] if a not in ("nan", "") else "#cccccc"
        for a in annotations
    ]

    # --- Panel 2: Metacells ---
    meta_ids_unique = np.unique(cell_to_meta)
    n_meta = len(meta_ids_unique)
    dcolors = _discrete_colorscale(n_meta)
    meta_colors = [dcolors[m % len(dcolors)] for m in cell_to_meta]
    meta_sizes = prep.meta_sizes.astype(int)

    # --- Panel 3: Inferred Type ---
    inferred_type_idx = np.argmax(cm.P, axis=0)
    P_max = np.max(cm.P, axis=0)
    cell_inferred_type = np.array([inferred_type_idx[m] for m in cell_to_meta])
    cell_P_max = np.array([P_max[m] for m in cell_to_meta])
    cell_family = np.array([families[t] for t in cell_inferred_type])
    cell_subsystem = np.array([subsystems[t] for t in cell_inferred_type])
    cell_category = np.array([categories[t] for t in cell_inferred_type])

    unique_types = np.unique(cell_inferred_type)
    type_to_color_idx = {t: i % 30 for i, t in enumerate(unique_types)}
    inferred_colors = [D30[type_to_color_idx[t] % len(D30)] for t in cell_inferred_type]

    epsilon = 1e-30
    P_norm = cm.P / np.maximum(np.sum(cm.P, axis=0, keepdims=True), epsilon)
    P_entropy = -np.sum(P_norm * np.log2(np.maximum(P_norm, epsilon)), axis=0)
    cell_entropy = np.array([P_entropy[m] for m in cell_to_meta])
    max_entropy = np.log2(cm.P.shape[0]) if cm.P.shape[0] > 1 else 1.0
    cell_entropy_norm = cell_entropy / max_entropy

    umap_x = raw.umap_xy[:, 0].astype(str)
    umap_y = raw.umap_xy[:, 1].astype(str)

    customdata = np.column_stack([
        barcodes,
        annotations,
        tiers,
        cell_to_meta.astype(str),
        [type_names[t] for t in cell_inferred_type],
        cell_P_max.astype(str),
        cell_entropy_norm.astype(str),
        meta_sizes[cell_to_meta].astype(str),
        clusters_raw,
        n_allowed.astype(str),
        cell_family,
        cell_subsystem,
        cell_category,
        umap_x,
        umap_y,
    ])

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Original Cell Type Annotations",
            "Cells Colored by Metacell",
            "After ConnectionMiner (Inferred Type)",
        ),
        horizontal_spacing=0.05,
    )

    fig.add_trace(
        go.Scattergl(
            x=raw.umap_xy[:, 0], y=raw.umap_xy[:, 1],
            mode="markers",
            marker=dict(size=2, color=ann_colors, opacity=0.7),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Barcode: %{customdata[0]}<br>"
                "Tier: %{customdata[2]}<br>"
                "Metacell: %{customdata[3]}<br>"
                "<extra></extra>"
            ),
            showlegend=False, name="annotation",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=raw.umap_xy[:, 0], y=raw.umap_xy[:, 1],
            mode="markers",
            marker=dict(size=2, color=meta_colors, opacity=0.7),
            customdata=customdata,
            hovertemplate=(
                "<b>Metacell %{customdata[3]}</b><br>"
                "Annotation: %{customdata[1]}<br>"
                "Tier: %{customdata[2]}<br>"
                "Barcode: %{customdata[0]}<br>"
                "<extra></extra>"
            ),
            showlegend=False, name="metacell",
        ),
        row=1, col=2,
    )

    fig.add_trace(
        go.Scattergl(
            x=raw.umap_xy[:, 0], y=raw.umap_xy[:, 1],
            mode="markers",
            marker=dict(size=2, color=inferred_colors, opacity=0.7),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[4]}</b><br>"
                "Confidence: %{customdata[5]}<br>"
                "Annotation: %{customdata[1]}<br>"
                "Barcode: %{customdata[0]}<br>"
                "Tier: %{customdata[2]}<br>"
                "Metacell: %{customdata[3]}<br>"
                "Metacell Size: %{customdata[7]}<br>"
                "Norm. Entropy: %{customdata[6]}<br>"
                "<extra></extra>"
            ),
            showlegend=False, name="inferred",
        ),
        row=1, col=3,
    )

    for _col in [1, 2, 3]:
        fig.add_trace(
            go.Scattergl(
                x=[None], y=[None],
                mode="markers",
                marker=dict(
                    size=14,
                    color="rgba(255, 50, 50, 0.9)",
                    line=dict(width=2, color="white"),
                    symbol="circle-open",
                ),
                showlegend=False,
                hoverinfo="none",
                name="highlight",
            ),
            row=1, col=_col,
        )

    fig.update_layout(
        title="FlyWire Visual System x ConnectionMiner: 3-Stage Pipeline Overview",
        width=1500, height=500,
        hovermode="closest",
        margin=dict(r=320),
    )
    fig.update_xaxes(title_text="UMAP1", row=1, col=1)
    fig.update_yaxes(title_text="UMAP2", row=1, col=1)
    fig.update_xaxes(title_text="UMAP1", row=1, col=2)
    fig.update_yaxes(title_text="UMAP2", row=1, col=2)
    fig.update_xaxes(title_text="UMAP1", row=1, col=3)
    fig.update_yaxes(title_text="UMAP2", row=1, col=3)

    html_path = output_dir / "viz_combined_three_panel.html"
    fig.write_html(html_path, include_plotlyjs="cdn", div_id="three-panel")

    _inject_cross_highlight_js(html_path)


def _inject_cross_highlight_js(html_path: Path) -> None:
    css_block = """
    <style>
    #cell-info-panel {
        position: fixed; top: 60px; right: 10px; width: 280px;
        background: white; border: 1px solid #ddd; border-radius: 8px;
        padding: 0; font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 13px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000; max-height: 80vh; overflow-y: auto;
    }
    #cell-info-header {
        padding: 10px 15px; border-radius: 8px 8px 0 0; color: white;
        font-weight: 700; font-size: 13px; letter-spacing: 0.3px;
    }
    #cell-info-body { padding: 10px 15px; }
    #cell-info-body h3 {
        margin: 0 0 8px 0; font-size: 14px;
        border-bottom: 1px solid #eee; padding-bottom: 4px; color: #333;
    }
    .info-row { margin: 3px 0; display: flex; }
    .info-label { font-weight: 600; width: 100px; color: #555; flex-shrink: 0; }
    .info-value { flex: 1; color: #222; word-break: break-all; }
    .no-selection { color: #999; font-style: italic; text-align: center; padding: 20px 0; }
    .key-info { margin: 6px 0 10px 0; padding: 8px; background: #f7f7f7; border-radius: 4px; font-size: 13px; }
    .key-info .key-label { font-weight: 600; color: #333; }
    .key-info .key-value { color: #1f77b4; font-weight: 700; }
    </style>
    """

    info_panel_html = """
    <div id="cell-info-panel">
        <div id="cell-info-header">Cell Information</div>
        <div id="cell-info-body">
            <div id="cell-info-content">
                <div class="no-selection">Hover over a cell to see details</div>
            </div>
        </div>
    </div>
    """

    js_block = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var plotDiv = document.getElementById('three-panel');
        if (!plotDiv) return;

        var hlIdx = function() {
            var n = plotDiv.data ? plotDiv.data.length : 0;
            return [n - 3, n - 2, n - 1];
        };
        var mainIdx = [0, 1, 2];
        var _syncing = false;

        var panelMeta = [
            { name: 'Original Annotations', color: '#2ca02c', keyLabel: 'Annotation', keyIdx: 1 },
            { name: 'Metacells', color: '#9467bd', keyLabel: 'Metacell', keyIdx: 3 },
            { name: 'Inferred Type', color: '#d62728', keyLabel: 'Inferred Type', keyIdx: 4 },
        ];

        var CD = {
            BARCODE: 0, ANNOTATION: 1, TIER: 2, META_ID: 3,
            INF_TYPE: 4, CONFIDENCE: 5, ENTROPY: 6, META_SIZE: 7,
            CLUSTER: 8, N_ALLOWED: 9, FAMILY: 10, SUBSYSTEM: 11,
            CATEGORY: 12, UMAP_X: 13, UMAP_Y: 14,
        };

        function renderInfo(cd, cn) {
            var pm = panelMeta[cn];
            document.getElementById('cell-info-header').style.background = pm.color;
            document.getElementById('cell-info-header').textContent = pm.name;

            var keyVal = cd[pm.keyIdx] || 'N/A';
            var keyExtra = '';
            if (cn === 1) keyExtra = ' (Size: ' + cd[CD.META_SIZE] + ' cells)';
            if (cn === 2) keyExtra = ' (confidence: ' + cd[CD.CONFIDENCE] + ', entropy: ' + cd[CD.ENTROPY] + ')';

            var sections = [
                { title: 'Identity', rows: [
                    ['Barcode', CD.BARCODE],
                    ['MultiomeNN Cluster', CD.CLUSTER],
                    ['Constraint Tier', CD.TIER],
                    ['Allowed Types', CD.N_ALLOWED],
                ]},
                { title: 'Type Annotation', rows: [
                    ['Annotated Type', CD.ANNOTATION],
                    ['Family', CD.FAMILY],
                    ['Subsystem', CD.SUBSYSTEM],
                    ['Category', CD.CATEGORY],
                ]},
                { title: 'Metacell', rows: [
                    ['Metacell ID', CD.META_ID],
                    ['Size (cells)', CD.META_SIZE],
                ]},
                { title: 'ConnectionMiner Result', rows: [
                    ['Inferred Type', CD.INF_TYPE],
                    ['Confidence', CD.CONFIDENCE],
                    ['Norm. Entropy', CD.ENTROPY],
                ]},
                { title: 'UMAP Position', rows: [
                    ['UMAP1', CD.UMAP_X],
                    ['UMAP2', CD.UMAP_Y],
                ]},
            ];

            var html = '<div class="key-info"><span class="key-label">' + pm.keyLabel + ': </span><span class="key-value">' + keyVal + '</span>' + keyExtra + '</div>';
            for (var s = 0; s < sections.length; s++) {
                html += '<h3>' + sections[s].title + '</h3>';
                for (var r = 0; r < sections[s].rows.length; r++) {
                    var row = sections[s].rows[r];
                    html += '<div class="info-row"><span class="info-label">' + row[0] + ':</span><span class="info-value">' + (cd[row[1]] || '') + '</span></div>';
                }
            }
            document.getElementById('cell-info-content').innerHTML = html;
        }

        plotDiv.on('plotly_hover', function(data) {
            if (!data.points || data.points.length === 0) return;
            var pt = data.points[0];
            var idx = pt.pointIndex;
            var hl = hlIdx();

            for (var c = 0; c < 3; c++) {
                var x = plotDiv.data[mainIdx[c]].x[idx];
                var y = plotDiv.data[mainIdx[c]].y[idx];
                Plotly.restyle(plotDiv, { 'x': [[x]], 'y': [[y]] }, hl[c]);
            }

            var cd = pt.customdata;
            if (!cd) return;

            var cn = pt.curveNumber;
            if (cn < 0 || cn > 2) cn = 0;
            renderInfo(cd, cn);
        });

        plotDiv.on('plotly_unhover', function() {
            var hl = hlIdx();
            for (var c = 0; c < 3; c++) {
                Plotly.restyle(plotDiv, { 'x': [[null]], 'y': [[null]] }, hl[c]);
            }
            document.getElementById('cell-info-header').style.background = '#1f77b4';
            document.getElementById('cell-info-header').textContent = 'Cell Information';
            document.getElementById('cell-info-content').innerHTML =
                '<div class="no-selection">Hover over a cell to see details</div>';
        });

        function getRange(ed, key) {
            if (ed[key] && Array.isArray(ed[key].range) && ed[key].range.length === 2)
                return ed[key].range;
            var r0 = ed[key + '.range[0]'], r1 = ed[key + '.range[1]'];
            if (r0 !== undefined && r1 !== undefined) return [r0, r1];
            return null;
        }

        var allAxes = ['xaxis','xaxis2','xaxis3','yaxis','yaxis2','yaxis3'];

        plotDiv.on('plotly_relayout', function(ed) {
            if (_syncing) return;
            if (!ed || Object.keys(ed).length === 0) return;

            var update = {};
            for (var a = 0; a < allAxes.length; a++) {
                var key = allAxes[a];
                var range = getRange(ed, key);
                if (!range) continue;
                var type = key.replace(/[0-9]/g, '');
                for (var b = 0; b < allAxes.length; b++) {
                    var target = allAxes[b];
                    if (target.startsWith(type) && target !== key) {
                        update[target + '.range[0]'] = range[0];
                        update[target + '.range[1]'] = range[1];
                    }
                }
            }

            if (Object.keys(update).length > 0) {
                _syncing = true;
                Plotly.relayout(plotDiv, update);
                setTimeout(function() { _syncing = false; }, 200);
            }
        });
    });
    </script>
    """

    with open(html_path, "r") as f:
        html = f.read()

    html = html.replace("</body>", css_block + "\n" + info_panel_html + "\n" + js_block + "\n</body>")

    with open(html_path, "w") as f:
        f.write(html)
