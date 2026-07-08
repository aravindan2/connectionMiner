# Ablation Experiments: XLSX Gene Lists vs HVG Baselines

## Objective

Run ConnectionMiner using gene sets from curated xlsx files (TFs, cell adhesion molecules, interactome pairs) instead of the default highly-variable gene (HVG) selection, and compare solver behavior and connectome reconstruction quality across all conditions.

---

## Gene Sets (from `Gene_list/`)

| # | Experiment Name | Source | Column(s) | Total Genes | In h5ad (12028) |
|---|----------------|--------|-----------|-------------|-----------------|
| 1 | `hvg_3000` | Baseline: top 3000 HVGs by variance | — | 3000 | 3000 |
| 2 | `hvg_5000` | Baseline: top 5000 HVGs by variance | — | 5000 | 5000 |
| 3 | `tfs_only` | TFs_groups.xlsx | `gene` | 628 | 521 |
| 4 | `adhesion_only` | cell adhesion molecules_new.xlsx | `Gene` | 322 | 278 |
| 5 | `interactome_only` | Interactome_v3.xlsx | `Partner 1` ∪ `Partner 2` | 119 | 87 |
| 6 | `tfs_adhesion` | Union of 3 + 4 | — | ~800 | ~700 |
| 7 | `tfs_interactome` | Union of 3 + 5 | — | ~700 | ~580 |
| 8 | `adhesion_interactome` | Union of 4 + 5 | — | ~400 | ~350 |
| 9 | `all_three_union` | Union of 3 + 4 + 5 | — | ~900 | ~800 |
| 10 | `all_three_hvg3000` | Intersection: union ∩ 3000 HVGs | — | — | ~250 |

---

## Solver Settings

| Parameter | Value |
|-----------|-------|
| `num_iter` | 20 |
| `lambda_sparsity` | 0.001 |
| `beta_rank` | 0 (full rank) |
| `use_complement` | False |
| `use_binary_connectome` | True |
| `P_init` | `random_proportional` |
| `beta_init` | `random` |
| `seed` | 750 |

---

## Output Structure

```
output/connectionMiner_ablation/
├── exp_01_hvg_3000/
│   ├── beta_learned.npy
│   ├── P_refined.npz
│   ├── C_reconstructed.npy
│   ├── cell_to_metacell_solver.npy
│   ├── solver_loss.csv
│   ├── type_gene_probabilities.xlsx
│   ├── run_config.json          # experiment metadata + params
│   ├── run_stats.json           # metrics: final loss, n_meta, timing, etc.
│   ├── viz_01_raw_clusters.html
│   ├── viz_02_cell_constraints.html
│   ├── viz_03_metacells.html
│   ├── viz_04_inferred_types.html
│   ├── viz_05_connectome_fit.html
│   ├── viz_06_loss_trajectory.html
│   └── viz_combined_three_panel.html
├── exp_02_hvg_5000/
│   └── ...
├── exp_03_tfs_only/
│   └── ...
├── ...
└── viz_ablation_comparison.html   # consolidated cross-experiment dashboard
```

---

## Implementation Plan

### 1. Modify `cm_visual/preprocess.py` — `cm_preprocess_binary()`

Add a `custom_gene_idx` parameter:
- When `None`: use current HVG selection (no change)
- When provided (np.ndarray of column indices): skip HVG variance filtering, use the given indices as `solver_gene_idx`

The key change is in lines 20–32. Instead of:

```python
n_genes_use = min(cfg.get("binary", {}).get("n_genes_use", 4000), ng_shared)
min_cells = ...
nz = np.sum(X_bin, axis=0)
cand = np.where(nz >= min_cells)[0]
mu = np.mean(X_bin[:, cand], axis=0)
var = mu * (1.0 - mu)
ord_idx = np.argsort(-var)
ng_use = min(n_genes_use, cand.size)
solver_gene_idx = cand[ord_idx[:ng_use]]
```

We add a branch:

```python
custom_gene_idx = cfg.get("binary", {}).get("custom_gene_idx", None)
if custom_gene_idx is not None:
    solver_gene_idx = np.asarray(custom_gene_idx, dtype=int)
    # Validate bounds
    solver_gene_idx = solver_gene_idx[solver_gene_idx < X_bin.shape[1]]
else:
    # ... existing HVG selection ...
```

### 2. Modify `cm_visual/run_visual.py` — `cm_run_visual()`

- Accept an `ablation_root` parameter to redirect all outputs to `output/connectionMiner_ablation/exp_NN_name/`
- Save `run_config.json` capturing: experiment name, gene count, solver params, elapsed time
- Save `run_stats.json` capturing: final loss, pearson r of C_recon vs C, metacell count, beta sparsity, type entropy

### 3. New: `run_ablation.py` — experiment orchestrator

```python
def run_ablation():
    experiments = [
        {"name": "hvg_3000", "n_hvg": 3000},
        {"name": "hvg_5000", "n_hvg": 5000},
        {"name": "tfs_only", "xlsx_path": "Gene_list/TFs_groups.xlsx", "xlsx_col": "gene"},
        {"name": "adhesion_only", "xlsx_path": "Gene_list/cell adhesion molecules_new.xlsx", "xlsx_col": "Gene"},
        {"name": "interactome_only", "xlsx_path": "Gene_list/Interactome_v3.xlsx", "xlsx_cols": ["Partner 1", "Partner 2"]},
        # ... combinations derived from union of the above ...
    ]
    for exp in experiments:
        run_single(exp)
    consolidate_results()
```

Key responsibilities:
1. Load h5ad var_names
2. For each experiment, determine gene indices in h5ad
3. Create config override (`custom_gene_idx`, `binary.n_genes_use`, `solver.num_iter=20`)
4. Call `cm_run_visual(ablation_root="output/connectionMiner_ablation", subdir=f"exp_{i:02d}_{name}")`
5. Run sequentially (one experiment at a time)

### 4. New: `cm_visual/viz_ablation.py` — consolidated dashboard

Single HTML page `viz_ablation_comparison.html` with:

**Tab 1 — Loss Trajectories**
- Overlay plot: total_loss vs iteration for all 10 experiments
- Color-coded by experiment
- Log/linear scale toggle

**Tab 2 — Connectome Fit Metrics**
- Bar chart: Pearson r of C_recon vs C_true per experiment
- Bar chart: Relative Frobenius error per experiment

**Tab 3 — Experiment Stats Table**
- Experiment name, gene count, metacell count, final loss, pearson r, timing, beta sparsity, P entropy

**Tab 4 — Beta Matrix Sparsity**
- Bar chart: fraction of beta entries == 0 per experiment

---

## Current Gene Coverage in h5ad

| Source | Total | In h5ad | Coverage |
|--------|-------|---------|----------|
| TFs_groups.xlsx | 628 | 521 | 83% |
| cell adhesion molecules_new.xlsx | 322 | 278 | 86% |
| Interactome_v3.xlsx | 119 | 87 | 73% |

Genes not found in h5ad are dropped with a warning.

---

## Comparison Metrics

| Metric | Description | How |
|--------|-------------|-----|
| Final total loss | `obj_P_fit + obj_P_ent` at iter 20 | from `solver_loss.csv` |
| Connectome Pearson r | Correlation between `C` and `C_recon` | `np.corrcoef(C.ravel(), C_recon.ravel())[0,1]` |
| Frobenius error | `||W * (C - C_recon)||_F / ||W * C||_F` | from solver |
| Metacell count | `n_meta` from preprocess | from `prep.meta["N_metacells"]` |
| Beta sparsity | Fraction of `beta < 1e-10` | from `beta_learned.npy` |
| P entropy | Mean entropy of P rows | from `P_refined.npz` |
| Runtime | Wall-clock seconds | from solver |
