# ConnectionMiner — FlyWire Visual System

**Connectome-constrained deconvolution of visual neuron types from FlyWire single-cell RNA-seq**

ConnectionMiner jointly infers neuronal type identities and synaptic gene interaction programs from single-cell transcriptomics data, using the measured synaptic connectome as a structural constraint. This fork adapts the original pipeline (motor system, ~730 types) for the **FlyWire visual system** (741 visual neuron types, ~110k cells).

---

## Overview

A core challenge in single-cell genomics of the nervous system is that many cells in a dataset are replicates of the same neuron type — but type identity is not directly observed. ConnectionMiner addresses this by formulating type assignment and gene interaction inference as a joint optimization problem:

- **P** — a soft assignment matrix mapping cells (via metacells) to neuron types, inferred via entropic optimal transport
- **β** — a gene–gene interaction weight matrix encoding which gene pairs predict synaptic connectivity, inferred via multiplicative non-negative regression

Both are optimized alternately, each step using the other as a fixed constraint, until convergence. The connectome (which neuron types are synaptically connected) serves as the supervisory signal throughout.

### Pipeline

```
Raw scRNA-seq              Connectome
(109k visual cells)        (741 × 741)
        │                       │
        ▼                       ▼
  ┌────────────────────────────────────┐
  │  1. Build RawData                  │
  │     load h5ad · P_constraints     │
  └──────────────┬─────────────────────┘
                 │
                 ▼
  ┌────────────────────────────────────┐
  │  2. Preprocess (binary)            │
  │     binarise · select 3000 HVGs   │
  │     build metacells (~6k)         │
  └──────────────┬─────────────────────┘
                 │
                 ▼
  ┌────────────────────────────────────┐
  │  3. Solve (alternating)            │
  │     P ← entropic OT (Sinkhorn)    │
  │     β ← multiplicative regression │
  └──────────────┬─────────────────────┘
                 │
                 ▼
  ┌────────────────────────────────────┐
  │  4. Postprocess + Export           │
  │     type × gene probabilities      │
  │     identifiability assessment     │
  └──────────────┬─────────────────────┘
                 │
                 ▼
  ┌────────────────────────────────────┐
  │  5. Interactive Visualizations     │
  │     Plotly HTML (WebGL)            │
  └────────────────────────────────────┘
```

---

## Repository Structure

```
connectionMiner/
├── cm_visual/                  # Python pipeline package (visual system)
│   ├── run_visual.py           # Core pipeline entrypoint
│   ├── config.py               # Default configuration (741 types, 3000 HVGs)
│   ├── paths.py                # Data path resolution
│   ├── models.py               # Data containers (RawData, PrepData, CmResult)
│   ├── preprocess.py           # HVG selection + metacell construction
│   ├── solver.py               # Alternating P / β optimisation
│   ├── postprocess.py          # Type–gene probabilities + identifiability
│   ├── exports.py              # Excel export (type-gene table)
│   ├── viz_plotly.py           # 6 interactive Plotly visualizations
│   ├── validate.py             # Shape/value invariant checks
│   ├── utils.py                # Shared helpers
│   └── runs/                   # Output directory (timestamped, gitignored)
├── run_flywire.py              # CLI entrypoint
├── output/                     # Pre-built matrices (symlink, gitignored)
│   ├── C_matrix.npz            # 741×741 binary visual connectome
│   ├── P_constraints_cells.npz # 741×109k cell type constraints
│   ├── cell_index.csv          # Cell metadata (barcode, tier, annotation)
│   ├── type_index.csv          # 741 type names
│   ├── gene_index.csv          # 3000 HVG names
│   └── Adult.h5ad              # Raw expression + t-SNE coords (109k cells)
├── requirements.txt
└── README.md
```

---

## Installation

**Python ≥ 3.9** required.

```bash
git clone https://github.com/aravindan2/connectionMiner.git
cd connectionMiner
pip install -r requirements.txt
pip install plotly kaleido anndata
```

---

## Data

Pre-built matrices are expected in `./output/`. These include the binary visual connectome (741×741), cell-level type constraints, gene and cell indices, and `Adult.h5ad` with raw expression and t-SNE coordinates.

---

## Running

All commands are run from the **repo root**.

### Quick smoke test (2 iterations)

```bash
python3 run_flywire.py --num-iter 2 --smoke
```

### Full run (100 iterations)

```bash
python3 run_flywire.py --num-iter 100 --lambda-sparsity 0.001
```

### Low-rank beta (faster, less expressive)

```bash
python3 run_flywire.py --num-iter 100 --beta-rank 50
```

---

## Outputs

Solver outputs and visualizations are written to `output/connectionMiner_solve/`:

| File | Description |
|---|---|
| `beta_learned.npy` | 3000×3000 fitted gene interaction matrix |
| `P_refined.npz` | 741×n_meta solver-refined type assignments |
| `C_reconstructed.npy` | 741×741 reconstructed connectome |
| `solver_loss.csv` | Per-iteration loss breakdown |
| `cell_to_metacell_solver.npy` | Solver's internal metacell mapping |
| `type_gene_probabilities.xlsx` | Inferred gene expression per type |
| `viz_01_raw_clusters.html` | Raw MultiomeNN clusters on t-SNE |
| `viz_02_cell_constraints.html` | Cell type constraints (named/numeric/orphan) |
| `viz_03_metacells.html` | Metacells colored by ID and size |
| `viz_04_inferred_types.html` | Inferred type assignments + entropy |
| `viz_05_connectome_fit.html` | C vs C_hat heatmaps + scatter |
| `viz_06_loss_trajectory.html` | Solver convergence curves |
| `viz_combined_three_panel.html` | Combined 3-panel pipeline overview |

---

## Configuration

Default parameters are in `cm_visual/config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `seed` | `750` | Global random seed |
| `binary.n_genes_use` | `3000` | Number of highly variable genes for solver |
| `metacell.target_size` | `15` | Target cells per metacell |
| `solver.num_iter` | `100` | Alternating optimisation iterations |
| `solver.lambda_sparsity` | `0.001` | L1 sparsity penalty on β |
| `solver.optimal_transport_epsilon` | `1e-12` | Entropic OT regularisation strength |
| `solver.use_complement` | `False` | Disable complement features (avoid 6000×6000 β) |
| `solver.use_binary_connectome` | `True` | Binarise C before fitting |

Override any parameter via CLI flags:

```bash
python3 run_flywire.py --num-iter 200 --lambda-sparsity 0.01 --beta-rank 50
```

---

## Interactive Visualizations

All plots use `Plotly Scattergl` (WebGL) for smooth rendering of 100k+ points and are saved as standalone HTML files.

1. **Raw Clusters** — MultiomeNN clustering before ConnectionMiner processing
2. **Cell Constraints** — Three constraint tiers: named (hard), numeric (ambiguous), orphan (unknown)
3. **Metacells** — Metacell clusters colored by ID and size
4. **Inferred Types** — Solver's type assignments with confidence entropy
5. **Connectome Fit** — True vs reconstructed connectome with fit metrics
6. **Loss Trajectory** — Solver convergence across iterations

A combined 3-panel view (`viz_combined_three_panel.html`) shows the full pipeline flow: original clusters → metacells → inferred types.

---

## Method Summary

**Metacell construction.** Cells are first grouped by their *constraint signature* — the unique pattern of neuron types they could plausibly belong to. Within each group, PCA + K-means produces compact metacells (target size 15 cells). This reduces compute while preserving biological structure.

**Joint optimisation.** The solver alternates between:

1. **β update** — given the current type assignment P, find gene–gene interaction weights β that minimise weighted connectome reconstruction error plus an L1 sparsity term. Solved via multiplicative updates that preserve non-negativity.

2. **P update** — given the current β, find soft type assignments P that minimise connectome reconstruction error plus an entropic regularisation term (encouraging uncertainty over plausible types). Solved via a Sinkhorn-style two-pass update with backtracking line search.

---

## Citation

If you use ConnectionMiner in your research, please cite the original paper:

> Gupta, H.P., Azevedo, A., Chen, Y.-C.D., Xing, K., Sims, P.A., Varol, E., & Mann, R.S. (2025). **Decoding neuronal wiring by joint inference of cell identity and synaptic connectivity.** *bioRxiv*. https://doi.org/10.1101/2025.03.04.640006

---

## License

*To be added.*
