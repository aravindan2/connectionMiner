# ConnectionMiner

**Connectome-constrained deconvolution of neuronal cell types from single-cell RNA-seq**

ConnectionMiner jointly infers neuronal type identities and synaptic gene interaction programs from single-cell transcriptomics data, using the measured synaptic connectome as a structural constraint. It is applied here to the motor system of *Drosophila melanogaster*, resolving ~730 premotor and motor neuron types from thousands of single-cell measurements.

---

## Overview

A core challenge in single-cell genomics of the nervous system is that many cells in a dataset are replicates of the same neuron type — but type identity is not directly observed. ConnectionMiner addresses this by formulating type assignment and gene interaction inference as a joint optimization problem:

- **P** — a soft assignment matrix mapping cells (via metacells) to neuron types, inferred via entropic optimal transport
- **β** — a gene–gene interaction weight matrix encoding which gene pairs predict synaptic connectivity, inferred via multiplicative non-negative regression

Both are optimized alternately, each step using the other as a fixed constraint, until convergence. The connectome (which neuron types are synaptically connected) serves as the supervisory signal throughout.

### Pipeline

```
Raw scRNA-seq                 Connectome
(preMN + MN cells)            (type × type)
        │                          │
        ▼                          ▼
  ┌─────────────────────────────────────────┐
  │  1. Load                                │
  │     align genes · build constraints     │
  └─────────────────────┬───────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────┐
  │  2. Preprocess                          │
  │     select HVGs · build metacells       │
  │     (PCA + K-means per constraint sig)  │
  └─────────────────────┬───────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────┐
  │  3. Solve  (alternating optimisation)   │
  │     P  ← entropic OT (Sinkhorn)         │
  │     β  ← multiplicative regression      │
  └─────────────────────┬───────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────┐
  │  4. Postprocess                         │
  │     type × gene probabilities           │
  │     identifiability assessment          │
  └─────────────────────┬───────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────┐
  │  5. Export + Visualize                  │
  │     Excel tables · diagnostic figures   │
  └─────────────────────────────────────────┘
```

---

## Example Outputs

### Cell embedding and type assignment

UMAP of preMN cells coloured by (top-left) raw sequencing cluster, (top-right) constraint set membership, (bottom-left) metacell assignment, and (bottom-right) the top inferred type from the solver across 730 types.

![UMAP panels](docs/images/umap_four_panels.png)

---

### Connectome structure (input)

The 730 × 730 observed connectome **C** (left) and its solver-predicted reconstruction **Ĉ** (right). Rows/columns are neuron types; yellow = connected.

![Connectome matrices](docs/images/connectome_matrices.png)

---

### Connectome fit (output)

Scatter of observed vs. predicted connectivity values across ~488 k measured type pairs, with residual distribution. Each point is a type-pair entry in **C**.

![Connectome fit](docs/images/connectome_fit.png)

---

### Solver convergence

Total objective loss over 100 alternating iterations of P and β updates.

![Solver loss](docs/images/solver_loss.png)

---

### Type identifiability

Cell contribution mass per inferred type. Green = identifiable (mass > 0); red = not identifiable. In a full run, 715 of 730 types are resolved.

![Identifiability](docs/images/identifiability.png)

---

### Inferred type × gene expression

Heatmap of inferred gene expression probabilities for all identifiable types across the top highly variable genes. Each row is a neuron type; each column is a gene.

![Type heatmap](docs/images/type_heatmap_identifiable.png)

---

## Repository Structure

```
connectionMiner/
├── cm_minimal/               # Python pipeline package
│   ├── run.py                # Entry point (CLI)
│   ├── config.py             # Default configuration
│   ├── paths.py              # Data path resolution
│   ├── models.py             # Data containers (RawData, PrepData, CmResult)
│   ├── loaders.py            # Load expression + connectome, align cells
│   ├── preprocess.py         # HVG selection + metacell construction
│   ├── solver.py             # Alternating P / β optimisation
│   ├── postprocess.py        # Type–gene probabilities + identifiability
│   ├── exports.py            # Excel export (type-gene table, synapse table)
│   ├── viz.py                # Diagnostic visualisations
│   ├── validate.py           # Shape/value invariant checks
│   ├── utils.py              # Shared helpers
│   ├── runs/                 # Output directory (timestamped, gitignored)
│   └── experiments/          # Exploratory scripts
├── matlab/                   # Archived MATLAB implementation
├── docs/
│   └── images/               # Example figures for this README
├── requirements.txt
└── README.md
```

---

## Installation

**Python ≥ 3.9** required.

```bash
git clone https://github.com/<your-org>/connectionMiner.git
cd connectionMiner
pip install -r requirements.txt
```

No additional R or MATLAB dependencies are needed for the Python pipeline.

---

## Data

Input data is **not included** in this repository (files are large, ~1.1 GB total).

### Download

All required files are bundled in a single Dropbox folder, arranged in the exact layout the pipeline expects:

**➜ [Download dataset (Dropbox, ~1.1 GB)](https://www.dropbox.com/scl/fo/pmbekzqpx120m3pz6br0w/AJzZCEqRSMpLW0omJGSQTf4?rlkey=5hoi38x3jidkbb96audp5jqdc&dl=0)**

The contents of this folder *are* a ready-to-use `data_root` — download it and point the pipeline at it (see below); no internal path edits are needed.

### Layout

```
data_root/
├── scRNAseq PreMNs/
│   ├── counts_cg_corrected.txt                 # preMN expression matrix, genes × cells (407 MB)
│   ├── Cell_Cluster.xlsx                        # preMN cluster → type mapping (580 KB)
│   ├── umapCoord_vnc.csv                        # preMN UMAP coordinates (1.7 MB)
│   └── PreMNs-MNs connection_20250107.xlsx      # connectome + lineage / motor pool (1.8 MB)
├── Matrix and umap raw files/
│   ├── MNs_detailed_info_matrix format_with_developmental_age.xlsx   # MN covariates + connectome columns (148 KB)
│   └── Merged/
│       ├── matched_gene_expression_cg_corrected.txt        # MN expression matrix (698 MB)
│       ├── matched_clusters.xlsx                            # MN clusters; drives timepoint filter (112 KB)
│       └── matched_umap_coordinates_time_specific.xlsx      # MN UMAP coordinates (308 KB)
└── Genes list/
    └── Interactome_v3.xlsx                      # ligand–receptor pairs; synaptic export only (16 KB)
```

The first eight files are required for a full run. `Genes list/Interactome_v3.xlsx` is needed only for the `synaptic_interaction_table.xlsx` export.

By default, `data_root` is resolved from `cm_minimal/paths.py`, which points at a hard-coded local Dropbox path. To use the downloaded folder, edit the `data_root` value in `paths.py` or pass overrides via `merge_config()`.

---

## Running

All commands are run from the **repo root**.

### Full run (binary expression mode)

```bash
python3 -m cm_minimal.run --mode binary
```

### Full run (Poisson-Gamma continuous mode)

```bash
python3 -m cm_minimal.run --mode pg --pg-run-dir <path-to-pg-output>
```

### Quick smoke test (subsampled, ~1 min)

```bash
python3 -m cm_minimal.run --mode binary --smoke --max-cells 500 --max-genes 500 --num-iter 2
```

### Unit tests

```bash
python3 -m cm_minimal.test_binary_smoke
python3 -m cm_minimal.test_export_one_iter
```

---

## Outputs

Each run writes to `cm_minimal/runs/run_YYYYMMDD_HHMMSS/`:

| File | Description |
|---|---|
| `prep.mat` / `prep_pg.mat` | Preprocessed metacell structures |
| `cm.mat` / `cm_pg.mat` | Solver result: P, β, objectives, reconstructed C |
| `run_manifest.json` | Run configuration and summary statistics |
| `solver_objectives.txt` | Loss per iteration (tab-separated) |
| `type_gene_probabilities.xlsx` | Inferred gene expression per type (all genes) |
| `synaptic_interaction_table.xlsx` | Gene pairs scored per synapse with effect sizes |
| `viz/` | Diagnostic PNG figures (see examples above) |

### Output 1 — `type_gene_probabilities.xlsx`

One row per neuron type (~730 rows). This is the primary deconvolution output: for each inferred type it reports how many cells were assigned to it, how confidently, and what gene expression profile was inferred.

| Column | Description |
|---|---|
| `type_name` | Neuron type identifier |
| `n_cells` | Number of cells whose maximum-probability assignment is this type |
| `cell_contributions` | **Weighted** cell mass assigned to this type across all cells (includes partial/soft assignments, not just max-probability) |
| `identifiable` | `1` if the type received non-zero probability mass from the solver; `0` if it could not be resolved (e.g. too few cells, ambiguous constraints) — expression values are set to `NaN` for non-identifiable types |
| `{gene}_prob` | Inferred expression probability for each gene (one column per gene, across all shared genes) |

> **`n_cells` vs `cell_contributions`**: `n_cells` counts hard assignments (each cell counted once, to its most likely type). `cell_contributions` is the soft total — summing the solver's probability weight for this type across all cells. A type with a large `cell_contributions` but small `n_cells` indicates cells that are ambiguously spread across multiple types.

---

### Output 2 — `synaptic_interaction_table.xlsx`

One row per **(synapse × gene pair)** combination. This table scores every candidate ligand–receptor interaction at every measured synaptic connection, reporting whether that gene pair is specifically enriched at that synapse compared to non-synaptically connected type pairs.

| Column | Description |
|---|---|
| `SynapseName` | `preType → postType` synapse identifier |
| `preSynapseName` / `postSynapseName` | Pre- and post-synaptic type names |
| `preSynapseType` / `postSynapseType` | `"preMN"` or `"MN"` |
| `preSynapseLineage` / `postSynapseLineage` | Developmental lineage of each partner |
| `preSynapseMotorPool` / `postSynapseMotorPool` | Motor pool of each partner |
| `interactionName` | `geneA → geneB` (directed ligand–receptor pair from interactome) |
| `preInteractionName` / `postInteractionName` | Individual pre- and post-synaptic gene names |
| `synapseStrength` | Raw synapse count from the connectome |
| `geneCoExp` | Co-expression score: pre-gene probability × post-gene probability |
| `preGeneExp` / `postGeneExp` | Individual gene expression probabilities at pre/post type |
| `effectSize` | Mean difference in gene-pair co-expression product between synaptically connected and non-synaptically connected type pairs (t-test); positive = enriched at this synapse |
| `preEffectSize` / `postEffectSize` | Pre- and post-synaptic contributions to the effect size separately |
| `interactionScore` | `geneCoExp × effectSize` — combined ranking score |
| `pValue` | Combined significance (Fisher normal method across pre + post t-tests) |
| `prePvalue` / `postPvalue` | Individual p-values for pre- and post-synaptic sides |

The table is pruned to the **top 3 highest**, **top 3 lowest**, and **top 3 lowest p-value** entries per synapse to keep the file within Excel's row limit. The full unpruned table can exceed 1 M rows across all type pairs and gene combinations.

---

## Configuration

Default parameters are in `cm_minimal/config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `seed` | `750` | Global random seed |
| `binary.n_genes_use` | `4000` | Number of highly variable genes for solver |
| `metacell.target_size` | `10` | Target cells per metacell |
| `solver.num_iter` | `100` | Alternating optimisation iterations |
| `solver.lambda_sparsity` | `0.001` | L1 sparsity penalty on β |
| `solver.optimal_transport_epsilon` | `1e-12` | Entropic OT regularisation strength |
| `solver.use_binary_connectome` | `True` | Binarise C before fitting |

Override any parameter by passing a nested dict to `merge_config()` in your own script:

```python
from cm_minimal.config import default_config, merge_config

cfg = merge_config(default_config("binary"), {
    "solver": {"num_iter": 200, "lambda_sparsity": 0.01},
})
```

---

## Method Summary

**Metacell construction.** Cells are first grouped by their *constraint signature* — the unique pattern of neuron types they could plausibly belong to. Within each group, PCA + K-means produces compact metacells (target size 10 cells). This reduces compute while preserving biological structure.

**Joint optimisation.** The solver alternates between:

1. **β update** — given the current type assignment P, find gene–gene interaction weights β that minimise weighted connectome reconstruction error plus an L1 sparsity term. Solved via multiplicative updates that preserve non-negativity.

2. **P update** — given the current β, find soft type assignments P that minimise connectome reconstruction error plus an entropic regularisation term (encouraging uncertainty over plausible types). Solved via a Sinkhorn-style two-pass update with backtracking line search.

**Synaptic interaction scoring.** For each synapse and each candidate ligand–receptor gene pair, an effect size is computed by comparing gene-product co-expression between synaptically connected vs. non-synaptically connected type pairs (t-test). This identifies which molecular programs are enriched at specific synaptic connections.

---

## Citation

If you use ConnectionMiner in your research, please cite:

> Gupta, H.P., Azevedo, A., Chen, Y.-C.D., Xing, K., Sims, P.A., Varol, E., & Mann, R.S. (2025). **Decoding neuronal wiring by joint inference of cell identity and synaptic connectivity.** *bioRxiv*. https://doi.org/10.1101/2025.03.04.640006

**BibTeX:**
```bibtex
@article{gupta2025decoding,
  title   = {Decoding neuronal wiring by joint inference of cell identity and synaptic connectivity},
  author  = {Gupta, Himanshu Pawankumar and Azevedo, Anthony and Chen, Yu-Chieh David and Xing, Kristi and Sims, Peter A. and Varol, E. and Mann, Richard S.},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.03.04.640006},
  url     = {https://doi.org/10.1101/2025.03.04.640006}
}
```

---

## License

*To be added.*
