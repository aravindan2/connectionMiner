from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import CmResult, RawData


def cm_export_type_gene_probabilities(raw: RawData, cm: CmResult, cfg: dict[str, Any]) -> None:
    run_dir = cfg.get("run_dir")
    if not run_dir:
        print("Warning: cfg.run_dir is empty, skipping type_gene_probabilities export.")
        return

    G_type_prob_full = cm.meta.get("G_type_prob_full")
    if G_type_prob_full is None:
        print("Warning: cm.G_type_prob_full missing, nothing to export.")
        return

    output_path = Path(run_dir) / "type_gene_probabilities.xlsx"

    Ntypes = G_type_prob_full.shape[0]
    genes = raw.genes_shared.astype(str)

    type_names = raw.meta.get("all_names", np.array([], dtype=str)).astype(str)
    if type_names.size < Ntypes:
        type_names = np.pad(type_names, (0, Ntypes - type_names.size), constant_values="")

    n_cells = cm.meta.get("n_cells_type", np.full(Ntypes, np.nan))
    cell_contributions = cm.meta.get("cell_contributions", np.full(Ntypes, np.nan))
    identifiable = cm.meta.get("identifiable_type", np.zeros(Ntypes, dtype=bool)).astype(bool)

    export_mat = G_type_prob_full.copy()
    export_mat[~identifiable, :] = np.nan

    base_df = pd.DataFrame(
        {
            "type_name": type_names[:Ntypes],
            "n_cells": np.asarray(n_cells)[:Ntypes],
            "cell_contributions": np.asarray(cell_contributions)[:Ntypes],
            "identifiable": identifiable[:Ntypes].astype(int),
        }
    )

    gene_df = pd.DataFrame(
        export_mat,
        columns=[f"{gene}_prob" for gene in genes.tolist()],
    )
    df = pd.concat([base_df, gene_df], axis=1)

    df.to_excel(output_path, index=False)
    print(f"  Exported type x gene probabilities to {output_path}")
