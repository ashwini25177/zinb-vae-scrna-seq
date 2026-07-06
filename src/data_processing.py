"""
data_processing.py
-------------------
Data acquisition and preprocessing pipeline for single-cell RNA-seq data.

Dataset: PBMC 3k (10x Genomics) — a classic, well-annotated peripheral blood
mononuclear cell dataset, bundled directly with scanpy for reproducibility.

Pipeline steps (standard scRNA-seq QC, following best practices from the
scanpy / Luecken & Theis "Current best practices" workflow):
    1. Load raw counts
    2. QC metric calculation (gene counts, total counts, % mitochondrial)
    3. Filtering low-quality cells and rarely-expressed genes
    4. Doublet-risk / extreme outlier trimming
    5. Normalization (library-size + log1p)
    6. Highly variable gene (HVG) selection
    7. Save processed AnnData object for modeling stage

Usage:
    python src/data_processing.py --output data/pbmc3k_processed.h5ad
"""

import argparse
import scanpy as sc
import anndata as ad
import numpy as np


def load_raw_data() -> ad.AnnData:
    """
    Load the PBMC 3k dataset bundled with scanpy.

    This avoids relying on external download servers that may be
    unreachable in restricted network environments, while still using
    the same canonical, published dataset (Zheng et al., 2017, 10x Genomics).
    """
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    return adata


def compute_qc_metrics(adata: ad.AnnData) -> ad.AnnData:
    """Annotate mitochondrial genes and compute standard QC metrics."""
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    return adata


def filter_cells_and_genes(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mt: float = 15.0,
    max_genes: int = 4000,
) -> ad.AnnData:
    """
    Apply standard QC thresholds.

    - min_genes: cells expressing fewer genes than this are likely empty
      droplets / low-quality captures.
    - min_cells: genes expressed in fewer cells than this are likely noise
      or dropout artifacts, and add little signal.
    - max_pct_mt: cells with high mitochondrial content are likely dying
      or stressed cells with ruptured membranes (cytoplasmic RNA leaks out,
      mitochondrial RNA is relatively retained).
    - max_genes: cells with unusually high gene counts are candidate
      doublets (two cells captured in one droplet).
    """
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt, :].copy()
    adata = adata[adata.obs["n_genes_by_counts"] < max_genes, :].copy()

    print(
        f"QC filtering: {n_cells_before} -> {adata.n_obs} cells, "
        f"{n_genes_before} -> {adata.n_vars} genes"
    )
    return adata


def normalize_data(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    """
    Library-size normalization + log1p transform.

    Note: raw counts are preserved in adata.layers["counts"] before this
    step, since downstream models expecting a count-based likelihood
    (e.g. Negative Binomial / ZINB, as used in the VAE stage) need the
    raw integer counts, not the log-normalized values.
    """
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def select_highly_variable_genes(
    adata: ad.AnnData, n_top_genes: int = 2000
) -> ad.AnnData:
    """Select HVGs to reduce dimensionality before modeling."""
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor="seurat", subset=False
    )
    return adata


def run_pipeline(output_path: str) -> ad.AnnData:
    adata = load_raw_data()
    adata = compute_qc_metrics(adata)
    adata = filter_cells_and_genes(adata)
    adata = normalize_data(adata)
    adata = select_highly_variable_genes(adata)

    adata.write_h5ad(output_path)
    print(f"Saved processed AnnData to: {output_path}")
    print(adata)
    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PBMC 3k scRNA-seq data")
    parser.add_argument(
        "--output",
        type=str,
        default="data/pbmc3k_processed.h5ad",
        help="Path to save the processed AnnData object",
    )
    args = parser.parse_args()
    run_pipeline(args.output)
