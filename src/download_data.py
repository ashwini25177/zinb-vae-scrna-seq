"""
download_data.py
-----------------
Downloads the PBMC 3k single-cell RNA-seq dataset (10x Genomics, Zheng et al. 2017).

Two methods are provided:
    1. via_scanpy()  -- simplest, uses scanpy's built-in loader (recommended)
    2. via_10x_direct() -- downloads the raw 10x Genomics tar.gz directly and
       builds the AnnData object manually. Use this only if method 1 fails
       (e.g. scanpy's data mirror is temporarily down).

Usage:
    python download_data.py                  # method 1 (default)
    python download_data.py --method direct   # method 2
"""

import argparse
import os
import tarfile
import urllib.request

import scanpy as sc
import anndata as ad


def via_scanpy(output_path: str) -> ad.AnnData:
    """Download via scanpy's built-in dataset loader (recommended)."""
    print("Downloading PBMC 3k via scanpy.datasets.pbmc3k() ...")
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    adata.write_h5ad(output_path)
    print(f"Saved raw data to: {output_path}")
    print(adata)
    return adata


def via_10x_direct(output_path: str, tmp_dir: str = "data/raw_10x") -> ad.AnnData:
    """
    Fallback: download the original 10x Genomics tar.gz directly and build
    the AnnData object with sc.read_10x_mtx(), in case scanpy's own mirror
    is unavailable.
    """
    url = (
        "https://cf.10xgenomics.com/samples/cell/pbmc3k/"
        "pbmc3k_filtered_gene_bc_matrices.tar.gz"
    )
    os.makedirs(tmp_dir, exist_ok=True)
    tar_path = os.path.join(tmp_dir, "pbmc3k_filtered_gene_bc_matrices.tar.gz")

    print(f"Downloading raw 10x Genomics data from:\n  {url}")
    urllib.request.urlretrieve(url, tar_path)

    print("Extracting...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp_dir)

    # Extracted structure: filtered_gene_bc_matrices/hg19/{matrix.mtx, genes.tsv, barcodes.tsv}
    matrix_dir = os.path.join(tmp_dir, "filtered_gene_bc_matrices", "hg19")
    adata = sc.read_10x_mtx(matrix_dir, var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()

    adata.write_h5ad(output_path)
    print(f"Saved raw data to: {output_path}")
    print(adata)
    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PBMC 3k scRNA-seq dataset")
    parser.add_argument(
        "--method",
        choices=["scanpy", "direct"],
        default="scanpy",
        help="Download method: 'scanpy' (recommended) or 'direct' (fallback)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/pbmc3k_raw.h5ad",
        help="Path to save the raw AnnData object",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.method == "scanpy":
        via_scanpy(args.output)
    else:
        via_10x_direct(args.output)
