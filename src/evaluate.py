"""
evaluate.py
-----------
Compares the ZINB-VAE latent space against the PCA and MSE-Autoencoder
baselines, using reference cell-type labels.

Reference labels: PBMC3k does not ship with an independently-curated
cell-type truth file -- every classic scanpy/Seurat tutorial derives cell
types the same way we do here: Leiden clustering followed by annotating
each cluster using canonical PBMC marker genes (e.g. MS4A1 -> B cells,
CD14/LYZ -> CD14+ Monocytes, GNLY/NKG7 -> NK cells). We do this once, on
the processed expression matrix (independent of the VAE/PCA/AE embeddings
being evaluated), and treat the result as the reference against which all
three embeddings' clusterability is measured.

Metrics:
  - ARI  (Adjusted Rand Index):        cluster agreement vs reference labels, chance-corrected
  - NMI  (Normalized Mutual Info):     information-theoretic cluster agreement
  - Silhouette score:                  how well-separated reference cell-type groups
                                        are in the embedding space (using reference
                                        labels directly, not derived clusters)

Usage:
    python src/evaluate.py \
        --processed_path data/pbmc3k_processed.h5ad \
        --latent_path data/pbmc3k_with_latent.h5ad \
        --baseline_path data/pbmc3k_with_baselines.h5ad
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Canonical PBMC marker genes (standard scanpy/Seurat PBMC3k tutorial set)
MARKER_GENES = {
    "CD4 T cells": ["IL7R", "CD3D", "CD3E"],
    "CD8 T cells": ["CD8A", "CD8B"],
    "B cells": ["MS4A1", "CD79A"],
    "CD14+ Monocytes": ["CD14", "LYZ"],
    "FCGR3A+ Monocytes": ["FCGR3A", "MS4A7"],
    "NK cells": ["GNLY", "NKG7"],
    "Dendritic cells": ["FCER1A", "CST3"],
    "Megakaryocytes": ["PPBP"],
}


def derive_reference_labels(processed_path: str, resolution: float = 1.0) -> pd.Series:
    """
    Cluster the processed expression matrix (Leiden) and annotate each
    cluster with the cell type whose marker genes score highest on average
    within that cluster.
    """
    adata = sc.read_h5ad(processed_path)

    # Clustering on scaled HVG expression (standard practice)
    adata_c = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_c, max_value=10)
    sc.tl.pca(adata_c, n_comps=30)
    sc.pp.neighbors(adata_c)
    sc.tl.leiden(adata_c, resolution=resolution)
    adata.obs["leiden"] = adata_c.obs["leiden"].values

    # Score each cell type's marker genes on the full (log-normalized) matrix,
    # restricting each marker list to genes actually present post-filtering.
    for cell_type, genes in MARKER_GENES.items():
        genes_present = [g for g in genes if g in adata.var_names]
        if not genes_present:
            print(f"Warning: none of the marker genes for '{cell_type}' "
                  f"({genes}) survived QC filtering -- skipping this cell type.")
            continue
        sc.tl.score_genes(adata, gene_list=genes_present, score_name=f"score_{cell_type}")

    score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
    if not score_cols:
        raise RuntimeError(
            "None of the canonical marker genes were found in this dataset -- "
            "cannot derive reference labels. Check gene naming (e.g. symbols vs IDs)."
        )

    # For each Leiden cluster, assign the cell type with the highest average marker score
    cluster_scores = adata.obs.groupby("leiden")[score_cols].mean()
    cluster_to_label = cluster_scores.idxmax(axis=1).str.replace("score_", "", regex=False)

    labels = adata.obs["leiden"].map(cluster_to_label)
    labels.index = adata.obs_names
    labels.name = "cell_type"

    print("Cluster -> cell type assignment:")
    print(cluster_to_label.to_string())
    print("\nCell type counts:")
    print(labels.value_counts().to_string())

    return labels


def merge_embeddings_and_labels(processed_path: str, latent_path: str, baseline_path: str):
    """
    Loads the VAE-latent file and the baselines file, merges their embeddings
    onto a common set of cells, and attaches reference cell-type labels
    derived from marker-gene-annotated clustering.
    """
    adata_vae = sc.read_h5ad(latent_path)
    adata_base = sc.read_h5ad(baseline_path)
    labels = derive_reference_labels(processed_path)

    common_cells = (
        set(adata_vae.obs_names) & set(adata_base.obs_names) & set(labels.index)
    )
    common_cells = sorted(common_cells)
    print(f"\nCells with VAE + baseline embeddings + reference labels: {len(common_cells)}")

    embeddings = {
        "ZINB-VAE": adata_vae[common_cells].obsm["X_vae"],
        "PCA": adata_base[common_cells].obsm["X_pca_baseline"],
        "MSE-Autoencoder": adata_base[common_cells].obsm["X_mse_ae"],
    }
    y_true = labels.loc[common_cells].values
    return embeddings, y_true, common_cells


def evaluate_embedding(embedding: np.ndarray, y_true: np.ndarray, seed: int = 0) -> dict:
    n_clusters = len(np.unique(y_true))
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    y_pred = kmeans.fit_predict(embedding)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    sil = silhouette_score(embedding, y_true)  # silhouette against TRUE labels

    return {"ARI": ari, "NMI": nmi, "Silhouette": sil}


def make_umap_figure(embedding: np.ndarray, y_true: np.ndarray, title: str, save_path: str):
    tmp = ad.AnnData(X=embedding)
    tmp.obs["cell_type"] = pd.Categorical(y_true)
    sc.pp.neighbors(tmp, use_rep="X")
    sc.tl.umap(tmp)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc.pl.umap(tmp, color="cell_type", ax=ax, show=False, title=title, frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved UMAP figure: {save_path}")


def run_evaluation(processed_path: str, latent_path: str, baseline_path: str, results_dir: str = "results"):
    embeddings, y_true, cells = merge_embeddings_and_labels(processed_path, latent_path, baseline_path)

    rows = []
    for name, emb in embeddings.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_embedding(emb, y_true)
        metrics["method"] = name
        rows.append(metrics)
        make_umap_figure(emb, y_true, title=name, save_path=f"{results_dir}/umap_{name.replace(' ', '_')}.png")

    df = pd.DataFrame(rows)[["method", "ARI", "NMI", "Silhouette"]].sort_values(
        "ARI", ascending=False
    )
    df.to_csv(f"{results_dir}/comparison_metrics.csv", index=False)
    print("\nFinal comparison:")
    print(df.to_string(index=False))
    print(f"\nSaved metrics table to: {results_dir}/comparison_metrics.csv")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VAE vs baselines against reference cell types")
    parser.add_argument("--processed_path", type=str, default="data/pbmc3k_processed.h5ad")
    parser.add_argument("--latent_path", type=str, default="data/pbmc3k_with_latent.h5ad")
    parser.add_argument("--baseline_path", type=str, default="data/pbmc3k_with_baselines.h5ad")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    run_evaluation(args.processed_path, args.latent_path, args.baseline_path, args.results_dir)
