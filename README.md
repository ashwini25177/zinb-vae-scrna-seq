# Deep Learning for Single-Cell RNA-seq Cell-Type Annotation

A deep generative modeling pipeline for single-cell RNA-seq data, using a
Variational Autoencoder with a Negative-Binomial / ZINB likelihood
(in the spirit of [scVI](https://scvi-tools.org/)) to learn a biologically
meaningful latent representation of cell state, and evaluating it against
classical baselines (PCA, standard MSE-based autoencoders) on cell-type
separation.

> **Status:** Complete. Full pipeline trained and evaluated; see [Results](#results) below.

## Motivation

Raw scRNA-seq count matrices are high-dimensional, sparse, zero-inflated,
and overdispersed — a plain PCA or MSE-based autoencoder implicitly assumes
Gaussian-distributed data, which is a poor fit for count data and tends to
produce embeddings dominated by sequencing depth artifacts rather than
biology. This project builds a count-aware generative model and empirically
demonstrates *why* that distributional choice matters, rather than just
asserting it.

## Dataset

**PBMC 3k** (10x Genomics, Zheng et al. 2017) — 2,700 peripheral blood
mononuclear cells, a well-characterized dataset containing the expected
mix of immune cell types (T cells, B cells, NK cells, monocytes, dendritic
cells), loaded via `scanpy.datasets.pbmc3k()`. Cell-type reference labels
used for evaluation are derived from the data itself (see below), not
supplied with the dataset.

## Pipeline

```
Raw counts (2700 cells × ~32k genes)
        │
        ▼
QC metrics: n_genes, total_counts, % mitochondrial
        │
        ▼
Filtering: min_genes=200, min_cells=3, max_pct_mt=15%, max_genes=4000
        │
        ▼
Normalization: library-size scaling + log1p  (raw counts preserved in .layers)
        │
        ▼
Highly Variable Gene selection (top 2000)
        │
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
        ┌───────────────────────┬───────────────────────┐        Reference labels:
        ▼                       ▼                       ▼        Leiden clustering +
   PCA baseline          MSE Autoencoder        ZINB-VAE (this project)   marker gene scoring
        │                       │                       │                │
        └───────────────────────┴───────────────────────┘                │
                                 ▼                                        │
                  Clustering evaluation: ARI, NMI, Silhouette  <──────────┘
                                 │
                                 ▼
              Denoising benchmark: mask 10% of real values,
           compare recovery accuracy (Pearson r, median abs. error)
```

## Repository structure

```
genomics-dl-project/
├── README.md
├── requirements.txt
├── data/                          # raw/processed data (gitignored, regenerate via script)
├── notebooks/
│   └── 01_eda_and_preprocessing.ipynb   # QC decisions + visualizations, step by step
├── src/
│   ├── download_data.py           # dataset download (scanpy + direct-source fallback)
│   ├── data_processing.py         # QC, filtering, normalization, HVG selection
│   ├── model.py                   # ZINB-VAE architecture + loss
│   ├── train.py                   # training loop with KL annealing
│   ├── baseline.py                # PCA + MSE-Autoencoder baselines
│   ├── evaluate.py                # ARI/NMI/Silhouette + UMAP comparison
│   └── denoising_experiment.py    # dropout-recovery / imputation benchmark
├── results/                       # figures, metrics tables
└── .gitignore
```

## Getting started

```bash
git clone https://github.com/<your-username>/genomics-dl-project.git
cd genomics-dl-project
pip install -r requirements.txt

# 1. Download raw data + preprocess (QC, normalize, HVG selection)
python src/download_data.py --output data/pbmc3k_raw.h5ad
python src/data_processing.py --output data/pbmc3k_processed.h5ad

# 2. Train the ZINB-VAE
python src/train.py --input data/pbmc3k_processed.h5ad \
                     --output_model results/zinb_vae.pt \
                     --output_latent data/pbmc3k_with_latent.h5ad \
                     --epochs 400 \
                     --kl_weight_max 0.3

# 3. Run baselines (PCA + plain MSE-Autoencoder, same architecture, no ZINB)
python src/baseline.py --input data/pbmc3k_processed.h5ad \
                        --output data/pbmc3k_with_baselines.h5ad \
                        --epochs 200

# 4. Evaluate: ARI / NMI / Silhouette vs marker-gene-derived reference labels + UMAP figures
python src/evaluate.py --processed_path data/pbmc3k_processed.h5ad \
                        --latent_path data/pbmc3k_with_latent.h5ad \
                        --baseline_path data/pbmc3k_with_baselines.h5ad \
                        --results_dir results

# 5. Denoising / imputation benchmark
python src/denoising_experiment.py --input data/pbmc3k_processed.h5ad \
                                    --corruption_rate 0.1 --epochs 200

# Or explore interactively
jupyter notebook notebooks/01_eda_and_preprocessing.ipynb
```

Reference cell-type labels for evaluation are derived directly from the
data itself: Leiden clustering followed by scoring each cluster against
canonical PBMC marker genes (see `src/evaluate.py` and the Results section
below for why — scanpy's previously-available pre-annotated version of
this dataset is no longer distributed in current versions).

## Model: ZINB-VAE

The core model (`src/model.py`) is a VAE where the decoder outputs the
parameters of a **Zero-Inflated Negative Binomial** distribution rather
than reconstructing expression directly via MSE:

- **Encoder**: log-normalized expression → latent mean/log-variance (10-dim by default)
- **Decoder**: latent z → per-gene mean (scaled by library size), dispersion, and dropout logits
- **Loss**: `-ZINB log-likelihood + KL(q(z|x) || N(0,I))`, with linear KL annealing over the first 20 epochs to avoid posterior collapse

This mirrors the core idea behind [scVI](https://scvi-tools.org/) (Lopez et
al., 2018) — implemented from scratch here rather than imported, so the
distributional reasoning is fully visible and explained line-by-line.

**Baselines** (`src/baseline.py`) hold architecture size constant and vary
only the loss/method, isolating the effect of the distributional
assumption:
- Classical PCA on scaled, log-normalized HVGs
- A plain autoencoder (identical encoder/decoder MLP sizes) trained with MSE loss

**Evaluation** (`src/evaluate.py`) derives reference cell-type labels
directly from the data via Leiden clustering + canonical PBMC marker gene
scoring (scanpy's previously-available pre-annotated version of this
dataset is no longer distributed in current scanpy versions, so labels
are derived rather than fetched), and reports ARI, NMI, and Silhouette
score for each method plus UMAP visualizations.

**Denoising benchmark** (`src/denoising_experiment.py`) tests the ZINB-VAE's
actual designed strength: 10% of real nonzero values are artificially
masked (simulating extra technical dropout), each method is trained
without knowledge of which values were masked, and recovery accuracy is
compared at those specific positions — see Results below.

## Results

### Clustering / cell-type separation

Reference cell types derived via Leiden clustering + canonical PBMC marker
gene scoring (see `src/evaluate.py`); 2698 cells, 7 resulting cell types
(CD4 T cells, NK cells, CD14+ Monocytes, B cells, FCGR3A+ Monocytes,
Dendritic cells, Megakaryocytes).

| Method | ARI | NMI | Silhouette |
|---|---|---|---|
| PCA | **0.889** | **0.862** | **0.405** |
| MSE-Autoencoder | 0.593 | 0.751 | 0.293 |
| ZINB-VAE | 0.484 | 0.669 | 0.039 |

**Finding: plain PCA wins decisively on pure cluster separation.** This is
a legitimate, reproducible result (confirmed stable across multiple
hyperparameter settings — KL weight capping, dropout removal, longer
training all left this ranking essentially unchanged), not a bug. PBMC 3k
is small (2698 cells), clean, and single-batch — exactly the regime where
a linear method can already capture most of the separable structure. The
scVI paper (Lopez et al., 2018), which this project's model is based on,
is itself explicit that generative models like this earn their keep
primarily on **larger, noisier, multi-batch** datasets — not tidy,
curated ones like this. A from-scratch ZINB-VAE trading off some
clustering purity for a properly-regularized generative latent space, on
a dataset this small and clean, matches expectations rather than
contradicting them.

### Denoising / imputation benchmark

10% of real nonzero expression values were artificially masked (set to
zero) to simulate additional technical dropout; each method was trained
*without* knowledge of which values were masked, then evaluated on how
well it recovered the true original values at those specific positions.

| Method | Pearson r | Median absolute error |
|---|---|---|
| Gene-mean imputation (trivial baseline) | 0.716 | 2.350 |
| PCA | 0.684 | 2.255 |
| MSE-Autoencoder | 0.682 | 2.317 |
| **ZINB-VAE** | 0.636 | **0.802** |
| Zero (no imputation) | undefined | 3.050 |

**Finding: the ZINB-VAE reduces typical-case recovery error by ~3x over
every other method** (median absolute error 0.80 vs. 2.25–2.35), while
scoring slightly lower on pooled Pearson correlation. These two metrics
disagree because Pearson correlation here is computed by pooling all
masked entries across genes with wildly different expression scales — a
metric dominated by *which gene* a value belongs to rather than *how
accurately a specific cell's value* was recovered. The gene-mean baseline
inflates this metric almost by construction (its prediction literally
is each gene's average), while ignoring cell-specific state entirely.
Median absolute error, being robust to this cross-gene scale confound and
to outliers, more fairly reflects the accuracy of recovering the typical
(usually small-count) masked value — and this is exactly where the
ZINB-VAE's explicit modeling of count sparsity and dropout gives it a
real, substantial advantage over methods with no such mechanism.

**Takeaway:** the two experiments tell complementary stories rather than
a simple "which model is better" — PCA is the stronger choice here for
pure cluster separation on small, clean, single-batch data, while the
ZINB-VAE is the stronger choice for recovering true expression values
from technical noise, which is the problem it was actually designed to
solve. Figures for both experiments are in `results/`
(`umap_*.png`, `denoising_scatter.png`).

## Preprocessing design decisions

| Step | Threshold | Rationale |
|---|---|---|
| Min genes/cell | 200 | Filters empty droplets / low-quality captures |
| Min cells/gene | 3 | Removes genes with near-zero signal (dropout noise) |
| Max % mitochondrial | 15% | High mito % indicates dying/stressed cells (cytoplasmic RNA leaks, mito RNA retained) |
| Max genes/cell | 4000 | Unusually high gene counts suggest doublets (two cells per droplet) |
| Normalization | library-size + log1p | Standard scanpy convention; raw counts kept in `.layers["counts"]` for count-based likelihoods downstream |
| HVG selection | top 2000, seurat flavor | Reduces dimensionality while retaining informative genes for clustering/modeling |

## Roadmap

- [x] Data acquisition + QC + normalization pipeline
- [x] EDA notebook with QC distribution plots and baseline PCA/UMAP/Leiden clustering
- [x] ZINB-VAE model implementation (PyTorch) — `src/model.py`
- [x] Training script with KL annealing (beta-VAE style capping) — `src/train.py`
- [x] Baseline comparisons: raw PCA, standard MSE autoencoder (same architecture) — `src/baseline.py`
- [x] Quantitative evaluation: ARI / NMI / Silhouette vs marker-gene-derived reference labels + UMAP figures — `src/evaluate.py`
- [x] Denoising/imputation benchmark (the ZINB-VAE's actual designed strength) — `src/denoising_experiment.py`
- [x] Full pipeline run + results populated above
- [ ] Optional: benchmark against `scvi-tools`' reference implementation
- [ ] Optional: batch-correction experiment (requires a second dataset/batch)

## References

- Lopez et al., *Deep generative modeling for single-cell transcriptomics*, Nature Methods (2018) — the scVI paper this project takes inspiration from
- Luecken & Theis, *Current best practices in single-cell RNA-seq analysis: a tutorial*, Molecular Systems Biology (2019)
- Zheng et al., *Massively parallel digital transcriptional profiling of single cells*, Nature Communications (2017) — source of the PBMC 3k dataset

## License

MIT
