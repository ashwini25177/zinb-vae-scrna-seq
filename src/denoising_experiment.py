"""
denoising_experiment.py
------------------------
Tests the actual designed strength of the ZINB-VAE: recovering true
expression values that have been artificially masked as technical dropout
-- the same benchmark used in the scVI, MAGIC, and DCA papers.

Procedure:
  1. Take the real (already QC'd) raw counts for the HVG subset.
  2. Randomly zero out `corruption_rate` fraction of the *nonzero* entries
     (simulating additional technical dropout), recording their true values.
  3. Train each method on the CORRUPTED data only -- none of the methods
     ever see which entries were corrupted or their true values during
     training. This mirrors reality: in real data, we don't know which
     zeros are "true biological absence" vs "technical dropout"; we can
     only ask whether a model's learned structure can recover values that
     happen to have been masked in a controlled test.
  4. Compare each method's reconstruction specifically at the masked
     positions against the true original values.

Why this is the fairer test of the ZINB-VAE's actual value proposition:
  PCA and a plain autoencoder have no mechanism to distinguish "this zero
  is technical noise" from "this zero is real biology" -- they just try to
  reconstruct whatever they're shown. The ZINB-VAE explicitly models a
  separate dropout probability per gene/cell on top of the NB count
  distribution, which is precisely the mechanism that should help it
  recover corrupted values more accurately than methods with no such
  concept.

Usage:
    python src/denoising_experiment.py --input data/pbmc3k_processed.h5ad \
                                        --corruption_rate 0.1 --epochs 200
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from model import ZINBVAE, vae_loss
from baseline import MSEAutoencoder


def corrupt_counts(x_counts: np.ndarray, corruption_rate: float = 0.1, seed: int = 0):
    """
    Randomly zero out corruption_rate fraction of nonzero entries.
    Returns: corrupted counts, boolean mask of corrupted positions, true
    original values at those positions.
    """
    rng = np.random.default_rng(seed)
    nonzero_rows, nonzero_cols = np.nonzero(x_counts)
    n_nonzero = len(nonzero_rows)
    n_corrupt = int(n_nonzero * corruption_rate)

    chosen = rng.choice(n_nonzero, size=n_corrupt, replace=False)
    rows, cols = nonzero_rows[chosen], nonzero_cols[chosen]

    x_corrupted = x_counts.copy()
    true_values = x_counts[rows, cols].copy()
    x_corrupted[rows, cols] = 0

    mask = np.zeros_like(x_counts, dtype=bool)
    mask[rows, cols] = True

    print(f"Corrupted {n_corrupt:,} / {n_nonzero:,} nonzero entries "
          f"({corruption_rate*100:.1f}% of nonzero values)")
    return x_corrupted, mask, rows, cols, true_values


def train_zinb_vae_denoising(x_corrupted, library_size, n_hidden=128, n_latent=10,
                              epochs=200, batch_size=128, lr=1e-3, kl_warmup=20,
                              kl_weight_max=0.3, seed=0):
    torch.manual_seed(seed)
    x_counts_t = torch.tensor(x_corrupted, dtype=torch.float32)
    lib_t = torch.tensor(library_size, dtype=torch.float32)
    x_norm_t = torch.log1p(x_counts_t / lib_t.unsqueeze(1) * 1e4)

    dataset = TensorDataset(x_counts_t, x_norm_t, lib_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ZINBVAE(n_genes=x_corrupted.shape[1], n_hidden=n_hidden, n_latent=n_latent)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        kl_weight = min(1.0, epoch / max(1, kl_warmup)) * kl_weight_max
        for xb_counts, xb_norm, xb_lib in loader:
            optimizer.zero_grad()
            px_rate, theta, px_dropout, mu, logvar, z = model(xb_norm, xb_lib)
            loss, recon, kl = vae_loss(xb_counts, px_rate, theta, px_dropout, mu, logvar, kl_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        if epoch % 40 == 0 or epoch == epochs - 1:
            print(f"[ZINB-VAE denoising] Epoch {epoch:4d} | loss {loss.item():.2f}")

    model.eval()
    with torch.no_grad():
        px_rate, theta, px_dropout, mu, logvar, z = model(x_norm_t, lib_t)
    return px_rate.numpy()  # predicted mean count-scale expression for every cell/gene


def train_mse_ae_denoising(x_norm_corrupted, n_hidden=128, n_latent=10,
                            epochs=200, batch_size=128, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    x_tensor = torch.tensor(x_norm_corrupted, dtype=torch.float32)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MSEAutoencoder(n_genes=x_norm_corrupted.shape[1], n_hidden=n_hidden, n_latent=n_latent)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for (xb,) in loader:
            optimizer.zero_grad()
            x_hat, _ = model(xb)
            loss = loss_fn(x_hat, xb)
            loss.backward()
            optimizer.step()
        if epoch % 40 == 0 or epoch == epochs - 1:
            print(f"[MSE-AE denoising] Epoch {epoch:4d} | loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        x_hat, _ = model(x_tensor)
    return x_hat.numpy()


def run_experiment(input_path: str, corruption_rate: float = 0.1, epochs: int = 200,
                    results_dir: str = "results", seed: int = 0):
    adata = sc.read_h5ad(input_path)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    x_counts = np.asarray(
        adata_hvg.layers["counts"].todense()
        if hasattr(adata_hvg.layers["counts"], "todense")
        else adata_hvg.layers["counts"]
    )
    library_size = x_counts.sum(axis=1)  # fixed reference library size, computed pre-corruption

    x_corrupted, mask, rows, cols, true_values = corrupt_counts(
        x_counts, corruption_rate=corruption_rate, seed=seed
    )

    # Normalized versions, both using the SAME (original, pre-corruption) library
    # size as the denominator -- this isolates the imputation task itself rather
    # than confounding it with library-size shifts caused by the corruption.
    x_norm_original = np.log1p(x_counts / library_size[:, None] * 1e4)
    x_norm_corrupted = np.log1p(x_corrupted / library_size[:, None] * 1e4)
    true_log_norm = x_norm_original[rows, cols]  # ground truth, log-normalized scale

    predictions = {}

    # --- Trivial baselines (context for how hard/easy this task is) ---
    predictions["Zero (no imputation)"] = np.zeros_like(true_log_norm)
    gene_means = x_norm_corrupted.mean(axis=0)
    predictions["Gene-mean imputation"] = gene_means[cols]

    # --- PCA reconstruction ---
    print("\nFitting PCA reconstruction...")
    pca = PCA(n_components=10, random_state=seed)
    z_pca = pca.fit_transform(x_norm_corrupted)
    x_pca_recon = pca.inverse_transform(z_pca)
    predictions["PCA"] = x_pca_recon[rows, cols]

    # --- MSE Autoencoder ---
    print("\nTraining MSE-Autoencoder for denoising...")
    x_mse_recon = train_mse_ae_denoising(x_norm_corrupted, epochs=epochs, seed=seed)
    predictions["MSE-Autoencoder"] = x_mse_recon[rows, cols]

    # --- ZINB-VAE ---
    print("\nTraining ZINB-VAE for denoising...")
    px_rate = train_zinb_vae_denoising(x_corrupted, library_size, epochs=epochs, seed=seed)
    # convert predicted count-scale mean back to the same log-normalized scale
    px_rate_log_norm = np.log1p(px_rate / library_size[:, None] * 1e4)
    predictions["ZINB-VAE"] = px_rate_log_norm[rows, cols]

    # --- Evaluate: correlation + median absolute error vs true values ---
    rows_out = []
    for name, pred in predictions.items():
        corr, _ = pearsonr(pred, true_log_norm)
        mae = np.median(np.abs(pred - true_log_norm))
        rows_out.append({"method": name, "pearson_r": corr, "median_abs_error": mae})

    df = pd.DataFrame(rows_out).sort_values("pearson_r", ascending=False)
    df.to_csv(f"{results_dir}/denoising_comparison.csv", index=False)
    print("\nDenoising / imputation comparison (recovering artificially masked values):")
    print(df.to_string(index=False))
    print(f"\nSaved to: {results_dir}/denoising_comparison.csv")

    # --- Scatter plots: true vs predicted, for the 3 real methods ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, ["PCA", "MSE-Autoencoder", "ZINB-VAE"]):
        ax.scatter(true_log_norm, predictions[name], s=3, alpha=0.3)
        lims = [0, max(true_log_norm.max(), predictions[name].max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("True (log-normalized)")
        ax.set_ylabel("Predicted")
        r = df.loc[df["method"] == name, "pearson_r"].values[0]
        ax.set_title(f"{name} (r={r:.3f})")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/denoising_scatter.png", dpi=150)
    print(f"Saved scatter plot to: {results_dir}/denoising_scatter.png")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoising/imputation benchmark")
    parser.add_argument("--input", type=str, default="data/pbmc3k_processed.h5ad")
    parser.add_argument("--corruption_rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    run_experiment(args.input, args.corruption_rate, args.epochs, args.results_dir)
