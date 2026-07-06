"""
baseline.py
-----------
Two baselines to compare against the ZINB-VAE:

  1. PCA -- the classical, non-deep-learning baseline. Standard practice
     in every scRNA-seq pipeline (Seurat, scanpy tutorials), so it's the
     natural "does deep learning even help here" sanity check.

  2. Plain MSE Autoencoder -- same architecture (encoder/decoder sizes,
     latent dimension) as the ZINB-VAE, but trained with a standard
     Gaussian/MSE reconstruction loss on the log-normalized data instead
     of a count-aware ZINB likelihood. This isolates the effect of the
     *distributional assumption* specifically, holding architecture size
     constant -- i.e. it answers "does modeling counts properly matter,
     independent of just using a neural network at all?"

Usage:
    python src/baseline.py --input data/pbmc3k_processed.h5ad
"""

import argparse
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MSEAutoencoder(nn.Module):
    """Same encoder/decoder sizing as the ZINB-VAE's MLP trunks, but a
    plain deterministic autoencoder with MSE loss on log-normalized data."""

    def __init__(self, n_genes: int, n_hidden: int = 128, n_latent: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_genes),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def run_pca_baseline(adata, n_comps: int = 10):
    """Standard PCA on scaled, log-normalized HVG expression."""
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=n_comps)
    return adata_hvg.obsm["X_pca"]


def train_mse_autoencoder(
    x_norm: np.ndarray,
    n_hidden: int = 128,
    n_latent: int = 10,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
):
    torch.manual_seed(seed)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MSEAutoencoder(n_genes=x_norm.shape[1], n_hidden=n_hidden, n_latent=n_latent)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for (xb,) in loader:
            optimizer.zero_grad()
            x_hat, _ = model(xb)
            loss = loss_fn(x_hat, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"[MSE-AE] Epoch {epoch:4d} | MSE loss {total_loss / n_batches:.4f}")

    model.eval()
    with torch.no_grad():
        _, z = model(x_tensor)
    return z.numpy()


def run_baselines(input_path: str, output_path: str, n_latent: int = 10, epochs: int = 200):
    adata = sc.read_h5ad(input_path)

    print("Running PCA baseline...")
    pca_latent = run_pca_baseline(adata, n_comps=n_latent)
    adata.obsm["X_pca_baseline"] = pca_latent

    print("Training MSE-Autoencoder baseline...")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    x_norm = np.asarray(
        adata_hvg.X.todense() if hasattr(adata_hvg.X, "todense") else adata_hvg.X
    )
    ae_latent = train_mse_autoencoder(x_norm, n_latent=n_latent, epochs=epochs)
    adata.obsm["X_mse_ae"] = ae_latent

    adata.write_h5ad(output_path)
    print(f"Saved baseline embeddings to: {output_path}")
    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCA and MSE-AE baselines")
    parser.add_argument("--input", type=str, default="data/pbmc3k_processed.h5ad")
    parser.add_argument("--output", type=str, default="data/pbmc3k_with_baselines.h5ad")
    parser.add_argument("--n_latent", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    run_baselines(args.input, args.output, n_latent=args.n_latent, epochs=args.epochs)
