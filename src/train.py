"""
train.py
--------
Trains the ZINB-VAE on the preprocessed PBMC 3k dataset and saves:
    - the trained model checkpoint
    - the learned latent embeddings for every cell (for downstream evaluation)

Usage:
    python src/train.py --input data/pbmc3k_processed.h5ad --epochs 200
"""

import argparse
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import ZINBVAE, vae_loss


def load_training_tensors(h5ad_path: str):
    """
    Loads the processed AnnData and prepares tensors for training:
      - x_counts: raw counts, subset to highly variable genes
      - x_norm:   log1p-normalized version, used as encoder input
      - library_size: per-cell raw total counts, used to scale decoder output
    """
    adata = sc.read_h5ad(h5ad_path)
    adata = adata[:, adata.var["highly_variable"]].copy()

    x_counts = np.asarray(adata.layers["counts"].todense()
                           if hasattr(adata.layers["counts"], "todense")
                           else adata.layers["counts"])
    x_norm = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X)
    library_size = x_counts.sum(axis=1)

    return (
        torch.tensor(x_counts, dtype=torch.float32),
        torch.tensor(x_norm, dtype=torch.float32),
        torch.tensor(library_size, dtype=torch.float32),
        adata,
    )


def kl_anneal_weight(epoch: int, warmup_epochs: int, max_weight: float = 1.0) -> float:
    """
    Linearly ramp KL weight from 0 to max_weight over warmup_epochs.

    Capping max_weight below 1.0 (the "beta-VAE" trick, beta = max_weight)
    is a standard fix for posterior collapse: if the KL term is allowed to
    dominate, the encoder is pushed to make every cell's posterior match
    the prior N(0,I), discarding cell-specific information rather than
    encoding real biological variation. A signature of this failure mode
    is the aggregate latent statistics looking suspiciously identical to
    the prior (std ~= 1.0 in every dimension) despite genuine biological
    heterogeneity in the input.
    """
    ramped = min(1.0, epoch / max(1, warmup_epochs))
    return ramped * max_weight


def train(
    input_path: str,
    output_model_path: str,
    output_latent_path: str,
    n_latent: int = 10,
    n_hidden: int = 128,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    kl_warmup_epochs: int = 20,
    kl_weight_max: float = 1.0,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_counts, x_norm, library_size, adata = load_training_tensors(input_path)
    n_cells, n_genes = x_counts.shape
    print(f"Training on {n_cells} cells x {n_genes} genes")

    dataset = TensorDataset(x_counts, x_norm, library_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ZINBVAE(n_genes=n_genes, n_hidden=n_hidden, n_latent=n_latent)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        model.train()
        kl_weight = kl_anneal_weight(epoch, kl_warmup_epochs, max_weight=kl_weight_max)
        epoch_loss, epoch_recon, epoch_kl, n_batches = 0.0, 0.0, 0.0, 0

        for xb_counts, xb_norm, xb_lib in loader:
            optimizer.zero_grad()
            px_rate, theta, px_dropout, mu, logvar, z = model(xb_norm, xb_lib)
            loss, recon, kl = vae_loss(
                xb_counts, px_rate, theta, px_dropout, mu, logvar, kl_weight=kl_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        history.append((epoch, avg_loss, avg_recon, avg_kl, kl_weight))

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d} | loss {avg_loss:10.2f} | "
                f"recon {avg_recon:10.2f} | kl {avg_kl:8.4f} | kl_weight {kl_weight:.2f}"
            )

    torch.save(
        {"model_state_dict": model.state_dict(), "n_genes": n_genes,
         "n_hidden": n_hidden, "n_latent": n_latent},
        output_model_path,
    )
    print(f"Saved model checkpoint to: {output_model_path}")

    # Compute final latent embeddings for every cell (no dropout / eval mode)
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encoder(x_norm)
    latent = mu.numpy()  # use the mean of q(z|x) as the point estimate, standard practice

    adata.obsm["X_vae"] = latent
    adata.write_h5ad(output_latent_path)
    print(f"Saved latent embeddings to: {output_latent_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ZINB-VAE on scRNA-seq data")
    parser.add_argument("--input", type=str, default="data/pbmc3k_processed.h5ad")
    parser.add_argument("--output_model", type=str, default="results/zinb_vae.pt")
    parser.add_argument("--output_latent", type=str, default="data/pbmc3k_with_latent.h5ad")
    parser.add_argument("--n_latent", type=int, default=10)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl_warmup_epochs", type=int, default=20)
    parser.add_argument("--kl_weight_max", type=float, default=1.0,
                         help="Cap on KL weight (beta-VAE style). Lower values "
                              "(e.g. 0.1-0.5) reduce risk of posterior collapse "
                              "at the cost of a less regularized latent space.")
    args = parser.parse_args()

    train(
        input_path=args.input,
        output_model_path=args.output_model,
        output_latent_path=args.output_latent,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        kl_warmup_epochs=args.kl_warmup_epochs,
        kl_weight_max=args.kl_weight_max,
    )
