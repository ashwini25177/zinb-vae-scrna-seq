"""
model.py
--------
A Variational Autoencoder for scRNA-seq count data using a Zero-Inflated
Negative Binomial (ZINB) reconstruction likelihood, in the spirit of scVI
(Lopez et al., 2018).

Why ZINB and not MSE?
  scRNA-seq counts are:
    - discrete (can't be negative, can't be fractional)
    - overdispersed (variance >> mean, unlike Poisson)
    - zero-inflated (many zeros arise from technical dropout, not biology)
  A plain autoencoder with MSE loss implicitly assumes Gaussian-distributed,
  continuous data -- a poor match for these properties. Modeling counts
  directly with a ZINB likelihood is the key design choice that separates
  this from a generic autoencoder, and is what lets the learned latent
  space represent genuine biological variation rather than being dominated
  by sequencing-depth / dropout noise.

Architecture:
  Encoder:  log1p(normalized counts) -> (mu, logvar) of latent z
  Decoder:  z (+ library size) -> (mean, dispersion, dropout logits) of a
            ZINB distribution over raw counts

Loss = -ZINB log-likelihood(raw counts | decoder output)  +  KL(q(z|x) || N(0,I))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, n_genes: int, n_hidden: int = 128, n_latent: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(n_hidden, n_latent)
        self.logvar = nn.Linear(n_hidden, n_latent)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """
    Decodes latent z back into the parameters of a ZINB distribution over
    raw gene counts:
        - mean (scaled by the cell's library size)
        - dispersion (gene-specific, learned as a free parameter -- this
          matches the scVI convention that dispersion is shared across
          cells for a given gene)
        - dropout logits (probability mass that is "structural zero",
          separate from the negative-binomial's own probability of a zero)
    """

    def __init__(self, n_genes: int, n_hidden: int = 128, n_latent: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
        )
        self.mean_decoder = nn.Linear(n_hidden, n_genes)   # log-scale mean, softmax-normalized
        self.dropout_decoder = nn.Linear(n_hidden, n_genes)  # dropout logits
        # gene-specific dispersion, shared across cells (standard scVI convention)
        self.log_theta = nn.Parameter(torch.randn(n_genes) * 0.1)

    def forward(self, z, library_size):
        h = self.net(z)
        # softmax gives per-gene proportions of expression, summing to 1 per cell
        px_scale = F.softmax(self.mean_decoder(h), dim=-1)
        px_rate = px_scale * library_size.unsqueeze(1)  # scale by cell's total counts
        px_dropout = self.dropout_decoder(h)
        theta = torch.exp(self.log_theta).clamp(min=1e-4, max=1e4)
        return px_rate, theta, px_dropout


class ZINBVAE(nn.Module):
    def __init__(self, n_genes: int, n_hidden: int = 128, n_latent: int = 10):
        super().__init__()
        self.encoder = Encoder(n_genes, n_hidden, n_latent)
        self.decoder = Decoder(n_genes, n_hidden, n_latent)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_norm, library_size):
        """
        x_norm: log1p-normalized input to the encoder (numerically stable)
        library_size: raw per-cell total counts, used to scale the decoder's
                      output back to count space
        """
        mu, logvar = self.encoder(x_norm)
        z = self.reparameterize(mu, logvar)
        px_rate, theta, px_dropout = self.decoder(z, library_size)
        return px_rate, theta, px_dropout, mu, logvar, z


def zinb_log_likelihood(x, mu, theta, pi, eps: float = 1e-8):
    """
    Zero-Inflated Negative Binomial log-likelihood.

    x:     observed raw counts                (batch, genes)
    mu:    NB mean (px_rate from decoder)      (batch, genes)
    theta: NB dispersion (inverse; larger = less dispersed) (genes,)
    pi:    dropout logits (structural zero probability, pre-sigmoid) (batch, genes)

    Follows the parameterization used in scVI:
        NB log-prob via the Gamma-Poisson mixture form
        ZINB mixes a point mass at zero (weight = sigmoid(pi)) with the NB
    """
    softplus_pi = F.softplus(-pi)  # log(1 + exp(-pi)) = -log(sigmoid(pi))

    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)

    # NB log-likelihood (standard form)
    nb_case = (
        theta * (log_theta_eps - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    zero_nb = theta * (log_theta_eps - log_theta_mu_eps)  # NB log-prob at x=0
    # log( sigmoid(-pi)*NB(0) + sigmoid(pi) ), computed via logsumexp for
    # numerical stability instead of raw exp/log.
    case_zero = torch.logsumexp(
        torch.stack([-pi - softplus_pi + zero_nb, -softplus_pi], dim=0), dim=0
    )
    case_nonzero = -softplus_pi + nb_case

    log_lik = torch.where(x < eps, case_zero, case_nonzero)
    return log_lik


def vae_loss(x_counts, px_rate, theta, px_dropout, mu, logvar, kl_weight: float = 1.0):
    """
    Total loss = -reconstruction log-likelihood + kl_weight * KL divergence.

    kl_weight allows for KL annealing (starting near 0 and ramping to 1 over
    the first several epochs), which helps prevent posterior collapse --
    a common failure mode where the decoder ignores the latent code entirely.
    """
    recon_log_lik = zinb_log_likelihood(x_counts, px_rate, theta, px_dropout)
    recon_loss = -recon_log_lik.sum(dim=-1).mean()

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kl = kl.mean()

    return recon_loss + kl_weight * kl, recon_loss, kl
