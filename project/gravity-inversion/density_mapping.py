"""
Differentiable mapping from 18D categorical embedding space to physical density.

Provides a soft-decode pathway so that gradients can flow from gravity data misfit
back through the density field into the embedding-space model state.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


# 15 density values aligned to embedding indices.
# Index 0 = air/background (raw data index -1, shifted by +1 in embed()).
# Indices 1-14 = user-provided rock categories 0-13.
DENSITY_TABLE_15: List[float] = [
    0.00,   # idx 0: air / background
    2.70,   # idx 1: bedrock
    2.10,   # idx 2: sediment
    2.20,   # idx 3: sediment
    2.25,   # idx 4: sediment
    2.30,   # idx 5: sediment
    2.35,   # idx 6: sediment
    2.85,   # idx 7: dike
    2.90,   # idx 8: dike
    2.95,   # idx 9: dike
    2.65,   # idx 10: intrusion
    2.70,   # idx 11: intrusion
    2.75,   # idx 12: intrusion
    3.40,   # idx 13: blob / ore
    3.80,   # idx 14: blob / ore
]


class DifferentiableDensityMapper:
    """Map embedding-space states to physical density via temperature-scaled soft decoding.

    Pipeline (all differentiable w.r.t. ``m_t``):

    1. Normalise ``m_t`` along the embedding dimension.
    2. Compute cosine similarity with each of the 15 embedding weight vectors.
    3. Apply temperature-scaled softmax → class probabilities.
    4. Weighted sum with the density look-up table → density volume.

    Parameters
    ----------
    embedding_weights : torch.Tensor
        Shape ``(n_categories, embedding_dim)`` — e.g. ``model.embedding.weight``.
    density_values : list[float] | None
        Per-category density.  Defaults to :data:`DENSITY_TABLE_15`.
    temperature : float
        Softmax temperature (higher → more peaked probabilities).
    """

    def __init__(
        self,
        embedding_weights: torch.Tensor,
        density_values: Optional[List[float]] = None,
        temperature: float = 10.0,
    ):
        self.embedding_weights = embedding_weights.detach().clone()  # (K, E)
        self.temperature = temperature

        densities = density_values if density_values is not None else DENSITY_TABLE_15
        self.density_values = torch.tensor(densities, dtype=torch.float32)  # (K,)

        assert self.embedding_weights.shape[0] == self.density_values.shape[0], (
            f"Number of categories in embedding ({self.embedding_weights.shape[0]}) "
            f"does not match density table length ({self.density_values.shape[0]})"
        )

    # ------------------------------------------------------------------
    def to(self, device: torch.device):
        """Move internal tensors to *device* (convenience helper)."""
        self.embedding_weights = self.embedding_weights.to(device)
        self.density_values = self.density_values.to(device)
        return self

    # ------------------------------------------------------------------
    def soft_decode_to_probabilities(self, m_t: torch.Tensor) -> torch.Tensor:
        """Return per-voxel class probabilities.

        Parameters
        ----------
        m_t : torch.Tensor
            ``(B, E, X, Y, Z)`` current state in embedding space.

        Returns
        -------
        torch.Tensor
            ``(B, K, X, Y, Z)`` class probabilities.
        """
        emb = self.embedding_weights  # (K, E)

        # Normalise
        m_norm = F.normalize(m_t, dim=1)  # (B, E, X, Y, Z)
        emb_norm = F.normalize(emb, dim=1)  # (K, E)

        # Cosine similarity  →  (B, K, X, Y, Z)
        logits = torch.einsum("bexyz, ke -> bkxyz", m_norm, emb_norm)

        # Temperature-scaled softmax along category axis
        probs = F.softmax(logits * self.temperature, dim=1)
        return probs

    # ------------------------------------------------------------------
    def soft_decode_to_density(self, m_t: torch.Tensor) -> torch.Tensor:
        """Differentiable mapping from embedding state to density volume.

        Parameters
        ----------
        m_t : torch.Tensor
            ``(B, E, X, Y, Z)`` current state in embedding space.
            **Must** have ``requires_grad=True`` if you need gradients.

        Returns
        -------
        torch.Tensor
            ``(B, 1, X, Y, Z)`` physical density.
        """
        probs = self.soft_decode_to_probabilities(m_t)  # (B, K, X, Y, Z)
        d = self.density_values.to(m_t.device)  # (K,)

        # Weighted sum:  density = sum_k  p_k * rho_k
        density = torch.einsum("bkxyz, k -> bxyz", probs, d)
        return density.unsqueeze(1)  # (B, 1, X, Y, Z)
