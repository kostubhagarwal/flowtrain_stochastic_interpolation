
import torch
import torch.nn.functional as F
from typing import List, Optional


# g/cm³ for each of the 15 rock categories. fixed, not learned.
DENSITY_TABLE_15: List[float] = [
    0.00,   # 0:  air
    2.70,   # 1:  bedrock
    2.10,   # 2:  sediment
    2.20,   # 3:  sediment
    2.25,   # 4:  sediment
    2.30,   # 5:  sediment
    2.35,   # 6:  sediment
    2.85,   # 7:  dike
    2.90,   # 8:  dike
    2.95,   # 9:  dike
    2.65,   # 10: intrusion
    2.70,   # 11: intrusion
    2.75,   # 12: intrusion
    3.40,   # 13: ore
    3.80,   # 14: ore
]


class DifferentiableDensityMapper:
    """
    Maps continuous voxel embedding at a given time step (m_t) to density (g/cm³). Keeps gradients flowing (gravity misfit loss -> density -> m_t), so the optimizer can directly update voxel embeddings to minimize gravity misfit. 

        1. cosine similarity (after L2 normalization) of each voxel @ given t against each category
        2. temperature softmax over categories  →  probs  (high T = confident its a specific category, low T = uncertain)
        3. weighted sum. (eg. density of a voxel = p(bedrock)*0.99+p(sediment)*0.1 = 2.58)

    Args:
        embedding_weights:  (K, E) category centroid vectors from the trained model. e.g. model.embedding.weight
        density_values:     (K,) density per category in g/cm³. defaults to DENSITY_TABLE_15
        temperature:        softmax sharpness. high T → near one-hot (confident). low T → uniform (uncertain). default 10
    """

    def __init__(
        self,
        embedding_weights: torch.Tensor,        # (K, E)  category centroids, e.g. model.embedding.weight
        density_values: Optional[List[float]] = None,
        temperature: float = 10.0,              # higher → sharper probabilities
    ):
        self.embedding_weights = embedding_weights.detach().clone()  # detach: grads flow into m_t, not the table
        self.temperature = temperature
        densities = density_values if density_values is not None else DENSITY_TABLE_15
        self.density_values = torch.tensor(densities, dtype=torch.float32)  # (K,)

    def to(self, device):
        self.embedding_weights = self.embedding_weights.to(device)
        self.density_values    = self.density_values.to(device)
        return self

    def soft_decode_to_probabilities(self, m_t: torch.Tensor) -> torch.Tensor:
        m_norm   = F.normalize(m_t,                    dim=1)  # unit vectors in embedding space
        emb_norm = F.normalize(self.embedding_weights, dim=1)  # unit category centroids
        logits   = torch.einsum("bexyz, ke -> bkxyz", m_norm, emb_norm)  # cosine sim per voxel per category
        return F.softmax(logits * self.temperature, dim=1)

    def soft_decode_to_density(self, m_t: torch.Tensor) -> torch.Tensor:
        probs   = self.soft_decode_to_probabilities(m_t)                       # (B, K, X, Y, Z)
        density = torch.einsum("bkxyz, k -> bxyz", probs, self.density_values) # weighted density sum
        return density.unsqueeze(1)
