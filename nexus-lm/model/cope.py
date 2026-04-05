import torch
import torch.nn as nn
import torch.nn.functional as F


class CoPE(nn.Module):
    """
    Contextual Position Encoding.
    Position is computed as a weighted sum of learned position vectors,
    where weights are conditioned on the query (context-dependent position).
    Position vectors are L2-normalized to live on the unit hypersphere.
    """

    def __init__(self, d_model: int, n_positions: int = 16):
        super().__init__()
        self.n_positions = n_positions
        self.pos_emb = nn.Embedding(n_positions, d_model)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: query tensor (B, T, D)
            x: input tensor (B, T, D) — unused, kept for interface consistency
        Returns:
            pos: contextual position signal (B, T, D)
        """
        pos_vecs = F.normalize(self.pos_emb.weight, dim=-1)          # (n_pos, D)
        gates = torch.softmax(
            q @ pos_vecs.T / (q.shape[-1] ** 0.5), dim=-1
        )                                                              # (B, T, n_pos)
        return gates @ pos_vecs                                        # (B, T, D)
