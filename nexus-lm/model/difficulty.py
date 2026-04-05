import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# K values corresponding to logit indices 0, 1, 2
K_VALUES = [1, 2, 4]


class DifficultyEstimator(nn.Module):
    """
    Estimates how many reasoning iterations (K) each position needs.
    Outputs K ∈ {1, 2, 4} per position via Gumbel-softmax for gradient flow.
    At training time: k_batch = max(K_i) across sequence for GPU efficiency.
    """

    def __init__(self, d_s: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_s, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, 3, bias=True),
        )
        # Bias toward K=1 at init — model learns to think harder as needed
        self.net[-1].bias.data.fill_(0)
        self.net[-1].bias.data[0] = 2.0

    def forward(self, s: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Args:
            s: surface hidden states (B, T, d_s)
        Returns:
            k_batch: int — max K across all positions (used to run R iterations)
            logits:  (B, T, 3) — raw logits for entropy regularization loss
        """
        logits = self.net(s)  # (B, T, 3)

        if self.training:
            k_onehot = F.gumbel_softmax(logits, tau=1.0, hard=True)
            k_indices = k_onehot.argmax(dim=-1)
        else:
            k_indices = logits.argmax(dim=-1)

        k_tensor = torch.tensor(K_VALUES, device=logits.device)[k_indices]
        k_batch = int(k_tensor.max().item())
        return k_batch, logits
