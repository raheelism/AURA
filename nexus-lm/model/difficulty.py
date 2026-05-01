import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple


# Default K values corresponding to logit indices 0, 1, 2.
K_VALUES = [1, 2, 4]


class DifficultyEstimator(nn.Module):
    """
    Estimates how many reasoning iterations each position needs.

    The returned hard weights use the straight-through Gumbel-Softmax estimator:
    the forward pass selects one K per token, while gradients flow through the
    soft relaxation when the weights are used downstream.
    """

    def __init__(
        self,
        d_s: int,
        hidden: int = 128,
        k_values: Sequence[int] = K_VALUES,
        tau: float = 1.0,
    ):
        super().__init__()
        self.k_values = tuple(int(k) for k in k_values)
        self.tau = tau
        self.net = nn.Sequential(
            nn.Linear(d_s, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, len(self.k_values), bias=True),
        )
        # Bias toward K=1 at init; the model learns to think harder as needed.
        self.net[-1].bias.data.fill_(0)
        self.net[-1].bias.data[0] = 2.0

    def forward(
        self,
        s: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[int, torch.Tensor]:
        """
        Args:
            s: surface hidden states (B, T, d_s)
            return_weights: include straight-through one-hot routing weights
        Returns:
            k_batch: max K across all positions
            logits:  raw routing logits (B, T, len(k_values))
            weights: optional straight-through routing weights
        """
        logits = self.net(s)

        if self.training:
            k_onehot = F.gumbel_softmax(logits, tau=self.tau, hard=True)
            k_indices = k_onehot.argmax(dim=-1)
        else:
            k_indices = logits.argmax(dim=-1)
            k_onehot = F.one_hot(k_indices, num_classes=len(self.k_values)).to(logits.dtype)

        k_tensor = torch.tensor(self.k_values, device=logits.device)[k_indices]
        k_batch = int(k_tensor.max().item())
        if return_weights:
            return k_batch, logits, k_onehot
        return k_batch, logits
