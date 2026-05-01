import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VerifyLayer(nn.Module):
    """
    Consistency verification stream.

    Computes a per-position 'surprise' score: how inconsistent is the Surface
    stream (S) with what the Reasoning stream (R) represents?

    High surprise → strong R-to-S write-back (R corrects S in Bridge Layer)
    Low surprise  → weak R-to-S write-back (S is already consistent with R)

    Auxiliary training loss: L_surprise = mean(surprise) — minimized over training.
    The model learns to make S and R mutually consistent.
    """

    def __init__(self, d_s: int, d_r: int, d_v: int = 64):
        super().__init__()
        self.d_r = d_r
        self.proj = nn.Linear(d_s + d_r, d_v, bias=True)
        self.out = nn.Linear(d_v, 1, bias=True)
        nn.init.constant_(self.out.bias, 0.1)

    def forward(
        self,
        s: torch.Tensor,
        r: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: surface hidden states (B, T, d_s)
            r: reasoning slots (B, n_slots, d_r)
        Returns:
            surprise: per-position surprise score (B, T, 1) in [0, 1]
            gate_v:   1 - surprise (B, T, 1), gates R→S write-back strength
        """
        B, T, _ = s.shape
        if r.dim() == 3:
            r_pooled = r.mean(dim=1).unsqueeze(1).expand(B, T, self.d_r)
        elif r.dim() == 4:
            r_pooled = r.mean(dim=2)
        else:
            raise ValueError(f"Expected r rank 3 or 4, got shape {tuple(r.shape)}")
        combined = torch.cat([s, r_pooled], dim=-1)
        h = F.gelu(self.proj(combined))
        surprise = torch.sigmoid(self.out(h))
        return surprise, 1.0 - surprise
