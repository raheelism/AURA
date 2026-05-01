import torch
import torch.nn as nn
from typing import Optional


class HaltingEstimator(nn.Module):
    """
    Lightweight halting probability estimator for tokens.

    Given surface states `s` (B, T, d_s) and pooled reasoning `r_pooled` (B, 1, d_r)
    emits per-token halting probabilities in (0,1).
    Also provides a utility to map a distribution of p_t to a GPU-friendly k_batch.
    """

    def __init__(self, d_s: int, d_r: int, hidden: int = 128, tau: float = 1.0, k_max: int = 4, quantile: float = 0.9):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_s + d_r, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.tau = tau
        self.k_max = k_max
        self.quantile = quantile

    def forward(self, s: torch.Tensor, r_pooled: torch.Tensor) -> torch.Tensor:
        """Return halting probabilities p_t for each token.

        Args:
            s: (B, T, d_s)
            r_pooled: (B, 1, d_r) or (B, d_r)
        Returns:
            p: (B, T, 1) halting probabilities
        """
        B, T, d_s = s.shape
        if r_pooled.dim() == 2:
            r_pooled = r_pooled.unsqueeze(1)
        if r_pooled.size(1) == 1:
            r_exp = r_pooled.expand(-1, T, -1)
        elif r_pooled.size(1) == T:
            r_exp = r_pooled
        else:
            raise ValueError(
                f"r_pooled sequence dim must be 1 or {T}, got {r_pooled.size(1)}"
            )
        x = torch.cat([s, r_exp], dim=-1)
        logits = self.mlp(x)  # (B, T, 1)
        p = torch.sigmoid(logits / max(self.tau, 1e-8))
        return p

    def compute_k_batch(self, p: torch.Tensor, quantile: Optional[float] = None) -> int:
        """Compute k_batch (int) from per-token probs using quantile.

        p: (B, T, 1)
        Returns: scalar int k_batch in [1, k_max]
        """
        if quantile is None:
            quantile = self.quantile
        flat = p.view(-1).detach().cpu().float()  # Ensure float dtype
        if flat.numel() == 0:
            return 1
        q = float(torch.quantile(flat, quantile).item())
        # Map quantile score [0,1] -> additional iterations [0, k_max-1]
        extra = int(torch.ceil(torch.tensor(q * (self.k_max - 1))).item())
        k = 1 + max(0, min(self.k_max - 1, extra))
        return k
