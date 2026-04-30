import torch
import torch.nn as nn


class SlotAllocator(nn.Module):
    """
    Minimal dynamic slot allocator.

    Computes a context-conditioned score for each reasoning slot and keeps the
    top-k slots active. Inactive slots are passed through unchanged, so this is
    a safe sparse gating mechanism rather than a destructive rewrite.
    """

    def __init__(self, d_s: int, d_r: int, hidden: int = 128, min_slots: int = 4, max_slots: int = 32):
        super().__init__()
        self.context_proj = nn.Linear(d_s, d_r, bias=False)
        self.slot_proj = nn.Linear(d_r, d_r, bias=False)
        self.score_mlp = nn.Sequential(
            nn.Linear(d_r, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.min_slots = min_slots
        self.max_slots = max_slots

    def forward(self, s: torch.Tensor, r: torch.Tensor):
        """Return an (B, N, 1) mask in [0,1]."""
        B, T, d_s = s.shape
        N = r.shape[1]
        context = s.mean(dim=1)  # (B, d_s)
        ctx = self.context_proj(context).unsqueeze(1)  # (B, 1, d_r)
        slot_feat = self.slot_proj(r) + ctx
        scores = self.score_mlp(torch.tanh(slot_feat)).squeeze(-1)  # (B, N)

        k = max(self.min_slots, min(self.max_slots, N))
        k = min(k, N)
        topk_idx = torch.topk(scores, k=k, dim=-1).indices
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)
        return mask.unsqueeze(-1)
