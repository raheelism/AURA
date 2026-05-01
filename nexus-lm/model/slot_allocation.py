import torch
import torch.nn as nn


class SlotAllocator(nn.Module):
    """
    Context-conditioned slot allocator with straight-through top-k masking.

    Forward values are hard 0/1 masks, preserving the old sparse behavior.
    Backward gradients flow through a soft top-k relaxation so the allocator
    can actually learn.
    """

    def __init__(
        self,
        d_s: int,
        d_r: int,
        hidden: int = 128,
        min_slots: int = 4,
        max_slots: int = 32,
        tau: float = 1.0,
    ):
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
        self.tau = tau

    def _prefix_mean(self, s: torch.Tensor) -> torch.Tensor:
        T = s.shape[1]
        denom = torch.arange(1, T + 1, device=s.device, dtype=s.dtype).view(1, T, 1)
        return s.cumsum(dim=1) / denom

    def forward(self, s: torch.Tensor, r: torch.Tensor):
        """
        Return a hard-in-forward, soft-in-backward mask.

        Args:
            s: (B, T, d_s)
            r: (B, N, d_r) or (B, T, N, d_r)
        Returns:
            mask: (B, N, 1) or (B, T, N, 1), matching r rank
        """
        N = r.shape[-2]
        k = max(self.min_slots, min(self.max_slots, N))
        k = min(k, N)

        if r.dim() == 3:
            context = s.mean(dim=1)
            ctx = self.context_proj(context).unsqueeze(1)
            slot_feat = self.slot_proj(r) + ctx
        elif r.dim() == 4:
            context = self._prefix_mean(s)
            ctx = self.context_proj(context).unsqueeze(2)
            slot_feat = self.slot_proj(r) + ctx
        else:
            raise ValueError(f"Expected r rank 3 or 4, got shape {tuple(r.shape)}")

        scores = self.score_mlp(torch.tanh(slot_feat)).squeeze(-1)
        topk_idx = torch.topk(scores, k=k, dim=-1).indices
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(-1, topk_idx, 1.0)

        soft_mask = torch.softmax(scores / max(self.tau, 1e-6), dim=-1) * float(k)
        soft_mask = soft_mask.clamp(0.0, 1.0)
        mask = hard_mask + soft_mask - soft_mask.detach()
        return mask.unsqueeze(-1)
