import torch
import torch.nn as nn
from model.surface import RMSNorm, SwiGLU


class ReasoningStream(nn.Module):
    """
    Private 32-slot reasoning workspace.
    - NOT autoregressive — slots attend to each other without causal masking
    - NEVER produces output tokens (exists only to improve Surface stream quality)
    - Has learned initial slot parameters (concept primitives)
    - Updated K times per block, where K is determined by DifficultyEstimator
    """

    def __init__(self, d_r: int = 256, n_slots: int = 32, n_heads: int = 4, d_ffn: int = 512):
        super().__init__()
        assert d_r % n_heads == 0
        self.d_r = d_r
        self.n_slots = n_slots
        self.n_heads = n_heads
        self.d_head = d_r // n_heads

        # Learned initial slot representations — expanded to batch in NexusAurora.forward()
        self.slots = nn.Parameter(torch.randn(n_slots, d_r) * 0.02)

        self.norm1 = RMSNorm(d_r)
        self.norm2 = RMSNorm(d_r)
        self.q_proj = nn.Linear(d_r, d_r, bias=False)
        self.k_proj = nn.Linear(d_r, d_r, bias=False)
        self.v_proj = nn.Linear(d_r, d_r, bias=False)
        self.out_proj = nn.Linear(d_r, d_r, bias=False)
        self.ffn = SwiGLU(d_r, d_ffn)
        self.scale = self.d_head ** -0.5

    def _self_attn(self, r: torch.Tensor) -> torch.Tensor:
        """Non-causal self-attention — all slots can attend to all other slots."""
        B, N, D = r.shape
        q = self.q_proj(r).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(r).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(r).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # No causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(attended)

    def forward(self, r: torch.Tensor, n_iter: int = 1) -> torch.Tensor:
        """
        Args:
            r: reasoning slot tensor (B, n_slots, d_r)
            n_iter: number of internal update iterations (K from DifficultyEstimator)
        Returns:
            r: updated reasoning slots (B, n_slots, d_r)
        """
        for _ in range(n_iter):
            r = r + self._self_attn(self.norm1(r))
            r = r + self.ffn(self.norm2(r))
        return r

    def forward_masked(self, r: torch.Tensor, n_iter: int = 1, slot_mask: torch.Tensor = None) -> torch.Tensor:
        """Reasoning update with optional slot_mask (B, N, 1). Inactive slots are preserved."""
        if slot_mask is None:
            return self.forward(r, n_iter=n_iter)

        for _ in range(n_iter):
            updated = r + self._self_attn(self.norm1(r))
            updated = updated + self.ffn(self.norm2(updated))
            r = slot_mask * updated + (1.0 - slot_mask) * r
        return r
