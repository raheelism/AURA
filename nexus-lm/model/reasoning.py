import torch
import torch.nn as nn
from model.surface import RMSNorm, SwiGLU


class ReasoningStream(nn.Module):
    """
    Private slot reasoning workspace.

    The module can process either a batch of slot sets (B, N, D) or a
    prefix-local slot set per token position (B, T, N, D). The full language
    model uses the latter form to preserve autoregressive causality across
    multiple blocks.
    """

    def __init__(self, d_r: int = 256, n_slots: int = 32, n_heads: int = 4, d_ffn: int = 512):
        super().__init__()
        assert d_r % n_heads == 0
        self.d_r = d_r
        self.n_slots = n_slots
        self.n_heads = n_heads
        self.d_head = d_r // n_heads

        # Compatibility buffer for standalone uses. NexusAurora owns the
        # trainable initial slots so later blocks do not carry unused params.
        self.register_buffer('slots', torch.randn(n_slots, d_r) * 0.02, persistent=False)

        self.norm1 = RMSNorm(d_r)
        self.norm2 = RMSNorm(d_r)
        self.q_proj = nn.Linear(d_r, d_r, bias=False)
        self.k_proj = nn.Linear(d_r, d_r, bias=False)
        self.v_proj = nn.Linear(d_r, d_r, bias=False)
        self.out_proj = nn.Linear(d_r, d_r, bias=False)
        self.ffn = SwiGLU(d_r, d_ffn)
        self.scale = self.d_head ** -0.5

    def _self_attn(self, r: torch.Tensor) -> torch.Tensor:
        """Non-causal self-attention among slots only."""
        B, N, D = r.shape
        q = self.q_proj(r).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(r).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(r).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(attended)

    def _flatten_positions(self, r: torch.Tensor):
        if r.dim() != 4:
            return r, None
        original_shape = r.shape
        return r.reshape(-1, original_shape[-2], original_shape[-1]), original_shape

    def _restore_positions(self, r: torch.Tensor, original_shape):
        if original_shape is None:
            return r
        return r.reshape(original_shape)

    def forward(self, r: torch.Tensor, n_iter: int = 1) -> torch.Tensor:
        """
        Args:
            r: slot tensor (B, N, D) or prefix-local slots (B, T, N, D)
            n_iter: number of internal update iterations
        """
        r, original_shape = self._flatten_positions(r)
        for _ in range(n_iter):
            r = r + self._self_attn(self.norm1(r))
            r = r + self.ffn(self.norm2(r))
        return self._restore_positions(r, original_shape)

    def forward_masked(self, r: torch.Tensor, n_iter: int = 1, slot_mask: torch.Tensor = None) -> torch.Tensor:
        """Reasoning update with optional slot mask. Inactive slots are preserved."""
        if slot_mask is None:
            return self.forward(r, n_iter=n_iter)

        r, original_shape = self._flatten_positions(r)
        if original_shape is not None:
            slot_mask = slot_mask.reshape(-1, original_shape[-2], 1)

        for _ in range(n_iter):
            updated = r + self._self_attn(self.norm1(r))
            updated = updated + self.ffn(self.norm2(updated))
            r = slot_mask * updated + (1.0 - slot_mask) * r
        return self._restore_positions(r, original_shape)
