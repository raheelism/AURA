import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cope import CoPE


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network: FFN(x) = (SiLU(W1*x) * W3*x) @ W2."""

    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _make_causal_local_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    """
    Boolean mask for causal local window attention.
    True = attend, False = block.
    Position i attends to positions max(0, i-window+1)..i only.
    """
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    causal = cols <= rows
    local = (rows - cols) < window
    return causal & local  # (T, T) bool


class PatternLayer(nn.Module):
    """
    Causal local window attention + narrow SwiGLU FFN.
    Captures syntax and local n-gram patterns.
    Each position attends to at most `window_size` preceding positions.
    """

    def __init__(self, d_model: int, d_ffn: int, window_size: int = 64):
        super().__init__()
        self.window_size = window_size
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = SwiGLU(d_model, d_ffn)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x_norm = self.norm1(x)
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        mask = _make_causal_local_mask(T, self.window_size, x.device)
        attn_bias = torch.zeros(T, T, device=x.device, dtype=x.dtype)
        attn_bias[~mask] = float('-inf')

        scores = torch.bmm(q, k.transpose(-2, -1)) * self.scale + attn_bias.unsqueeze(0)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.bmm(weights, v)
        x = residual + self.out_proj(attended)
        x = x + self.ffn(self.norm2(x))
        return x


class SemanticLayer(nn.Module):
    """
    Full causal GQA attention + wide SwiGLU FFN.
    Captures long-range semantic relationships.
    """

    def __init__(self, d_model: int, d_ffn: int, n_heads: int = 8, n_kv_heads: int = 2):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = SwiGLU(d_model, d_ffn)
        self.scale = self.d_head ** -0.5

    def forward(self, x: torch.Tensor, cope: CoPE) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # CoPE: add contextual position signal to queries
        q_flat = q.transpose(1, 2).reshape(B, T, -1)
        pos = cope(q_flat, x_norm)
        pos_heads = pos.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = q + pos_heads

        # Expand KV for GQA
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + causal_mask
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).reshape(B, T, D)
        x = residual + self.out_proj(attended)
        x = x + self.ffn(self.norm2(x))
        return x
