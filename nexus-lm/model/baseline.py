import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from model.surface import RMSNorm, SwiGLU


@dataclass
class LLaMAConfig:
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ffn: int = 1408
    vocab_size: int = 8192
    max_seq_len: int = 512
    rope_theta: float = 10000.0


def _make_rope_freqs(
    d_head: int, seq_len: int, theta: float, device: torch.device
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex tensor


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (B, T, n_heads, d_head) or (B, T, n_kv_heads, d_head)
    # freqs: (T, d_head//2) complex
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # x_: (B, T, n_heads, d_head//2); freqs unsqueezed: (1, T, 1, d_head//2)
    x_rot = x_ * freqs.unsqueeze(0).unsqueeze(2)
    return torch.view_as_real(x_rot).flatten(-2).to(x.dtype)


class LLaMALayer(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups = config.n_heads // config.n_kv_heads
        self.d_head = config.d_model // config.n_heads

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        self.q = nn.Linear(config.d_model, config.n_heads * self.d_head, bias=False)
        self.k = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=False)
        self.v = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=False)
        self.o = nn.Linear(config.d_model, config.d_model, bias=False)
        self.ffn = SwiGLU(config.d_model, config.d_ffn)
        self.scale = self.d_head ** -0.5

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x_norm = self.norm1(x)

        q = self.q(x_norm).view(B, T, self.n_heads, self.d_head)
        k = self.k(x_norm).view(B, T, self.n_kv_heads, self.d_head)
        v = self.v(x_norm).view(B, T, self.n_kv_heads, self.d_head)

        q = _apply_rope(q, freqs[:T])
        k = _apply_rope(k, freqs[:T])

        q = q.transpose(1, 2)                                           # (B, n_heads, T, d_head)
        k = k.transpose(1, 2).repeat_interleave(self.n_groups, dim=1)  # (B, n_heads, T, d_head)
        v = v.transpose(1, 2).repeat_interleave(self.n_groups, dim=1)

        causal = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device, dtype=x.dtype), diagonal=1
        )
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + causal
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        out = out.transpose(1, 2).reshape(B, T, D)

        x = residual + self.o(out)
        x = x + self.ffn(self.norm2(x))
        return x


class LLaMABaseline(nn.Module):
    """
    LLaMA-style baseline model for comparison with NEXUS-AURORA.
    Architecture: GQA + RoPE + SwiGLU + RMSNorm + weight-tied embedding.
    Comparable parameter count to NexusAurora (~50M at full scale).
    """

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([LLaMALayer(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        d_head = config.d_model // config.n_heads
        self.register_buffer(
            'rope_freqs',
            _make_rope_freqs(d_head, config.max_seq_len, config.rope_theta, torch.device('cpu'))
        )

    def forward(
        self,
        ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        x = self.embedding(ids)
        freqs = self.rope_freqs.to(ids.device)
        for layer in self.layers:
            x = layer(x, freqs)
        x = self.norm(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss
