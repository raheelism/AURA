import torch
import torch.nn as nn
import torch.nn.functional as F
from model.surface import RMSNorm
from typing import Tuple


class BridgeLayer(nn.Module):
    """
    Bidirectional cross-attention between Surface stream (S) and Reasoning stream (R).

    Three operations per forward pass:
    1. S reads R (dense):   each S position attends to all R slots
    2. R reads S (non-causal from R's perspective, but S is already causally computed):
       R slots attend to all S positions
    3. Sparse R → S write-back (top-k): S positions attend to top-k relevant R slots,
       gated by the Verification stream's surprise signal

    Causality guarantee: S is already causally computed by Pattern + Semantic layers
    upstream. No future token leaks through this layer.
    """

    def __init__(self, d_s: int, d_r: int, n_heads: int = 4, top_k: int = 8):
        super().__init__()
        assert d_s % n_heads == 0
        self.n_heads = n_heads
        self.d_head_s = d_s // n_heads
        self.top_k = top_k
        self.d_s = d_s
        self.d_r = d_r

        self.norm_s = RMSNorm(d_s)
        self.norm_r = RMSNorm(d_r)

        # S reads R: Q from S, K/V projected from R to S dim
        self.sq = nn.Linear(d_s, d_s, bias=False)
        self.sk = nn.Linear(d_r, d_s, bias=False)
        self.sv = nn.Linear(d_r, d_s, bias=False)
        self.so = nn.Linear(d_s, d_s, bias=False)

        # R reads S: Q from R, K/V projected from S to R dim
        self.rq = nn.Linear(d_r, d_r, bias=False)
        self.rk = nn.Linear(d_s, d_r, bias=False)
        self.rv = nn.Linear(d_s, d_r, bias=False)
        self.ro = nn.Linear(d_r, d_r, bias=False)

        self.scale_s = self.d_head_s ** -0.5
        self.scale_r = (d_r // n_heads) ** -0.5

    def _s_reads_r(self, s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Dense: each S position reads all R slots."""
        B, T, D_s = s.shape
        N = r.shape[1]
        q = self.sq(s).view(B, T, self.n_heads, self.d_head_s).transpose(1, 2)
        k = self.sk(r).view(B, N, self.n_heads, self.d_head_s).transpose(1, 2)
        v = self.sv(r).view(B, N, self.n_heads, self.d_head_s).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale_s
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(B, T, D_s)
        return self.so(out)

    def _r_reads_s(self, r: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """R slots read S positions. S is already causally correct upstream."""
        B, N, D_r = r.shape
        T = s.shape[1]
        d_head_r = D_r // self.n_heads
        q = self.rq(r).view(B, N, self.n_heads, d_head_r).transpose(1, 2)
        k = self.rk(s).view(B, T, self.n_heads, d_head_r).transpose(1, 2)
        v = self.rv(s).view(B, T, self.n_heads, d_head_r).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale_r
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(B, N, D_r)
        return self.ro(out)

    def _sparse_r_to_s(self, s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Sparse top-k: S positions attend to their most relevant R slots."""
        B, T, D_s = s.shape
        N = r.shape[1]
        q = self.sq(s)
        k = self.sk(r)
        v = self.sv(r)
        scores = torch.bmm(q, k.transpose(-2, -1)) * self.scale_s  # (B, T, N)
        topk_vals, topk_idx = torch.topk(scores, k=min(self.top_k, N), dim=-1)
        sparse_scores = torch.full_like(scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_idx, topk_vals)
        weights = torch.softmax(sparse_scores, dim=-1)
        return torch.bmm(weights, v)

    def forward(
        self,
        s: torch.Tensor,
        r: torch.Tensor,
        gate_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:      surface hidden states (B, T, d_s)
            r:      reasoning slots (B, n_slots, d_r)
            gate_v: verification gate (B, T, 1), range [0, 1]
        Returns:
            s: updated (B, T, d_s)
            r: updated (B, n_slots, d_r)
        """
        s_norm = self.norm_s(s)
        r_norm = self.norm_r(r)

        # 1. S reads R (surface absorbs all reasoning context)
        s = s + self._s_reads_r(s_norm, r_norm)

        # 2. R grounds itself in surface evidence
        r = r + self._r_reads_s(r_norm, s_norm)

        # 3. Sparse R → S write-back, gated by verification surprise
        r_sparse = self._sparse_r_to_s(s_norm, r_norm)
        s = s + gate_v * r_sparse

        return s, r
