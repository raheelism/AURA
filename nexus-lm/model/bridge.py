import torch
import torch.nn as nn
from model.surface import RMSNorm
from typing import Tuple


class BridgeLayer(nn.Module):
    """
    Bidirectional communication between Surface stream (S) and Reasoning stream (R).

    The full language model passes prefix-local reasoning states with shape
    (B, T, N, d_r). In that mode, each token position only reads its own private
    slots, and each private slot attends only to surface positions in its prefix.
    This preserves autoregressive causality across multiple blocks.
    """

    def __init__(self, d_s: int, d_r: int, n_heads: int = 4, top_k: int = 8):
        super().__init__()
        assert d_s % n_heads == 0
        assert d_r % n_heads == 0
        self.n_heads = n_heads
        self.d_head_s = d_s // n_heads
        self.d_head_r = d_r // n_heads
        self.top_k = top_k
        self.d_s = d_s
        self.d_r = d_r

        self.norm_s = RMSNorm(d_s)
        self.norm_r = RMSNorm(d_r)

        self.sq = nn.Linear(d_s, d_s, bias=False)
        self.sk = nn.Linear(d_r, d_s, bias=False)
        self.sv = nn.Linear(d_r, d_s, bias=False)
        self.so = nn.Linear(d_s, d_s, bias=False)

        self.rq = nn.Linear(d_r, d_r, bias=False)
        self.rk = nn.Linear(d_s, d_r, bias=False)
        self.rv = nn.Linear(d_s, d_r, bias=False)
        self.ro = nn.Linear(d_r, d_r, bias=False)

        self.scale_s = self.d_head_s ** -0.5
        self.scale_r = self.d_head_r ** -0.5
        self.low_rank = None
        self.U_r = None
        self.W_s = None

    def enable_low_rank_writeback(self, rank: int):
        assert rank > 0 and isinstance(rank, int)
        self.low_rank = rank
        self.U_r = nn.Parameter(torch.randn(self.d_r, rank) * 0.02)
        self.W_s = nn.Parameter(torch.randn(rank, self.d_s) * 0.02)

    def _s_reads_r(self, s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
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
        B, N, D_r = r.shape
        T = s.shape[1]
        q = self.rq(r).view(B, N, self.n_heads, self.d_head_r).transpose(1, 2)
        k = self.rk(s).view(B, T, self.n_heads, self.d_head_r).transpose(1, 2)
        v = self.rv(s).view(B, T, self.n_heads, self.d_head_r).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale_r
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(B, N, D_r)
        return self.ro(out)

    def _sparse_r_to_s(self, s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        B, T, D_s = s.shape
        N = r.shape[1]
        q = self.sq(s)
        k = self.sk(r)
        v = self.sv(r)
        if self.low_rank is not None and self.U_r is not None and self.W_s is not None:
            v = torch.matmul(torch.matmul(r, self.U_r), self.W_s)
        scores = torch.bmm(q, k.transpose(-2, -1)) * self.scale_s
        topk_vals, topk_idx = torch.topk(scores, k=min(self.top_k, N), dim=-1)
        sparse_scores = torch.full_like(scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_idx, topk_vals)
        weights = torch.softmax(sparse_scores, dim=-1)
        return torch.bmm(weights, v)

    def _s_reads_prefix_r(self, s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        B, T, D_s = s.shape
        N = r.shape[2]
        q = self.sq(s).view(B, T, self.n_heads, self.d_head_s)
        k = self.sk(r).view(B, T, N, self.n_heads, self.d_head_s)
        v = self.sv(r).view(B, T, N, self.n_heads, self.d_head_s)
        scores = torch.einsum('bthd,btnhd->bthn', q, k) * self.scale_s
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum('bthn,btnhd->bthd', weights, v).reshape(B, T, D_s)
        return self.so(out)

    def _r_reads_s_causal(self, r: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        B, T, N, D_r = r.shape
        q = self.rq(r).view(B, T, N, self.n_heads, self.d_head_r)
        k = self.rk(s).view(B, T, self.n_heads, self.d_head_r)
        v = self.rv(s).view(B, T, self.n_heads, self.d_head_r)
        scores = torch.einsum('btnhd,bphd->btnhp', q, k) * self.scale_r

        rows = torch.arange(T, device=s.device).view(T, 1)
        cols = torch.arange(T, device=s.device).view(1, T)
        causal = cols <= rows
        scores = scores.masked_fill(~causal.view(1, T, 1, 1, T), float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum('btnhp,bphd->btnhd', weights, v).reshape(B, T, N, D_r)
        return self.ro(out)

    def _sparse_prefix_r_to_s(self, s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        B, T, D_s = s.shape
        N = r.shape[2]
        q = self.sq(s)
        k = self.sk(r)
        v = self.sv(r)
        if self.low_rank is not None and self.U_r is not None and self.W_s is not None:
            v = torch.matmul(torch.matmul(r, self.U_r), self.W_s)

        scores = torch.einsum('btd,btnd->btn', q, k) * self.scale_s
        topk_vals, topk_idx = torch.topk(scores, k=min(self.top_k, N), dim=-1)
        sparse_scores = torch.full_like(scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_idx, topk_vals)
        weights = torch.softmax(sparse_scores, dim=-1)
        return torch.einsum('btn,btnd->btd', weights, v)

    def forward(
        self,
        s: torch.Tensor,
        r: torch.Tensor,
        gate_v: torch.Tensor,
        halting_probs: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_norm = self.norm_s(s)
        r_norm = self.norm_r(r)

        if r.dim() == 3:
            s = s + self._s_reads_r(s_norm, r_norm)
            r = r + self._r_reads_s(r_norm, s_norm)
            r_sparse = self._sparse_r_to_s(s_norm, r_norm)
        elif r.dim() == 4:
            s = s + self._s_reads_prefix_r(s_norm, r_norm)
            r = r + self._r_reads_s_causal(r_norm, s_norm)
            r_sparse = self._sparse_prefix_r_to_s(s_norm, r_norm)
        else:
            raise ValueError(f"Expected r rank 3 or 4, got shape {tuple(r.shape)}")

        if halting_probs is not None:
            effective_gate = gate_v * (1.0 - halting_probs)
        else:
            effective_gate = gate_v
        s = s + effective_gate * r_sparse
        return s, r
