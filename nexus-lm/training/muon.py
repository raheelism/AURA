import torch
from torch.optim import Optimizer
from typing import List


def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute approximate orthogonal factor of G.
    For matrix G (m x n, m >= n): converges to U where G ≈ U * Sigma * V^T.
    This is the key operation in Muon that keeps weight updates near orthogonal,
    preserving update diversity across the training run.
    """
    assert G.dim() == 2, f"Expected 2D tensor, got shape {G.shape}"
    m, n = G.shape
    X = G.float() / (G.norm() + 1e-7)
    # Work with the wider dimension on the right: transpose if tall
    transposed = m > n
    if transposed:
        X = X.T  # now X has shape (min_dim, max_dim)

    # NS5 iteration: X_{k+1} = a*X + (b*A + c*A^2) @ X
    # Computes the polar factor (orthogonal part) of the matrix.
    # Coefficients (15/8, -5/4, 3/8) give the 5th-degree minimax polynomial
    # for the sign function; converges in ~5 steps when singular values ≤ 1.
    a, b, c = 15 / 8, -5 / 4, 3 / 8
    for _ in range(steps):
        A = X @ X.T              # (n, n) — smaller Gram matrix
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer: applies Newton-Schulz orthogonalization to gradient matrices.

    Only operates on 2D matrix parameters (Linear weight matrices).
    For embeddings and 1D parameters, use AdamW instead.

    Key insight: orthogonalizing the update ensures each weight matrix update
    covers all directions equally, preventing collapse or redundancy in learned
    representations. This is especially valuable for deep transformer weight matrices.

    Reference: https://github.com/KellerJordan/Muon
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                assert p.dim() == 2, \
                    f"Muon requires 2D parameters, got shape {p.shape}"

                g = p.grad.float()

                # Momentum buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Orthogonalize gradient
                g_orth = newton_schulz_orthogonalize(g, steps=ns_steps)

                # Scale orthogonalized gradient to match original gradient RMS
                rms_original = g.norm() / (g.numel() ** 0.5 + 1e-7)
                rms_orth = g_orth.norm() / (g_orth.numel() ** 0.5 + 1e-7)
                g_orth = g_orth * (rms_original / (rms_orth + 1e-7))

                p.add_(g_orth.to(p.dtype), alpha=-lr)
