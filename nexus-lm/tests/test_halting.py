import torch
import pytest
from model.halting import HaltingEstimator


def test_halting_estimator_shapes_and_k():
    B, T, d_s, d_r = 2, 16, 64, 32
    s = torch.randn(B, T, d_s)
    r_pooled = torch.randn(B, d_r)
    h = HaltingEstimator(d_s=d_s, d_r=d_r, hidden=32, tau=1.0, k_max=4, quantile=0.9)
    p = h(s, r_pooled)
    assert p.shape == (B, T, 1)
    k = h.compute_k_batch(p)
    assert isinstance(k, int)
    assert 1 <= k <= 4


def test_halting_decreases_k_when_probs_low():
    B, T, d_s, d_r = 2, 16, 64, 32
    s = torch.randn(B, T, d_s)
    r_pooled = torch.randn(B, d_r)
    h = HaltingEstimator(d_s=d_s, d_r=d_r, hidden=32, tau=1.0, k_max=4, quantile=0.9)
    # force very low probs
    p_low = torch.zeros(B, T, 1)
    k_low = h.compute_k_batch(p_low)
    assert k_low == 1
