import torch
import pytest
from model.verify import VerifyLayer


def test_verify_output_shapes():
    B, T, d_s, n_slots, d_r = 2, 16, 512, 32, 256
    verify = VerifyLayer(d_s=d_s, d_r=d_r, d_v=64)
    s = torch.randn(B, T, d_s)
    r = torch.randn(B, n_slots, d_r)
    surprise, gate_v = verify(s, r)
    assert surprise.shape == (B, T, 1)
    assert gate_v.shape == (B, T, 1)


def test_verify_surprise_in_zero_one():
    B, T, d_s, n_slots, d_r = 2, 16, 512, 32, 256
    verify = VerifyLayer(d_s=d_s, d_r=d_r, d_v=64)
    s = torch.randn(B, T, d_s)
    r = torch.randn(B, n_slots, d_r)
    surprise, _ = verify(s, r)
    assert (surprise >= 0).all() and (surprise <= 1).all()


def test_verify_gate_is_one_minus_surprise():
    B, T, d_s, n_slots, d_r = 2, 8, 64, 8, 32
    verify = VerifyLayer(d_s=d_s, d_r=d_r, d_v=16)
    s = torch.randn(B, T, d_s)
    r = torch.randn(B, n_slots, d_r)
    surprise, gate_v = verify(s, r)
    assert torch.allclose(gate_v, 1.0 - surprise, atol=1e-6)


def test_verify_gradients_flow():
    B, T, d_s, n_slots, d_r = 2, 8, 64, 8, 32
    verify = VerifyLayer(d_s=d_s, d_r=d_r, d_v=16)
    s = torch.randn(B, T, d_s, requires_grad=True)
    r = torch.randn(B, n_slots, d_r)
    surprise, gate_v = verify(s, r)
    surprise.mean().backward()
    assert s.grad is not None
    for p in verify.parameters():
        assert p.grad is not None
