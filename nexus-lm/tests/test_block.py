import torch
import pytest
from model.block import AuroraBlock
from model.cope import CoPE


def make_block(d_s=64, d_r=32, d_v=16, n_slots=8):
    """Create a small AuroraBlock for testing."""
    config = {
        'd_surface': d_s,
        'd_reasoning': d_r,
        'd_verify': d_v,
        'n_reasoning_slots': n_slots,
        'n_heads_surface': 4,
        'n_kv_heads_surface': 2,
        'n_heads_reasoning': 2,
        'd_ffn_pattern': d_s,
        'd_ffn_semantic': d_s * 4,
        'd_ffn_reasoning': d_r * 2,
        'local_window': 8,
        'bridge_top_k': 4,
        'surprise_loss_weight': 0.1,
        'difficulty_entropy_weight': 0.01,
    }
    return AuroraBlock(config)


def test_aurora_block_output_shapes():
    B, T = 2, 16
    d_s, d_r, n_slots = 64, 32, 8
    block = make_block(d_s, d_r, n_slots=n_slots)
    cope = CoPE(d_s)
    s = torch.randn(B, T, d_s)
    r = torch.randn(B, n_slots, d_r)
    gate_v = torch.ones(B, T, 1)
    s_out, r_out, gate_v_out, surprise_loss = block(s, r, gate_v, cope)
    assert s_out.shape == (B, T, d_s)
    assert r_out.shape == (B, n_slots, d_r)
    assert gate_v_out.shape == (B, T, 1)
    assert surprise_loss.shape == ()  # scalar


def test_aurora_block_surprise_loss_positive():
    B, T = 2, 16
    block = make_block()
    cope = CoPE(64)
    s = torch.randn(B, T, 64)
    r = torch.randn(B, 8, 32)
    gate_v = torch.ones(B, T, 1)
    _, _, _, surprise_loss = block(s, r, gate_v, cope)
    assert surprise_loss.item() >= 0


def test_aurora_block_gradients_flow():
    B, T = 2, 8
    block = make_block()
    cope = CoPE(64)
    s = torch.randn(B, T, 64, requires_grad=True)
    r = torch.randn(B, 8, 32)
    gate_v = torch.ones(B, T, 1)
    s_out, r_out, gate_v_out, surprise_loss = block(s, r, gate_v, cope)
    loss = s_out.sum() + surprise_loss
    loss.backward()
    assert s.grad is not None


def test_aurora_block_causal_property():
    """Changing S at position 8 should not affect S output at positions < 8."""
    B, T = 1, 16
    block = make_block()
    block.eval()
    cope = CoPE(64)
    s1 = torch.randn(B, T, 64)
    s2 = s1.clone()
    s2[0, 8, :] = torch.randn(64)
    r = torch.randn(B, 8, 32)
    gate_v = torch.ones(B, T, 1)
    s_out1, _, _, _ = block(s1, r, gate_v, cope)
    s_out2, _, _, _ = block(s2, r, gate_v, cope)
    assert torch.allclose(s_out1[0, :8], s_out2[0, :8], atol=1e-4), \
        "AuroraBlock must be causally correct"
