import torch
import pytest
from model.bridge import BridgeLayer


def test_bridge_output_shapes():
    B, T, d_s, n_slots, d_r = 2, 16, 512, 32, 256
    bridge = BridgeLayer(d_s=d_s, d_r=d_r, n_heads=4, top_k=8)
    s = torch.randn(B, T, d_s)
    r = torch.randn(B, n_slots, d_r)
    gate_v = torch.ones(B, T, 1)
    s_out, r_out = bridge(s, r, gate_v)
    assert s_out.shape == (B, T, d_s)
    assert r_out.shape == (B, n_slots, d_r)


def test_bridge_s_reads_r_dense():
    """Every S position should receive signal from all R slots."""
    B, T, d_s, n_slots, d_r = 1, 8, 64, 8, 32
    bridge = BridgeLayer(d_s=d_s, d_r=d_r, n_heads=2, top_k=4)
    bridge.eval()
    s = torch.zeros(B, T, d_s)
    r1 = torch.randn(B, n_slots, d_r)
    r2 = torch.randn(B, n_slots, d_r)
    gate_v = torch.ones(B, T, 1)
    s_out1, _ = bridge(s, r1, gate_v)
    s_out2, _ = bridge(s, r2, gate_v)
    assert not torch.allclose(s_out1, s_out2), \
        "S output should differ when R is different"


def test_bridge_r_reads_s_causally():
    """S output at positions < 12 should be unaffected by change at position 12."""
    B, T, d_s, n_slots, d_r = 1, 16, 64, 8, 32
    bridge = BridgeLayer(d_s=d_s, d_r=d_r, n_heads=2, top_k=4)
    bridge.eval()
    s1 = torch.randn(B, T, d_s)
    s2 = s1.clone()
    s2[0, 12, :] = torch.randn(d_s)
    r = torch.randn(B, n_slots, d_r)
    gate_v = torch.ones(B, T, 1)
    s_out1, _ = bridge(s1, r, gate_v)
    s_out2, _ = bridge(s2, r, gate_v)
    assert torch.allclose(s_out1[0, :12], s_out2[0, :12], atol=1e-5), \
        "Bridge should not leak future S information to past S positions"


def test_bridge_gate_v_zeros_writeback():
    """When gate_v=0, R write-back to S should be zero."""
    B, T, d_s, n_slots, d_r = 2, 8, 64, 8, 32
    bridge = BridgeLayer(d_s=d_s, d_r=d_r, n_heads=2, top_k=4)
    s = torch.randn(B, T, d_s)
    r = torch.randn(B, n_slots, d_r)
    gate_ones = torch.ones(B, T, 1)
    gate_zeros = torch.zeros(B, T, 1)
    s_out_ones, _ = bridge(s, r, gate_ones)
    s_out_zeros, _ = bridge(s, r, gate_zeros)
    assert not torch.allclose(s_out_ones, s_out_zeros, atol=1e-5)
