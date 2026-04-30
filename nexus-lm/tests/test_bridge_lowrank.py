import torch
from model.bridge import BridgeLayer


def test_bridge_lowrank_writeback_shapes():
    B, T, d_s, d_r, N = 2, 16, 64, 32, 8
    s = torch.randn(B, T, d_s)
    r = torch.randn(B, N, d_r)
    gate_v = torch.ones(B, T, 1)
    bridge = BridgeLayer(d_s=d_s, d_r=d_r, n_heads=4, top_k=4)
    bridge.enable_low_rank_writeback(rank=16)
    s2, r2 = bridge(s, r, gate_v)
    assert s2.shape == (B, T, d_s)
    assert r2.shape == (B, N, d_r)
