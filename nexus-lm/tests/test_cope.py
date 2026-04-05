import torch
import pytest
from model.cope import CoPE


def test_cope_output_shape():
    B, T, D = 2, 16, 512
    cope = CoPE(d_model=D, n_positions=16)
    q = torch.randn(B, T, D)
    x = torch.randn(B, T, D)
    pos = cope(q, x)
    assert pos.shape == (B, T, D), f"Expected ({B},{T},{D}), got {pos.shape}"


def test_cope_output_varies_with_query():
    """Different queries should produce different positional signals."""
    B, T, D = 2, 8, 64
    cope = CoPE(d_model=D, n_positions=8)
    q1 = torch.randn(B, T, D)
    q2 = torch.randn(B, T, D)
    x = torch.zeros(B, T, D)
    pos1 = cope(q1, x)
    pos2 = cope(q2, x)
    assert not torch.allclose(pos1, pos2), "CoPE output should vary with query"


def test_cope_gradients_flow():
    B, T, D = 2, 8, 64
    cope = CoPE(d_model=D, n_positions=8)
    q = torch.randn(B, T, D, requires_grad=True)
    x = torch.randn(B, T, D)
    pos = cope(q, x)
    loss = pos.sum()
    loss.backward()
    assert q.grad is not None
    assert cope.pos_emb.weight.grad is not None
