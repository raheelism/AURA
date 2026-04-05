import torch
import pytest
from model.reasoning import ReasoningStream


def test_reasoning_stream_output_shape():
    B, n_slots, d_r = 3, 32, 256
    stream = ReasoningStream(d_r=d_r, n_slots=n_slots, n_heads=4, d_ffn=512)
    r = torch.randn(B, n_slots, d_r)
    out = stream(r, n_iter=1)
    assert out.shape == (B, n_slots, d_r)


def test_reasoning_stream_multiple_iterations():
    B, n_slots, d_r = 2, 32, 256
    stream = ReasoningStream(d_r=d_r, n_slots=n_slots, n_heads=4, d_ffn=512)
    r = torch.randn(B, n_slots, d_r)
    out1 = stream(r, n_iter=1)
    out4 = stream(r, n_iter=4)
    assert not torch.allclose(out1, out4), "n_iter=1 and n_iter=4 should differ"


def test_reasoning_not_causal():
    """Changing slot j should affect slot i (bidirectional, non-causal)."""
    B, n_slots, d_r = 1, 8, 64
    stream = ReasoningStream(d_r=d_r, n_slots=n_slots, n_heads=2, d_ffn=64)
    stream.eval()
    r1 = torch.randn(B, n_slots, d_r)
    r2 = r1.clone()
    r2[0, 7, :] = torch.randn(d_r)
    out1 = stream(r1, n_iter=1)
    out2 = stream(r2, n_iter=1)
    assert not torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5), \
        "Reasoning stream should be non-causal"


def test_reasoning_stream_slot_init():
    stream = ReasoningStream(d_r=256, n_slots=32, n_heads=4, d_ffn=512)
    assert hasattr(stream, 'slots')
    assert stream.slots.shape == (32, 256)


def test_reasoning_stream_gradients_flow():
    B, n_slots, d_r = 2, 16, 64
    stream = ReasoningStream(d_r=d_r, n_slots=n_slots, n_heads=2, d_ffn=64)
    r = torch.randn(B, n_slots, d_r, requires_grad=True)
    out = stream(r, n_iter=2)
    out.sum().backward()
    assert r.grad is not None
