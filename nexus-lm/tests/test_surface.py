import torch
import pytest
from model.surface import PatternLayer, SemanticLayer, SwiGLU, RMSNorm
from model.cope import CoPE


def test_rmsnorm_output_shape():
    x = torch.randn(2, 16, 512)
    norm = RMSNorm(512)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_normalizes():
    x = torch.randn(2, 16, 512) * 100
    norm = RMSNorm(512)
    out = norm(x)
    rms = (out ** 2).mean(dim=-1).sqrt()
    assert rms.mean().item() == pytest.approx(1.0, abs=0.1)


def test_swiglu_output_shape():
    B, T, D, D_ffn = 2, 16, 512, 1024
    ffn = SwiGLU(D, D_ffn)
    x = torch.randn(B, T, D)
    out = ffn(x)
    assert out.shape == (B, T, D)


def test_pattern_layer_output_shape():
    B, T, D = 2, 64, 512
    layer = PatternLayer(d_model=D, d_ffn=512, window_size=32)
    x = torch.randn(B, T, D)
    out = layer(x)
    assert out.shape == (B, T, D)


def test_pattern_layer_causal_no_future_leak():
    """Changing token at position t should not affect positions < t."""
    B, T, D = 1, 16, 64
    layer = PatternLayer(d_model=D, d_ffn=64, window_size=8)
    layer.eval()
    x1 = torch.randn(B, T, D)
    x2 = x1.clone()
    x2[0, 8, :] = torch.randn(D)
    out1 = layer(x1)
    out2 = layer(x2)
    assert torch.allclose(out1[0, :8], out2[0, :8], atol=1e-5), \
        "Pattern Layer is leaking future information"


def test_semantic_layer_output_shape():
    B, T, D = 2, 32, 512
    layer = SemanticLayer(d_model=D, d_ffn=2048, n_heads=8, n_kv_heads=2)
    x = torch.randn(B, T, D)
    cope = CoPE(D)
    out = layer(x, cope)
    assert out.shape == (B, T, D)


def test_semantic_layer_causal():
    """Changing token at position t should not affect positions < t."""
    B, T, D = 1, 16, 64
    layer = SemanticLayer(d_model=D, d_ffn=128, n_heads=4, n_kv_heads=2)
    cope = CoPE(D)
    layer.eval()
    x1 = torch.randn(B, T, D)
    x2 = x1.clone()
    x2[0, 8, :] = torch.randn(D)
    out1 = layer(x1, cope)
    out2 = layer(x2, cope)
    assert torch.allclose(out1[0, :8], out2[0, :8], atol=1e-5), \
        "Semantic Layer is leaking future information"
