import torch
import pytest
from training.muon import Muon, newton_schulz_orthogonalize


def test_newton_schulz_returns_orthogonal():
    """Newton-Schulz iteration should produce approximately orthogonal matrix."""
    G = torch.randn(64, 32)
    O = newton_schulz_orthogonalize(G, steps=5)
    assert O.shape == G.shape
    # For a tall matrix, O^T @ O should be close to I
    product = O.T @ O
    identity = torch.eye(32)
    assert torch.allclose(product, identity, atol=0.1), \
        f"O^T @ O not close to identity: max err={(product - identity).abs().max():.4f}"


def test_muon_step_updates_params():
    """Muon should update parameters after a step."""
    model = torch.nn.Linear(64, 32, bias=False)
    params_before = model.weight.data.clone()
    optimizer = Muon([model.weight], lr=0.01, momentum=0.95)
    x = torch.randn(4, 64)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    assert not torch.allclose(model.weight.data, params_before), \
        "Muon should update parameters"


def test_muon_only_works_on_2d_params():
    """Muon requires 2D matrix parameters."""
    bias = torch.nn.Parameter(torch.randn(32))
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        opt = Muon([bias], lr=0.01)
        bias.grad = torch.randn(32)
        opt.step()


def test_muon_zero_grad_clears_gradients():
    model = torch.nn.Linear(32, 16, bias=False)
    opt = Muon([model.weight], lr=0.01)
    x = torch.randn(4, 32)
    model(x).sum().backward()
    assert model.weight.grad is not None
    opt.zero_grad()
    assert model.weight.grad is None or (model.weight.grad == 0).all()
