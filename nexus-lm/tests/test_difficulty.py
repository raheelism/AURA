import torch
import pytest
from model.difficulty import DifficultyEstimator


def test_difficulty_output_k_values():
    B, T, D = 2, 16, 512
    est = DifficultyEstimator(d_s=D, hidden=128)
    s = torch.randn(B, T, D)
    k_batch, logits = est(s)
    assert k_batch in [1, 2, 4], f"k_batch must be in {{1,2,4}}, got {k_batch}"
    assert logits.shape == (B, T, 3)


def test_difficulty_k_batch_is_max():
    B, T, D = 2, 32, 64
    est = DifficultyEstimator(d_s=D, hidden=32)
    with torch.no_grad():
        est.net[-1].bias.fill_(0)
        est.net[-1].bias[2] = 10.0  # strongly predict K=4 (index 2)
    s = torch.randn(B, T, D)
    k_batch, _ = est(s)
    assert k_batch == 4


def test_difficulty_gradients_via_gumbel():
    B, T, D = 2, 8, 64
    est = DifficultyEstimator(d_s=D, hidden=32)
    s = torch.randn(B, T, D)
    k_batch, logits = est(s)
    loss = logits.sum()
    loss.backward()
    for p in est.parameters():
        assert p.grad is not None, f"Parameter {p.shape} has no gradient"


def test_difficulty_entropy_regularization():
    B, T, D = 2, 16, 64
    est = DifficultyEstimator(d_s=D, hidden=32)
    s = torch.randn(B, T, D)
    k_batch, logits = est(s)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
    assert entropy.item() >= 0
    assert entropy.requires_grad
