import math
import torch
import pytest
from model.model import NexusAurora, AuroraConfig


def small_config():
    return AuroraConfig(
        d_surface=64,
        d_reasoning=32,
        d_verify=16,
        n_reasoning_slots=8,
        n_blocks=2,
        n_heads_surface=4,
        n_kv_heads_surface=2,
        n_heads_reasoning=2,
        d_ffn_pattern=64,
        d_ffn_semantic=256,
        d_ffn_reasoning=64,
        local_window=8,
        bridge_top_k=4,
        cope_positions=8,
        vocab_size=256,
        max_seq_len=32,
        surprise_loss_weight=0.1,
        difficulty_entropy_weight=0.01,
    )


def test_aurora_forward_logits_shape():
    config = small_config()
    model = NexusAurora(config)
    B, T = 2, 16
    input_ids = torch.randint(0, 256, (B, T))
    logits = model(input_ids)
    assert logits.shape == (B, T, 256)


def test_aurora_forward_with_loss():
    config = small_config()
    model = NexusAurora(config)
    B, T = 2, 16
    input_ids = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    logits, loss = model(input_ids, targets)
    assert logits.shape == (B, T, 256)
    assert loss.shape == ()
    assert loss.item() > 0


def test_aurora_loss_decreases_with_training():
    """One gradient step should reduce loss."""
    config = small_config()
    model = NexusAurora(config)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    B, T = 2, 16
    input_ids = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    _, loss1 = model(input_ids, targets)
    loss1.backward()
    opt.step()
    opt.zero_grad()
    _, loss2 = model(input_ids, targets)
    assert loss2.item() < loss1.item(), \
        f"Loss should decrease after one step: {loss1.item():.4f} -> {loss2.item():.4f}"


def test_aurora_embedding_weight_tied():
    """Input embedding and output projection must share weights."""
    config = small_config()
    model = NexusAurora(config)
    assert model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr(), \
        "Embedding and lm_head must share weight tensors"


def test_aurora_parameter_count():
    """Full-scale model should be approximately 50M parameters."""
    config = AuroraConfig(
        d_surface=512, d_reasoning=256, d_verify=64,
        n_reasoning_slots=32, n_blocks=6,
        n_heads_surface=8, n_kv_heads_surface=2,
        n_heads_reasoning=4,
        d_ffn_pattern=512, d_ffn_semantic=2048, d_ffn_reasoning=512,
        local_window=64, bridge_top_k=8,
        cope_positions=16, vocab_size=8192, max_seq_len=512,
        surprise_loss_weight=0.1, difficulty_entropy_weight=0.01,
    )
    model = NexusAurora(config)
    n_params = sum(p.numel() for p in model.parameters())
    assert 40_000_000 < n_params < 60_000_000, \
        f"Expected ~50M params, got {n_params:,}"


def test_bpb_computation():
    from evaluation.perplexity import perplexity_to_bpb
    ppl = 100.0
    bpb = perplexity_to_bpb(ppl, mean_bytes_per_token=3.5)
    expected = math.log2(100.0) / 3.5
    assert abs(bpb - expected) < 1e-5
