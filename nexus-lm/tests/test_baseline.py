import torch
from model.baseline import LLaMABaseline, LLaMAConfig


def test_llama_forward_shape():
    config = LLaMAConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
                         d_ffn=128, vocab_size=256, max_seq_len=32)
    model = LLaMABaseline(config)
    B, T = 2, 16
    ids = torch.randint(0, 256, (B, T))
    logits = model(ids)
    assert logits.shape == (B, T, 256)


def test_llama_loss_computed():
    config = LLaMAConfig(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
                         d_ffn=128, vocab_size=256, max_seq_len=32)
    model = LLaMABaseline(config)
    B, T = 2, 16
    ids = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    logits, loss = model(ids, targets)
    assert loss.shape == ()
    assert loss.item() > 0


def test_llama_param_count():
    # d_ffn=3300 is chosen to yield ~50M params, matching NexusAurora for fair comparison
    config = LLaMAConfig(n_layers=8, d_model=512, n_heads=8, n_kv_heads=2,
                         d_ffn=3300, vocab_size=8192, max_seq_len=512)
    model = LLaMABaseline(config)
    n = sum(p.numel() for p in model.parameters())
    assert 40_000_000 < n < 60_000_000, f"Expected ~50M, got {n:,}"
