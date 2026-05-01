import torch
from torch.utils.data import DataLoader, TensorDataset

from model.model import NexusAurora, AuroraConfig
from evaluation.perplexity import perplexity_to_bpb, compute_bpb
from evaluation.routing_analysis import analyze_routing


def make_tiny_model():
    cfg = AuroraConfig(
        d_surface=32,
        d_reasoning=16,
        d_verify=8,
        n_reasoning_slots=4,
        n_blocks=1,
        n_heads_surface=2,
        n_kv_heads_surface=1,
        n_heads_reasoning=2,
        d_ffn_pattern=32,
        d_ffn_semantic=64,
        d_ffn_reasoning=32,
        local_window=4,
        bridge_top_k=2,
        cope_positions=4,
        vocab_size=64,
        max_seq_len=16,
        surprise_loss_weight=0.1,
        difficulty_entropy_weight=0.01,
    )
    return NexusAurora(cfg)


def make_loader(batch_size=2, batches=2, seq_len=8, vocab_size=64):
    x = torch.randint(0, vocab_size, (batch_size * batches, seq_len))
    y = torch.randint(0, vocab_size, (batch_size * batches, seq_len))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size)


def test_perplexity_to_bpb_monotonic():
    low = perplexity_to_bpb(10.0)
    high = perplexity_to_bpb(100.0)
    assert low < high


def test_compute_bpb_smoke_cpu():
    model = make_tiny_model()
    loader = make_loader()
    bpb = compute_bpb(model, loader, device=torch.device('cpu'), dtype=torch.float32, max_batches=2)
    assert isinstance(bpb, float)
    assert bpb > 0


def test_compute_bpb_uses_ce_loss_when_metrics_available():
    class DummyMetricModel(torch.nn.Module):
        def forward(self, x, y=None, return_metrics=False):
            logits = torch.zeros(*x.shape, 4)
            if return_metrics:
                return logits, {
                    'loss': torch.tensor(10.0),
                    'ce_loss': torch.tensor(1.0),
                }
            return logits, torch.tensor(10.0)

    x = torch.randint(0, 4, (2, 4))
    ds = TensorDataset(x, x)
    loader = DataLoader(ds, batch_size=2)
    bpb = compute_bpb(
        DummyMetricModel(),
        loader,
        device=torch.device('cpu'),
        dtype=torch.float32,
        mean_bytes_per_token=1.0,
        max_batches=1,
    )
    assert abs(bpb - (1.0 / torch.log(torch.tensor(2.0))).item()) < 1e-5


def test_analyze_routing_smoke_cpu():
    model = make_tiny_model()
    loader = make_loader()
    stats = analyze_routing(model, loader, device=torch.device('cpu'), n_batches=1)
    assert 'k_1' in stats and 'k_2' in stats and 'k_4' in stats
    assert 'surprise_mean' in stats and 'surprise_std' in stats
    assert stats['total_positions_analyzed'] > 0
