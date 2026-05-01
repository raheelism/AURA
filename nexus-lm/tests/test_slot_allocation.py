import torch
from model.slot_allocation import SlotAllocator
from model.reasoning import ReasoningStream


def test_slot_allocator_mask_shape_and_bounds():
    B, T, d_s, d_r, N = 2, 16, 64, 32, 8
    s = torch.randn(B, T, d_s)
    r = torch.randn(B, N, d_r)
    alloc = SlotAllocator(d_s=d_s, d_r=d_r, hidden=32, min_slots=2, max_slots=6)
    mask = alloc(s, r)
    assert mask.shape == (B, N, 1)
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0
    active = mask.squeeze(-1).sum(dim=-1)
    assert torch.all(active >= 2)
    assert torch.all(active <= 6)


def test_masked_reasoning_leaves_inactive_slots_unchanged():
    B, N, d_r = 2, 8, 32
    r = torch.randn(B, N, d_r)
    stream = ReasoningStream(d_r=d_r, n_slots=N, n_heads=4, d_ffn=64)
    mask = torch.zeros(B, N, 1)
    mask[:, :2] = 1.0
    out = stream.forward_masked(r, n_iter=1, slot_mask=mask)
    assert out.shape == r.shape
    # inactive slots should remain exactly the same because mask=0
    assert torch.allclose(out[:, 2:], r[:, 2:], atol=1e-6)


def test_slot_allocator_straight_through_gradients():
    B, T, d_s, d_r, N = 2, 8, 32, 16, 4
    s = torch.randn(B, T, d_s)
    r = torch.randn(B, T, N, d_r)
    alloc = SlotAllocator(d_s=d_s, d_r=d_r, hidden=16, min_slots=1, max_slots=2)
    mask = alloc(s, r)
    loss = (mask * r).sum()
    loss.backward()
    for p in alloc.parameters():
        assert p.grad is not None
