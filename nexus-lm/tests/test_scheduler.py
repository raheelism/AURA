import pytest
from training.scheduler import WSDScheduler


def test_wsd_warmup_phase():
    """LR should increase linearly during warmup."""
    sched = WSDScheduler(
        max_lr=1e-3, min_lr=1e-4,
        warmup_tokens=1000, stable_tokens=8000, decay_tokens=1000,
    )
    lr0 = sched.get_lr(0)
    lr500 = sched.get_lr(500)
    lr1000 = sched.get_lr(1000)
    assert lr0 < lr500 < lr1000, "LR should increase during warmup"
    assert abs(lr1000 - 1e-3) < 1e-6, f"At end of warmup, LR should be max_lr, got {lr1000}"


def test_wsd_stable_phase():
    """LR should be constant = max_lr during stable phase."""
    sched = WSDScheduler(
        max_lr=1e-3, min_lr=1e-4,
        warmup_tokens=1000, stable_tokens=8000, decay_tokens=1000,
    )
    lr_stable_start = sched.get_lr(1000)
    lr_stable_mid = sched.get_lr(5000)
    lr_stable_end = sched.get_lr(9000)
    assert abs(lr_stable_start - 1e-3) < 1e-7
    assert abs(lr_stable_mid - 1e-3) < 1e-7
    assert abs(lr_stable_end - 1e-3) < 1e-7


def test_wsd_decay_phase():
    """LR should decrease from max to min during decay."""
    sched = WSDScheduler(
        max_lr=1e-3, min_lr=1e-4,
        warmup_tokens=1000, stable_tokens=8000, decay_tokens=1000,
    )
    lr_decay_start = sched.get_lr(9000)
    lr_decay_mid = sched.get_lr(9500)
    lr_decay_end = sched.get_lr(10000)
    assert lr_decay_start >= lr_decay_mid >= lr_decay_end
    assert abs(lr_decay_start - 1e-3) < 1e-6
    assert abs(lr_decay_end - 1e-4) < 1e-6


def test_wsd_update_lr():
    """update() should set optimizer param_group lr."""
    import torch
    sched = WSDScheduler(max_lr=1e-3, min_lr=1e-4,
                         warmup_tokens=100, stable_tokens=800, decay_tokens=100)
    param = torch.nn.Parameter(torch.randn(4, 4))
    opt = torch.optim.SGD([param], lr=0.0)
    sched.update(opt, tokens_seen=50)
    lr = opt.param_groups[0]['lr']
    expected = sched.get_lr(50)
    assert abs(lr - expected) < 1e-9
