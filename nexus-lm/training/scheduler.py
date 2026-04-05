import math
from torch.optim import Optimizer


class WSDScheduler:
    """
    Warmup-Stable-Decay (WSD) learning rate schedule.
    Proven optimal for LLM pretraining (ICML 2025).

    Phase 1 — Warmup:  LR increases linearly from 0 to max_lr
    Phase 2 — Stable:  LR held constant at max_lr
    Phase 3 — Decay:   LR decreases via cosine from max_lr to min_lr

    Token-based (not step-based) so batch size and gradient accumulation
    changes don't affect the schedule shape.
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_tokens: int,
        stable_tokens: int,
        decay_tokens: int,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_end = warmup_tokens
        self.stable_end = warmup_tokens + stable_tokens
        self.decay_end = warmup_tokens + stable_tokens + decay_tokens

    def get_lr(self, tokens_seen: int) -> float:
        """Compute learning rate at a given number of tokens seen."""
        if tokens_seen < self.warmup_end:
            # Linear warmup from 0 to max_lr
            return self.max_lr * tokens_seen / max(self.warmup_end, 1)
        elif tokens_seen < self.stable_end:
            # Constant stable phase
            return self.max_lr
        elif tokens_seen < self.decay_end:
            # Cosine decay from max_lr to min_lr
            progress = (tokens_seen - self.stable_end) / (self.decay_end - self.stable_end)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine
        else:
            return self.min_lr

    def update(self, optimizer: Optimizer, tokens_seen: int) -> float:
        """Update all param groups in optimizer with current LR. Returns new LR."""
        lr = self.get_lr(tokens_seen)
        for group in optimizer.param_groups:
            group['lr'] = lr
        return lr
