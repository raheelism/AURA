import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


def perplexity_to_bpb(perplexity: float, mean_bytes_per_token: float = 3.5) -> float:
    """
    Convert perplexity to bits-per-byte (BPB).
    BPB is tokenizer-independent and enables fair cross-model comparison.

    BPB = log2(perplexity) / mean_bytes_per_token

    For BPE vocab=8192 on English text, mean ~3.5 bytes/token is typical.
    Lower BPB = better model.
    """
    perplexity = max(float(perplexity), 1e-12)
    return math.log2(perplexity) / mean_bytes_per_token


def compute_bpb(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    mean_bytes_per_token: float = 3.5,
    max_batches: int = 200,
) -> float:
    """
    Compute BPB on a dataset.

    Args:
        model: language model with forward(ids, targets) -> (logits, loss) interface
        dataloader: yields (x, y) batches
        device: torch device
        dtype: computation dtype (float16 or float32)
        mean_bytes_per_token: tokenizer-specific constant for BPB conversion
        max_batches: cap evaluation at this many batches for speed

    Returns:
        BPB (bits per byte) — lower is better
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    autocast_device = device.type if device.type != 'mps' else 'cpu'

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=autocast_device, dtype=dtype,
                                enabled=(dtype == torch.float16)):
                try:
                    _, metrics = model(x, y, return_metrics=True)
                    loss = metrics.get('ce_loss', metrics.get('loss'))
                except TypeError:
                    _, loss = model(x, y)
            n_tokens = y.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(mean_loss, 80.0))
    return perplexity_to_bpb(perplexity, mean_bytes_per_token)
