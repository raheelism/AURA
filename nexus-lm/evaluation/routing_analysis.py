import torch
import torch.nn as nn
from typing import Dict


def analyze_routing(
    model: nn.Module,
    dataloader,
    device: torch.device,
    n_batches: int = 50,
) -> Dict[str, float]:
    """
    Compute routing diagnostics for NexusAurora.

    Handles adaptive models with active DifficultyEstimator modules and ablation
    variants where routing is disabled or fixed.
    """
    model.eval()
    k_counts = {}
    surprise_list = []

    def capture_k_logits(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            logits = output[1]
        else:
            return
        k_values = getattr(module, 'k_values', (1, 2, 4))
        k_indices = logits.detach().cpu().argmax(dim=-1)
        for j, k_val in enumerate(k_values):
            k_counts[k_val] = k_counts.get(k_val, 0) + (k_indices == j).sum().item()

    def capture_surprise(module, input, output):
        surprise, _ = output
        surprise_list.append(surprise.detach().cpu())

    hooks = []
    for block in getattr(model, 'blocks', []):
        if getattr(block, 'difficulty', None) is not None:
            hooks.append(block.difficulty.register_forward_hook(capture_k_logits))
        if getattr(block, 'verify', None) is not None:
            hooks.append(block.verify.register_forward_hook(capture_surprise))

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= n_batches:
                break
            x = x.to(device)
            _ = model(x)

    for h in hooks:
        h.remove()

    total = max(sum(k_counts.values()), 1)
    k_dist = {f'k_{k}': v / total for k, v in sorted(k_counts.items())}
    for k in (1, 2, 4):
        k_dist.setdefault(f'k_{k}', 0.0)

    if surprise_list:
        all_surprise = torch.cat([s.view(-1) for s in surprise_list])
        surprise_mean = all_surprise.mean().item()
        surprise_std = all_surprise.std().item()
    else:
        surprise_mean, surprise_std = 0.0, 0.0

    return {
        **k_dist,
        'surprise_mean': surprise_mean,
        'surprise_std': surprise_std,
        'total_positions_analyzed': total,
    }
