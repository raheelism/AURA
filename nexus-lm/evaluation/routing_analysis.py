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
    Compute routing diagnostics for NexusAurora:

    - k_1 / k_2 / k_4: fraction of positions routed to K=1, K=2, K=4
    - surprise_mean: mean surprise score (lower = S and R more consistent)
    - surprise_std: std of surprise scores
    - total_positions_analyzed: total positions used for statistics

    Returns dict of diagnostic metrics.
    """
    from model.difficulty import K_VALUES

    model.eval()
    k_counts = {1: 0, 2: 0, 4: 0}
    k_logits_list = []
    surprise_list = []

    def capture_k_logits(module, input, output):
        k_batch, logits = output
        k_logits_list.append(logits.detach().cpu())

    def capture_surprise(module, input, output):
        surprise, gate_v = output
        surprise_list.append(surprise.detach().cpu())

    hooks = []
    for block in model.blocks:
        hooks.append(block.difficulty.register_forward_hook(capture_k_logits))
        hooks.append(block.verify.register_forward_hook(capture_surprise))

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= n_batches:
                break
            x = x.to(device)
            _ = model(x)

    for h in hooks:
        h.remove()

    # Aggregate K distribution
    for logits in k_logits_list:
        k_indices = logits.argmax(dim=-1)  # (B, T)
        for j, k_val in enumerate(K_VALUES):
            k_counts[k_val] += (k_indices == j).sum().item()

    total = max(sum(k_counts.values()), 1)
    k_dist = {f'k_{k}': v / total for k, v in k_counts.items()}

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
