import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple
from model.surface import PatternLayer, SemanticLayer
from model.reasoning import ReasoningStream
from model.bridge import BridgeLayer
from model.verify import VerifyLayer
from model.difficulty import DifficultyEstimator
from model.cope import CoPE
from model.halting import HaltingEstimator
from model.slot_allocation import SlotAllocator


class AuroraBlock(nn.Module):
    """
    Single AURORA block with causally safe prefix-local reasoning support.

    When the reasoning state has shape (B, T, N, d_r), each position owns a
    private set of slots grounded only in its prefix. This is the mode used by
    NexusAurora and is the critical fix for multi-block autoregressive safety.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        d_s = config['d_surface']
        d_r = config['d_reasoning']
        d_v = config['d_verify']

        self.use_reasoning = bool(config.get('use_reasoning', True))
        self.use_verify = bool(config.get('use_verify', True))
        self.use_slot_allocator = bool(config.get('use_slot_allocator', True))
        self.use_halting_gate = bool(config.get('use_halting_gate', True))
        self.fixed_k = config.get('fixed_k', None)

        max_k = int(config.get('max_k', 4))
        self.k_values = tuple(k for k in (1, 2, 4) if k <= max_k)
        if not self.k_values:
            self.k_values = (1,)

        self.pattern = PatternLayer(
            d_model=d_s,
            d_ffn=config['d_ffn_pattern'],
            window_size=config['local_window'],
        )
        self.semantic = SemanticLayer(
            d_model=d_s,
            d_ffn=config['d_ffn_semantic'],
            n_heads=config['n_heads_surface'],
            n_kv_heads=config['n_kv_heads_surface'],
        )

        self.difficulty = None
        self.halting = None
        self.reasoning = None
        self.slot_allocator = None
        self.bridge = None
        self.verify = None

        if self.use_reasoning:
            if self.fixed_k is None:
                self.difficulty = DifficultyEstimator(d_s=d_s, hidden=128, k_values=self.k_values)
            self.halting = HaltingEstimator(
                d_s=d_s,
                d_r=d_r,
                hidden=128,
                tau=config.get('halting_tau', 1.0),
                k_max=max_k,
                quantile=config.get('halting_quantile', 0.9),
            )
            self.reasoning = ReasoningStream(
                d_r=d_r,
                n_slots=config['n_reasoning_slots'],
                n_heads=config['n_heads_reasoning'],
                d_ffn=config['d_ffn_reasoning'],
            )
            if self.use_slot_allocator:
                self.slot_allocator = SlotAllocator(
                    d_s=d_s,
                    d_r=d_r,
                    hidden=config.get('slot_allocator_hidden', 128),
                    min_slots=config.get('slot_allocator_min_slots', max(1, config['n_reasoning_slots'] // 4)),
                    max_slots=config.get('slot_allocator_max_slots', config['n_reasoning_slots']),
                )
            self.bridge = BridgeLayer(
                d_s=d_s,
                d_r=d_r,
                n_heads=config['n_heads_surface'],
                top_k=config['bridge_top_k'],
            )

        if self.use_verify:
            self.verify = VerifyLayer(d_s=d_s, d_r=d_r, d_v=d_v)

    def _routing_balance_loss(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        target = torch.full_like(probs.mean(dim=(0, 1)), 1.0 / probs.size(-1))
        return F.mse_loss(probs.mean(dim=(0, 1)), target)

    def _combine_adaptive_reasoning(
        self,
        r: torch.Tensor,
        slot_mask: torch.Tensor,
        k_weights: torch.Tensor,
    ) -> torch.Tensor:
        max_k = max(self.k_values)
        candidates = {}
        r_step = r
        for step in range(1, max_k + 1):
            r_step = self.reasoning.forward_masked(r_step, n_iter=1, slot_mask=slot_mask)
            if step in self.k_values:
                candidates[step] = r_step
        if r.dim() == 3:
            stacked = torch.stack([candidates[k] for k in self.k_values], dim=1)
            weights = k_weights.mean(dim=1)
            return (weights[..., None, None] * stacked).sum(dim=1)
        stacked = torch.stack([candidates[k] for k in self.k_values], dim=2)
        return (k_weights[..., None, None] * stacked).sum(dim=2)

    @torch.no_grad()
    def _run_selected_reasoning(
        self,
        r: torch.Tensor,
        slot_mask: torch.Tensor,
        k_logits: torch.Tensor,
    ) -> torch.Tensor:
        k_idx = k_logits.argmax(dim=-1)
        k_values = torch.tensor(self.k_values, device=k_logits.device)
        k_per_position = k_values[k_idx]

        if r.dim() != 4:
            k_max = int(k_per_position.max().item())
            return self.reasoning.forward_masked(r, n_iter=k_max, slot_mask=slot_mask)

        B, T, N, D = r.shape
        r_out = r
        flat_mask = slot_mask.reshape(B * T, N, 1)
        for step in range(1, int(k_per_position.max().item()) + 1):
            active = (k_per_position >= step).reshape(-1)
            if not bool(active.any()):
                continue
            flat_r = r_out.reshape(B * T, N, D).clone()
            updated = self.reasoning.forward_masked(
                flat_r[active],
                n_iter=1,
                slot_mask=flat_mask[active],
            )
            flat_r[active] = updated
            r_out = flat_r.reshape(B, T, N, D)
        return r_out

    def forward(
        self,
        s: torch.Tensor,
        r: torch.Tensor,
        gate_v: torch.Tensor,
        cope: CoPE,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_r_was_global = r.dim() == 3
        if input_r_was_global:
            r = r.unsqueeze(1).expand(-1, s.size(1), -1, -1).clone()

        s = self.pattern(s)
        s = self.semantic(s, cope)

        device = s.device
        surprise_loss = torch.tensor(0.0, device=device, dtype=s.dtype)
        routing_loss = torch.tensor(0.0, device=device, dtype=s.dtype)
        mean_k = torch.tensor(0.0, device=device, dtype=s.dtype)

        if self.use_reasoning:
            if self.slot_allocator is not None:
                slot_mask = self.slot_allocator(s, r)
            else:
                slot_mask = torch.ones(*r.shape[:-1], 1, device=device, dtype=s.dtype)

            if self.fixed_k is not None:
                fixed_k = int(self.fixed_k)
                r = self.reasoning.forward_masked(r, n_iter=fixed_k, slot_mask=slot_mask)
                mean_k = torch.tensor(float(fixed_k), device=device, dtype=s.dtype)
            else:
                _, k_logits, k_weights = self.difficulty(s, return_weights=True)
                routing_loss = self._routing_balance_loss(k_logits)
                probs = torch.softmax(k_logits, dim=-1)
                k_tensor = torch.tensor(self.k_values, device=device, dtype=s.dtype)
                mean_k = (probs * k_tensor.view(1, 1, -1)).sum(dim=-1).mean()
                if self.training:
                    r = self._combine_adaptive_reasoning(r, slot_mask, k_weights)
                else:
                    r = self._run_selected_reasoning(r, slot_mask, k_logits)

            r_pooled = r.mean(dim=2) if r.dim() == 4 else r.mean(dim=1)
            halting_p = self.halting(s, r_pooled) if self.use_halting_gate else None
            s, r = self.bridge(s, r, gate_v, halting_probs=halting_p)

        if self.verify is not None:
            surprise, gate_v_new = self.verify(s, r)
            surprise_loss = surprise.mean()
        else:
            gate_v_new = torch.ones_like(gate_v)

        r_out = r[:, -1] if input_r_was_global else r
        if return_aux:
            aux = {
                'surprise_loss': surprise_loss,
                'routing_loss': routing_loss,
                'mean_k': mean_k,
            }
            return s, r_out, gate_v_new, surprise_loss, aux
        return s, r_out, gate_v_new, surprise_loss
