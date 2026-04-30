import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
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
    Single AURORA block. Execution order per forward pass:
    1. Pattern Layer    — causal local-window attention (syntax/surface patterns)
    2. Semantic Layer   — full causal GQA with CoPE (long-range semantics)
    3. Difficulty Est.  — predict K ∈ {1, 2, 4} reasoning iterations needed
    4. Reasoning Stream — R self-updates K times (private workspace, non-causal)
    5. Bridge Layer     — bidirectional S↔R cross-attention + gated write-back
    6. Verify Layer     — compute surprise signal, output gate_v for next block

    Causality guarantee:
    - S is processed causally in steps 1-2 before Bridge
    - R reads the already-causal S state → no future token leakage
    - gate_v from Verify gates the sparse R→S write-back in next block
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        d_s = config['d_surface']
        d_r = config['d_reasoning']
        d_v = config['d_verify']

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
        self.difficulty = DifficultyEstimator(d_s=d_s, hidden=128)
        self.halting = HaltingEstimator(
            d_s=d_s,
            d_r=d_r,
            hidden=128,
            tau=config.get('halting_tau', 1.0),
            k_max=config.get('halting_k_max', 4),
            quantile=config.get('halting_quantile', 0.9),
        )
        self.reasoning = ReasoningStream(
            d_r=d_r,
            n_slots=config['n_reasoning_slots'],
            n_heads=config['n_heads_reasoning'],
            d_ffn=config['d_ffn_reasoning'],
        )
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
        self.verify = VerifyLayer(d_s=d_s, d_r=d_r, d_v=d_v)

    def forward(
        self,
        s: torch.Tensor,
        r: torch.Tensor,
        gate_v: torch.Tensor,
        cope: CoPE,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            s:      surface states (B, T, d_s)
            r:      reasoning slots (B, n_slots, d_r)
            gate_v: write-back gate from previous block (B, T, 1)
            cope:   shared CoPE module (from NexusAurora)
        Returns:
            s:             updated surface states (B, T, d_s)
            r:             updated reasoning slots (B, n_slots, d_r)
            gate_v_new:    updated gate for next block (B, T, 1)
            surprise_loss: scalar auxiliary loss for this block
        """
        # Step 1: Local pattern processing (causal local-window attention)
        s = self.pattern(s)

        # Step 2: Long-range semantic processing (full causal GQA + CoPE)
        s = self.semantic(s, cope)

        # Step 3: Decide how many reasoning iterations are needed (difficulty prior)
        k_batch_diff, k_logits = self.difficulty(s)

        # Two-stage halting: run one cheap reasoning iteration, evaluate per-token halting,
        # then run additional iterations up to k_batch determined from difficulty vs halting.
        slot_mask = self.slot_allocator(s, r)
        r = self.reasoning.forward_masked(r, n_iter=1, slot_mask=slot_mask)

        # compute per-token halting probabilities using pooled R
        r_pooled = r.mean(dim=1)  # (B, d_r)
        halting_p = self.halting(s, r_pooled)  # (B, T, 1)
        halting_k = self.halting.compute_k_batch(halting_p)

        # choose GPU-friendly batch iteration count
        k_batch = max(int(k_batch_diff), int(halting_k))
        if k_batch > 1:
            r = self.reasoning.forward_masked(r, n_iter=k_batch - 1, slot_mask=slot_mask)

        # Step 5: Bidirectional S↔R communication
        # gate_v gates the sparse R→S write-back (from previous block's Verify)
        # Pass halting probabilities to bridge so R->S writeback can be gated per-token
        s, r = self.bridge(s, r, gate_v, halting_probs=halting_p)

        # Step 6: Compute surprise signal and produce gate for next block
        surprise, gate_v_new = self.verify(s, r)
        surprise_loss = surprise.mean()

        return s, r, gate_v_new, surprise_loss
