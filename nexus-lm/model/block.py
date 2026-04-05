import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from model.surface import PatternLayer, SemanticLayer
from model.reasoning import ReasoningStream
from model.bridge import BridgeLayer
from model.verify import VerifyLayer
from model.difficulty import DifficultyEstimator
from model.cope import CoPE


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
        self.reasoning = ReasoningStream(
            d_r=d_r,
            n_slots=config['n_reasoning_slots'],
            n_heads=config['n_heads_reasoning'],
            d_ffn=config['d_ffn_reasoning'],
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

        # Step 3: Decide how many reasoning iterations are needed
        k_batch, k_logits = self.difficulty(s)

        # Step 4: Run reasoning stream K times (non-causal private workspace)
        r = self.reasoning(r, n_iter=k_batch)

        # Step 5: Bidirectional S↔R communication
        # gate_v gates the sparse R→S write-back (from previous block's Verify)
        s, r = self.bridge(s, r, gate_v)

        # Step 6: Compute surprise signal and produce gate for next block
        surprise, gate_v_new = self.verify(s, r)
        surprise_loss = surprise.mean()

        return s, r, gate_v_new, surprise_loss
