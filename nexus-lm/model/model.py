import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from model.block import AuroraBlock
from model.cope import CoPE
from model.surface import RMSNorm


@dataclass
class AuroraConfig:
    d_surface: int = 512
    d_reasoning: int = 256
    d_verify: int = 64
    n_reasoning_slots: int = 32
    n_blocks: int = 6
    n_heads_surface: int = 8
    n_kv_heads_surface: int = 2
    n_heads_reasoning: int = 4
    d_ffn_pattern: int = 512
    d_ffn_semantic: int = 2048
    d_ffn_reasoning: int = 512
    local_window: int = 64
    bridge_top_k: int = 8
    cope_positions: int = 16
    vocab_size: int = 8192
    max_seq_len: int = 512
    surprise_loss_weight: float = 0.1
    difficulty_entropy_weight: float = 0.01

    def to_block_config(self) -> dict:
        return {
            'd_surface': self.d_surface,
            'd_reasoning': self.d_reasoning,
            'd_verify': self.d_verify,
            'n_reasoning_slots': self.n_reasoning_slots,
            'n_heads_surface': self.n_heads_surface,
            'n_kv_heads_surface': self.n_kv_heads_surface,
            'n_heads_reasoning': self.n_heads_reasoning,
            'd_ffn_pattern': self.d_ffn_pattern,
            'd_ffn_semantic': self.d_ffn_semantic,
            'd_ffn_reasoning': self.d_ffn_reasoning,
            'local_window': self.local_window,
            'bridge_top_k': self.bridge_top_k,
            'surprise_loss_weight': self.surprise_loss_weight,
            'difficulty_entropy_weight': self.difficulty_entropy_weight,
        }


class NexusAurora(nn.Module):
    """
    Full NEXUS-AURORA model.

    Three parallel streams:
    - Surface (S, d_surface): generates tokens, autoregressive
    - Reasoning (R, d_reasoning): private workspace, never generates tokens
    - Verification gate (V): scalar gate controlling R→S write-back strength

    Each of n_blocks AuroraBlocks processes all three streams.
    Loss = CrossEntropy(S) + surprise_loss_weight * mean(surprise_losses)
    """

    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_surface)
        self.cope = CoPE(config.d_surface, n_positions=config.cope_positions)
        self.blocks = nn.ModuleList([
            AuroraBlock(config.to_block_config()) for _ in range(config.n_blocks)
        ])
        self.norm = RMSNorm(config.d_surface)
        self.lm_head = nn.Linear(config.d_surface, config.vocab_size, bias=False)

        # Weight tying: input embedding = output projection
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for block in self.blocks:
            for name, p in block.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.normal_(p, std=0.02)
                elif 'bias' in name:
                    nn.init.zeros_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            input_ids: (B, T) token IDs
            targets:   (B, T) target token IDs for loss (optional)
        Returns:
            If targets is None: logits (B, T, vocab_size)
            If targets given:   (logits, total_loss)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Surface stream: token embeddings
        s = self.embedding(input_ids)  # (B, T, d_surface)

        # Reasoning stream: each block owns its own slots parameter;
        # we initialize R from the first block's learned slots
        r = self.blocks[0].reasoning.slots.unsqueeze(0).expand(B, -1, -1).clone()
        # (B, n_slots, d_reasoning)

        # Verification gate: starts at 1.0 (full write-back initially)
        gate_v = torch.ones(B, T, 1, device=device, dtype=s.dtype)

        # Forward through all blocks
        total_surprise_loss = torch.tensor(0.0, device=device, dtype=s.dtype)
        for block in self.blocks:
            s, r, gate_v, surprise_loss = block(s, r, gate_v, self.cope)
            total_surprise_loss = total_surprise_loss + surprise_loss

        # Final norm and project to vocabulary
        s = self.norm(s)
        logits = self.lm_head(s)  # (B, T, vocab_size)

        if targets is None:
            return logits

        ce_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
        )
        mean_surprise = total_surprise_loss / self.config.n_blocks
        total_loss = ce_loss + self.config.surprise_loss_weight * mean_surprise

        return logits, total_loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Simple autoregressive generation for evaluation."""
        self.eval()
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.config.max_seq_len:])
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids
