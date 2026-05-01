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
    use_reasoning: bool = True
    use_verify: bool = True
    use_slot_allocator: bool = True
    use_halting_gate: bool = True
    fixed_k: Optional[int] = None
    max_k: int = 4

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
            'use_reasoning': self.use_reasoning,
            'use_verify': self.use_verify,
            'use_slot_allocator': self.use_slot_allocator,
            'use_halting_gate': self.use_halting_gate,
            'fixed_k': self.fixed_k,
            'max_k': self.max_k,
        }


class NexusAurora(nn.Module):
    """
    Full NEXUS-AURORA model.

    Surface states produce tokens. The reasoning stream is prefix-local in the
    full model: r has shape (B, T, n_slots, d_reasoning), so each position's
    private workspace can only be grounded in its own prefix.
    """

    def __init__(self, config: AuroraConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_surface)
        self.reasoning_slots = nn.Parameter(
            torch.randn(config.n_reasoning_slots, config.d_reasoning) * 0.02
        )
        self.cope = CoPE(config.d_surface, n_positions=config.cope_positions)
        self.blocks = nn.ModuleList([
            AuroraBlock(config.to_block_config()) for _ in range(config.n_blocks)
        ])
        self.norm = RMSNorm(config.d_surface)
        self.lm_head = nn.Linear(config.d_surface, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.reasoning_slots, std=0.02)
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
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        B, T = input_ids.shape
        device = input_ids.device

        s = self.embedding(input_ids)
        r = self.reasoning_slots.view(1, 1, self.config.n_reasoning_slots, self.config.d_reasoning)
        r = r.expand(B, T, -1, -1).clone()
        gate_v = torch.ones(B, T, 1, device=device, dtype=s.dtype)

        total_surprise_loss = torch.tensor(0.0, device=device, dtype=s.dtype)
        total_routing_loss = torch.tensor(0.0, device=device, dtype=s.dtype)
        total_mean_k = torch.tensor(0.0, device=device, dtype=s.dtype)
        for block in self.blocks:
            s, r, gate_v, surprise_loss, aux = block(
                s, r, gate_v, self.cope, return_aux=True
            )
            total_surprise_loss = total_surprise_loss + surprise_loss
            total_routing_loss = total_routing_loss + aux['routing_loss']
            total_mean_k = total_mean_k + aux['mean_k']

        s = self.norm(s)
        logits = self.lm_head(s)

        if targets is None:
            return logits

        ce_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
        )
        mean_surprise = total_surprise_loss / max(self.config.n_blocks, 1)
        mean_routing = total_routing_loss / max(self.config.n_blocks, 1)
        mean_k = total_mean_k / max(self.config.n_blocks, 1)
        aux_loss = (
            self.config.surprise_loss_weight * mean_surprise
            + self.config.difficulty_entropy_weight * mean_routing
        )
        total_loss = ce_loss + aux_loss

        if return_metrics:
            return logits, {
                'loss': total_loss,
                'ce_loss': ce_loss,
                'aux_loss': aux_loss,
                'surprise_loss': mean_surprise,
                'routing_loss': mean_routing,
                'mean_k': mean_k.detach(),
            }
        return logits, total_loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
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
