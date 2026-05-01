import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import csv
import time
from pathlib import Path
from training.muon import Muon
from training.scheduler import WSDScheduler


@dataclass
class TrainerConfig:
    batch_size: int = 16
    gradient_accumulation: int = 8
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_tokens: int = 100_000_000
    stable_tokens: int = 1_300_000_000
    decay_tokens: int = 100_000_000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    checkpoint_dir: str = "results/checkpoints"
    checkpoint_every_tokens: int = 50_000_000
    log_interval: int = 100
    val_interval: int = 2000
    max_tokens: int = 1_500_000_000
    device: str = "cuda"
    dtype: str = "float16"
    surprise_loss_weight: float = 0.1
    difficulty_entropy_weight: float = 0.01
    resume_from: Optional[str] = None


class Trainer:
    """
    Training loop for NexusAurora and LLaMABaseline.

    Optimizer strategy:
    - 2D matrix parameters → Muon (orthogonalized gradient updates)
    - All other parameters → AdamW (embeddings, biases, norms)

    Schedule: WSD (Warmup-Stable-Decay), token-based.
    Mixed precision: FP16 on GPU via GradScaler; float32 on CPU.
    Checkpointing: saves model + optimizer state to disk.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: TrainerConfig,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.float16 if config.dtype == 'float16' else torch.float32

        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=0, pin_memory=(config.device == 'cuda'),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size,
            shuffle=False, num_workers=0, drop_last=False,
        )
        self.train_iter = iter(self.train_loader)

        model.to(self.device)
        
        # Enable multi-GPU training with DataParallel if available
        if config.device == 'cuda' and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        else:
            self.model = model

        # Partition parameters: 2D matrices get Muon, rest get AdamW
        muon_params = [p for p in model.parameters()
                       if p.requires_grad and p.dim() == 2]
        adamw_params = [p for p in model.parameters()
                        if p.requires_grad and p.dim() != 2]

        self.optimizer = torch.optim.AdamW(
            adamw_params, lr=config.max_lr * 0.3,
            betas=(0.9, 0.95), weight_decay=config.weight_decay,
        )
        self.muon = Muon(muon_params, lr=config.max_lr, momentum=0.95, nesterov=True) \
            if muon_params else None

        self.scheduler = WSDScheduler(
            max_lr=config.max_lr, min_lr=config.min_lr,
            warmup_tokens=config.warmup_tokens,
            stable_tokens=config.stable_tokens,
            decay_tokens=config.decay_tokens,
        )

        # GradScaler for FP16; disabled (no-op) for float32
        use_amp = (config.dtype == 'float16' and config.device == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # State tracking
        self.step = 0
        self.tokens_seen = 0
        self.last_train_loss = 0.0
        self.last_grad_norm = 0.0
        self.last_val_loss: Optional[float] = None
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_tokens = 0
        self.elapsed_wall_time = 0.0

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self._init_log_file(append=bool(config.resume_from))

        if config.resume_from:
            self.load_checkpoint(config.resume_from)
            resume_now = time.time()
            self.last_log_tokens = self.tokens_seen
            self.last_log_time = resume_now
            self.start_time = resume_now
            print(f"Resumed training from: {config.resume_from}")

    def _init_log_file(self, append: bool = False):
        self.log_path = os.path.join(self.config.checkpoint_dir, "train_log.csv")
        write_header = not (append and os.path.exists(self.log_path))
        mode = 'a' if append else 'w'
        with open(self.log_path, mode, newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'step', 'tokens_seen', 'train_loss', 'val_loss', 'lr',
                    'grad_norm', 'tokens_per_sec', 'wall_time'
                ])

    def _append_log_row(
        self,
        train_loss: float,
        val_loss: Optional[float],
        tokens_per_sec: float,
        wall_time: float,
    ) -> None:
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.step,
                self.tokens_seen,
                train_loss,
                val_loss if val_loss is not None else "",
                self.scheduler.get_lr(self.tokens_seen),
                self.last_grad_norm,
                tokens_per_sec,
                wall_time,
            ])

    def _get_batch(self):
        try:
            x, y = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            x, y = next(self.train_iter)
        return x.to(self.device), y.to(self.device)

    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Make DataParallel scalar gathers safe by collapsing to one scalar."""
        return loss.mean() if loss.dim() > 0 else loss

    def train_one_step(self) -> float:
        """Run one gradient accumulation cycle. Returns mean train loss."""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        if self.muon:
            self.muon.zero_grad()

        for _ in range(self.config.gradient_accumulation):
            x, y = self._get_batch()
            batch_tokens = x.numel()

            autocast_device = self.device.type if self.device.type != 'mps' else 'cpu'
            with torch.autocast(device_type=autocast_device, dtype=self.dtype,
                                enabled=(self.dtype == torch.float16)):
                _, loss = self.model(x, y)

            loss = self._reduce_loss(loss)

            self.scaler.scale(loss / self.config.gradient_accumulation).backward()
            total_loss += loss.item()
            self.tokens_seen += batch_tokens

        # Unscale before clipping
        self.scaler.unscale_(self.optimizer)
        if self.muon:
            self.scaler.unscale_(self.muon)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        self.last_grad_norm = grad_norm.item()

        self.scaler.step(self.optimizer)
        if self.muon:
            self.scaler.step(self.muon)
        self.scaler.update()

        # Update learning rate
        self.scheduler.update(self.optimizer, self.tokens_seen)
        if self.muon:
            self.scheduler.update(self.muon, self.tokens_seen)

        self.step += 1
        mean_loss = total_loss / self.config.gradient_accumulation
        self.last_train_loss = mean_loss
        return mean_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute mean loss on validation set (capped at 50 batches)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            autocast_device = self.device.type if self.device.type != 'mps' else 'cpu'
            with torch.autocast(device_type=autocast_device, dtype=self.dtype,
                                enabled=(self.dtype == torch.float16)):
                _, loss = self.model(x, y)
            loss = self._reduce_loss(loss)
            total_loss += loss.item()
            n_batches += 1
            if n_batches >= 50:
                break
        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, tag: str):
        path = os.path.join(self.config.checkpoint_dir, f"ckpt_{tag}.pt")
        torch.save({
            'step': self.step,
            'tokens_seen': self.tokens_seen,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'muon_state': self.muon.state_dict() if self.muon else None,
            'scaler_state': self.scaler.state_dict() if self.scaler.is_enabled() else None,
            'elapsed_wall_time': self.elapsed_wall_time + (time.time() - self.start_time),
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint['model_state']
        target_is_dp = isinstance(self.model, nn.DataParallel)
        state_is_dp = bool(state) and all(k.startswith('module.') for k in state.keys())
        if state_is_dp and not target_is_dp:
            state = {k[len('module.'):]: v for k, v in state.items()}
        elif target_is_dp and not state_is_dp:
            state = {f'module.{k}': v for k, v in state.items()}
        self.model.load_state_dict(state)
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.muon and checkpoint.get('muon_state') is not None:
            self.muon.load_state_dict(checkpoint['muon_state'])
        scaler_state = checkpoint.get('scaler_state')
        if scaler_state is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(scaler_state)
        self.step = int(checkpoint.get('step', 0))
        self.tokens_seen = int(checkpoint.get('tokens_seen', 0))
        self.elapsed_wall_time = float(checkpoint.get('elapsed_wall_time', 0.0))

    def get_log_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step,
            'train_loss': self.last_train_loss,
            'lr': self.scheduler.get_lr(self.tokens_seen),
            'tokens_seen': self.tokens_seen,
            'grad_norm': self.last_grad_norm,
        }

    def train(self):
        """Full training loop until max_tokens reached."""
        print(f"Training for {self.config.max_tokens:,} tokens")
        while self.tokens_seen < self.config.max_tokens:
            loss = self.train_one_step()
            val_loss: Optional[float] = None

            if self.step % self.config.log_interval == 0:
                now = time.time()
                elapsed = max(now - self.last_log_time, 1e-8)
                delta_tokens = self.tokens_seen - self.last_log_tokens
                tokens_per_sec = delta_tokens / elapsed
                wall_time = self.elapsed_wall_time + (now - self.start_time)
                print(
                    f"step={self.step} | loss={loss:.4f} | "
                    f"tokens={self.tokens_seen:,} | "
                    f"lr={self.scheduler.get_lr(self.tokens_seen):.2e} | "
                    f"tok/s={tokens_per_sec:,.0f} | wall={wall_time:.1f}s"
                )
                self._append_log_row(loss, self.last_val_loss, tokens_per_sec, wall_time)
                self.last_log_tokens = self.tokens_seen
                self.last_log_time = now

            if self.step % self.config.val_interval == 0:
                val_loss = self.evaluate()
                self.last_val_loss = val_loss
                print(f"  val_loss={val_loss:.4f}")

            checkpoint_interval = self.config.batch_size * self.config.gradient_accumulation * 512
            if self.tokens_seen % self.config.checkpoint_every_tokens < checkpoint_interval:
                self.save_checkpoint(f"tokens_{self.tokens_seen // 1_000_000}M")
        self.save_checkpoint(
            f"completed_step_{self.step}_tokens_{self.tokens_seen // 1_000_000}M"
        )
