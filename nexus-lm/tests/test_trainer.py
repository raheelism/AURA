import torch
import tempfile
import os
import pytest
from pathlib import Path
from model.model import NexusAurora, AuroraConfig
from training.trainer import Trainer, TrainerConfig
from data.dataloader import MemmapDataset, create_memmap_file


def make_tiny_aurora():
    return NexusAurora(AuroraConfig(
        d_surface=32, d_reasoning=16, d_verify=8,
        n_reasoning_slots=4, n_blocks=1,
        n_heads_surface=2, n_kv_heads_surface=1,
        n_heads_reasoning=2,
        d_ffn_pattern=32, d_ffn_semantic=64, d_ffn_reasoning=32,
        local_window=4, bridge_top_k=2,
        cope_positions=4, vocab_size=64, max_seq_len=16,
        surprise_loss_weight=0.1, difficulty_entropy_weight=0.01,
    ))


def make_tiny_dataset(tmp_path, n_tokens=2000, vocab_size=64):
    import random
    tokens = [random.randint(0, vocab_size - 1) for _ in range(n_tokens)]
    path = str(tmp_path / "data.bin")
    create_memmap_file(tokens, path)
    return MemmapDataset(path, seq_len=16)


def test_trainer_runs_one_step(tmp_path):
    model = make_tiny_aurora()
    dataset = make_tiny_dataset(tmp_path)
    config = TrainerConfig(
        batch_size=2, gradient_accumulation=1,
        max_lr=1e-3, min_lr=1e-4,
        warmup_tokens=100, stable_tokens=800, decay_tokens=100,
        grad_clip=1.0, weight_decay=0.1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        log_interval=1,
        val_interval=100,
        max_tokens=32,
        device='cpu',
        dtype='float32',
    )
    trainer = Trainer(model, dataset, dataset, config)
    initial_loss = trainer.train_one_step()
    assert isinstance(initial_loss, float)
    assert initial_loss > 0


def test_trainer_saves_checkpoint(tmp_path):
    model = make_tiny_aurora()
    dataset = make_tiny_dataset(tmp_path)
    ckpt_dir = str(tmp_path / "ckpts")
    config = TrainerConfig(
        batch_size=2, gradient_accumulation=1,
        max_lr=1e-3, min_lr=1e-4,
        warmup_tokens=100, stable_tokens=800, decay_tokens=100,
        grad_clip=1.0, weight_decay=0.1,
        checkpoint_dir=ckpt_dir,
        log_interval=1, val_interval=10, max_tokens=64,
        device='cpu', dtype='float32',
    )
    trainer = Trainer(model, dataset, dataset, config)
    trainer.save_checkpoint("step_0")
    assert Path(ckpt_dir).exists()
    files = list(Path(ckpt_dir).glob("*.pt"))
    assert len(files) > 0


def test_trainer_log_dict_has_required_keys(tmp_path):
    model = make_tiny_aurora()
    dataset = make_tiny_dataset(tmp_path)
    config = TrainerConfig(
        batch_size=2, gradient_accumulation=1,
        max_lr=1e-3, min_lr=1e-4,
        warmup_tokens=100, stable_tokens=800, decay_tokens=100,
        grad_clip=1.0, weight_decay=0.1,
        checkpoint_dir=str(tmp_path),
        log_interval=1, val_interval=100, max_tokens=32,
        device='cpu', dtype='float32',
    )
    trainer = Trainer(model, dataset, dataset, config)
    log = trainer.get_log_dict()
    required = ['step', 'train_loss', 'lr', 'tokens_seen', 'grad_norm']
    for key in required:
        assert key in log, f"Missing key: {key}"
