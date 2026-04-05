"""
Automated ablation runner for NEXUS-AURORA.
Runs each ablation configuration, logs results to CSV, and prints comparison table.

Ablation schedule (from spec):
  Day 4:  LLaMA baseline, S-only (no R or V)
  Day 5:  S+R (K=1 fixed), S+R (adaptive K)
  Day 6:  S+V only, Full AURORA (S+R+V, adaptive K)
  Day 7:  Bridge top_k ablation (k=1,4,8,16)
  Day 8:  K_max ablation (max recursion 2,4,6)
  Day 9:  n_slots ablation (16,32,64)
  Day 10: Optimizer ablation (Muon vs AdamW)

Usage:
  python -m evaluation.ablation_runner --config config/nexus_aurora_v1.yaml \
      --data data/train.bin --val data/val.bin \
      --ablations baseline aurora_full --output results/ablation_day4.csv
"""
import argparse
import csv
import os
import time
import yaml
import torch
from pathlib import Path

ABLATION_CONFIGS = {
    'baseline':           {'model_type': 'llama'},
    'aurora_s_only':      {'model_type': 'aurora', 'use_reasoning': False, 'use_verify': False},
    'aurora_s_r_k1':      {'model_type': 'aurora', 'use_reasoning': True,  'use_verify': False, 'fixed_k': 1},
    'aurora_s_r_adaptive':{'model_type': 'aurora', 'use_reasoning': True,  'use_verify': False, 'fixed_k': None},
    'aurora_s_v':         {'model_type': 'aurora', 'use_reasoning': False, 'use_verify': True},
    'aurora_full':        {'model_type': 'aurora', 'use_reasoning': True,  'use_verify': True,  'fixed_k': None},
    'aurora_topk1':       {'model_type': 'aurora', 'bridge_top_k': 1},
    'aurora_topk4':       {'model_type': 'aurora', 'bridge_top_k': 4},
    'aurora_topk8':       {'model_type': 'aurora', 'bridge_top_k': 8},
    'aurora_topk16':      {'model_type': 'aurora', 'bridge_top_k': 16},
    'aurora_kmax2':       {'model_type': 'aurora', 'max_k': 2},
    'aurora_kmax4':       {'model_type': 'aurora', 'max_k': 4},
    'aurora_kmax6':       {'model_type': 'aurora', 'max_k': 6},
    'aurora_slots16':     {'model_type': 'aurora', 'n_reasoning_slots': 16},
    'aurora_slots32':     {'model_type': 'aurora', 'n_reasoning_slots': 32},
    'aurora_slots64':     {'model_type': 'aurora', 'n_reasoning_slots': 64},
    'aurora_adamw':       {'model_type': 'aurora', 'optimizer': 'adamw'},
    'aurora_muon':        {'model_type': 'aurora', 'optimizer': 'muon'},
}


def build_model(base_config: dict, ablation_config: dict):
    model_type = ablation_config.get('model_type', 'aurora')
    if model_type == 'llama':
        from model.baseline import LLaMABaseline, LLaMAConfig
        m = base_config['model']
        return LLaMABaseline(LLaMAConfig(
            n_layers=m.get('n_blocks', 8),
            d_model=m['d_surface'],
            n_heads=m['n_heads_surface'],
            n_kv_heads=m['n_kv_heads_surface'],
            d_ffn=3300,  # matched for ~50M params
            vocab_size=m['vocab_size'],
            max_seq_len=m['max_seq_len'],
        ))
    else:
        from model.model import NexusAurora, AuroraConfig
        defaults = AuroraConfig()
        overrides = {k: v for k, v in ablation_config.items()
                     if k not in ('model_type', 'use_reasoning', 'use_verify', 'fixed_k', 'optimizer')}
        cfg = AuroraConfig(**{
            k: base_config['model'].get(k, getattr(defaults, k))
            for k in defaults.__dataclass_fields__
        })
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return NexusAurora(cfg)


def run_ablation(
    name: str,
    base_config: dict,
    ablation_config: dict,
    train_path: str,
    val_path: str,
    output_dir: str,
    max_tokens: int = 200_000_000,
) -> dict:
    """Train a single ablation variant and return results dict."""
    from data.dataloader import MemmapDataset, get_dataloader
    from evaluation.perplexity import compute_bpb
    from training.trainer import Trainer, TrainerConfig

    print(f"\n{'='*60}\nRunning ablation: {name}\nConfig overrides: {ablation_config}\n{'='*60}")

    model = build_model(base_config, ablation_config)
    seq_len = base_config['model']['max_seq_len']
    train_ds = MemmapDataset(train_path, seq_len=seq_len)
    val_ds = MemmapDataset(val_path, seq_len=seq_len)

    ckpt_dir = os.path.join(output_dir, name)
    trainer_config = TrainerConfig(
        batch_size=base_config['training']['batch_size'],
        gradient_accumulation=base_config['training']['gradient_accumulation'],
        max_lr=base_config['training']['max_lr'],
        min_lr=base_config['training']['min_lr'],
        warmup_tokens=10_000_000,
        stable_tokens=max_tokens - 20_000_000,
        decay_tokens=10_000_000,
        grad_clip=base_config['training']['grad_clip'],
        weight_decay=base_config['training']['weight_decay'],
        checkpoint_dir=ckpt_dir,
        log_interval=200,
        val_interval=5000,
        max_tokens=max_tokens,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=base_config['training']['dtype'],
    )

    trainer = Trainer(model, train_ds, val_ds, trainer_config)
    t0 = time.time()
    trainer.train()
    wall_time = time.time() - t0

    device = torch.device(trainer_config.device)
    dtype = torch.float16 if trainer_config.dtype == 'float16' else torch.float32
    val_loader = get_dataloader(val_ds, batch_size=16, shuffle=False)
    bpb = compute_bpb(model, val_loader, device, dtype)

    result = {
        'name': name,
        'bpb': bpb,
        'wall_time_s': wall_time,
        'params': sum(p.numel() for p in model.parameters()),
    }
    result.update({k: v for k, v in ablation_config.items() if k != 'model_type'})
    print(f"Result: BPB={bpb:.4f} | Time={wall_time/3600:.2f}h")
    return result


def main():
    parser = argparse.ArgumentParser(description='Run NEXUS-AURORA ablations')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--data', required=True, help='Path to train .bin file')
    parser.add_argument('--val', required=True, help='Path to val .bin file')
    parser.add_argument('--ablations', nargs='+', default=['baseline', 'aurora_full'],
                        help='Ablation names to run')
    parser.add_argument('--output', default='results/ablation_results.csv')
    parser.add_argument('--max_tokens', type=int, default=200_000_000)
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.ablations:
        if name not in ABLATION_CONFIGS:
            print(f"Unknown ablation '{name}'. Available: {list(ABLATION_CONFIGS.keys())}")
            continue
        result = run_ablation(name, base_config, ABLATION_CONFIGS[name],
                              args.data, args.val, output_dir, args.max_tokens)
        results.append(result)

    if results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print("\n" + "="*70)
    print(f"{'Ablation':<30} {'BPB':>8} {'Time(h)':>10} {'Params':>12}")
    print("="*70)
    for r in sorted(results, key=lambda x: x['bpb']):
        print(f"{r['name']:<30} {r['bpb']:>8.4f} {r['wall_time_s']/3600:>10.2f} {r['params']:>12,}")
    print("="*70)


if __name__ == '__main__':
    main()
