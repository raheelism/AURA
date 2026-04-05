#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_root_to_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def cmd_data_prep(args: argparse.Namespace) -> None:
    _add_repo_root_to_path()
    import numpy as np
    from data.prepare import prepare_dataset, train_tokenizer_on_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer_path or str(output_dir / "tokenizer.model")

    if args.train_tokenizer or not Path(tokenizer_path).exists():
        train_tokenizer_on_dataset(args.dataset, tokenizer_path, vocab_size=args.vocab_size)
    else:
        print(f"Tokenizer already exists at {tokenizer_path}")

    train_path, val_path = prepare_dataset(
        dataset_name=args.dataset,
        output_dir=str(output_dir),
        tokenizer_path=tokenizer_path,
        n_tokens_approx=args.n_tokens,
        val_fraction=args.val_fraction,
    )

    if args.verify:
        train = np.memmap(train_path, dtype=np.uint16, mode="r")
        val = np.memmap(val_path, dtype=np.uint16, mode="r")
        print(f"Train tokens: {len(train):,}")
        print(f"Val tokens:   {len(val):,}")
        print(f"Token range:  [{train.min()}, {train.max()}]")

    print(json.dumps({"train_path": train_path, "val_path": val_path, "tokenizer_path": tokenizer_path}))


def cmd_train(args: argparse.Namespace) -> None:
    _add_repo_root_to_path()
    import torch
    from data.dataloader import MemmapDataset
    from model.model import AuroraConfig, NexusAurora
    from training.trainer import Trainer, TrainerConfig

    cfg = _load_yaml(args.config)

    model = NexusAurora(AuroraConfig(**cfg["model"]))
    train_ds = MemmapDataset(args.train_bin, seq_len=cfg["model"]["max_seq_len"])
    val_ds = MemmapDataset(args.val_bin, seq_len=cfg["model"]["max_seq_len"])

    trainer_cfg = TrainerConfig(
        batch_size=cfg["training"]["batch_size"],
        gradient_accumulation=cfg["training"]["gradient_accumulation"],
        max_lr=cfg["training"]["max_lr"],
        min_lr=cfg["training"]["min_lr"],
        warmup_tokens=cfg["training"]["warmup_tokens"],
        stable_tokens=cfg["training"]["stable_tokens"],
        decay_tokens=cfg["training"]["decay_tokens"],
        grad_clip=cfg["training"]["grad_clip"],
        weight_decay=cfg["training"]["weight_decay"],
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every_tokens=args.checkpoint_every_tokens,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        max_tokens=args.max_tokens,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=cfg["training"]["dtype"],
        surprise_loss_weight=cfg["training"]["surprise_loss_weight"],
        difficulty_entropy_weight=cfg["training"]["difficulty_entropy_weight"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Device: {trainer_cfg.device}")
    trainer = Trainer(model, train_ds, val_ds, trainer_cfg)
    trainer.train()


def cmd_ablations(args: argparse.Namespace) -> None:
    _add_repo_root_to_path()
    from evaluation.ablation_runner import ABLATION_CONFIGS, run_ablation

    cfg = _load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.ablations:
        if name not in ABLATION_CONFIGS:
            raise ValueError(f"Unknown ablation '{name}'. Available: {sorted(ABLATION_CONFIGS)}")
        results.append(
            run_ablation(
                name=name,
                base_config=cfg,
                ablation_config=ABLATION_CONFIGS[name],
                train_path=args.train_bin,
                val_path=args.val_bin,
                output_dir=str(output_dir),
                max_tokens=args.max_tokens,
            )
        )

    if not results:
        print("No ablation results to save.")
        return

    output_csv = output_dir / "ablation_results.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {output_csv}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    _add_repo_root_to_path()
    import torch
    from data.dataloader import MemmapDataset, get_dataloader
    from data.tokenizer import decode, encode, load_tokenizer
    from evaluation.perplexity import compute_bpb
    from evaluation.routing_analysis import analyze_routing
    from model.model import AuroraConfig, NexusAurora

    cfg = _load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NexusAurora(AuroraConfig(**cfg["model"]))
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"Loaded checkpoint: step={ckpt.get('step')} tokens={ckpt.get('tokens_seen')}")

    val_ds = MemmapDataset(args.val_bin, seq_len=cfg["model"]["max_seq_len"])
    val_loader = get_dataloader(val_ds, batch_size=args.eval_batch_size, shuffle=False)

    bpb = compute_bpb(model, val_loader, device, torch.float16 if device.type == "cuda" else torch.float32)
    routing = analyze_routing(model, val_loader, device, n_batches=args.routing_batches)

    print(f"BPB: {bpb:.4f}")
    print(
        f"Routing: K=1 {routing['k_1']:.3f} | K=2 {routing['k_2']:.3f} | "
        f"K=4 {routing['k_4']:.3f} | surprise_mean {routing['surprise_mean']:.4f}"
    )

    generated_text = None
    if args.tokenizer_path:
        sp = load_tokenizer(args.tokenizer_path)
        ids = torch.tensor([encode(sp, args.prompt)], device=device)
        with torch.no_grad():
            generated = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        prompt_len = len(ids[0])
        generated_ids = generated[0].tolist()[prompt_len:]
        generated_text = decode(sp, generated_ids)
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {generated_text}")

    metrics = {"bpb": bpb, **routing, "generated_text": generated_text}
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle command runner for NEXUS-AURORA")
    sub = parser.add_subparsers(dest="command", required=True)

    data_prep = sub.add_parser("data-prep", help="Prepare tokenizer and memmap bins")
    data_prep.add_argument("--dataset", default="fineweb_edu", choices=["fineweb_edu", "tinystories"])
    data_prep.add_argument("--output-dir", default="/kaggle/working/data")
    data_prep.add_argument("--tokenizer-path", default=None)
    data_prep.add_argument("--n-tokens", type=int, default=1_500_000_000)
    data_prep.add_argument("--vocab-size", type=int, default=8192)
    data_prep.add_argument("--val-fraction", type=float, default=0.01)
    data_prep.add_argument("--train-tokenizer", action="store_true")
    data_prep.add_argument("--verify", action="store_true")
    data_prep.set_defaults(func=cmd_data_prep)

    train = sub.add_parser("train", help="Run full training")
    train.add_argument("--config", default="/kaggle/working/nexus-lm/config/nexus_aurora_v1.yaml")
    train.add_argument("--train-bin", default="/kaggle/input/nexus-data/fineweb_edu_train.bin")
    train.add_argument("--val-bin", default="/kaggle/input/nexus-data/fineweb_edu_val.bin")
    train.add_argument("--checkpoint-dir", default="/kaggle/working/checkpoints")
    train.add_argument("--max-tokens", type=int, default=1_500_000_000)
    train.add_argument("--checkpoint-every-tokens", type=int, default=100_000_000)
    train.add_argument("--log-interval", type=int, default=100)
    train.add_argument("--val-interval", type=int, default=2000)
    train.set_defaults(func=cmd_train)

    ablations = sub.add_parser("ablations", help="Run selected ablations")
    ablations.add_argument("--config", default="/kaggle/working/nexus-lm/config/nexus_aurora_v1.yaml")
    ablations.add_argument("--train-bin", default="/kaggle/input/nexus-data/fineweb_edu_train.bin")
    ablations.add_argument("--val-bin", default="/kaggle/input/nexus-data/fineweb_edu_val.bin")
    ablations.add_argument("--output-dir", default="/kaggle/working/ablation_results")
    ablations.add_argument("--max-tokens", type=int, default=200_000_000)
    ablations.add_argument("--ablations", nargs="+", default=["baseline", "aurora_s_only"])
    ablations.set_defaults(func=cmd_ablations)

    evaluate = sub.add_parser("evaluate", help="Evaluate checkpoint and optional generation")
    evaluate.add_argument("--config", default="/kaggle/working/nexus-lm/config/nexus_aurora_v1.yaml")
    evaluate.add_argument("--ckpt-path", default="/kaggle/working/checkpoints/ckpt_tokens_1500M.pt")
    evaluate.add_argument("--val-bin", default="/kaggle/input/nexus-data/fineweb_edu_val.bin")
    evaluate.add_argument("--tokenizer-path", default="/kaggle/input/nexus-data/tokenizer.model")
    evaluate.add_argument("--eval-batch-size", type=int, default=16)
    evaluate.add_argument("--routing-batches", type=int, default=50)
    evaluate.add_argument("--prompt", default="The theory of relativity states that")
    evaluate.add_argument("--max-new-tokens", type=int, default=100)
    evaluate.add_argument("--temperature", type=float, default=0.8)
    evaluate.add_argument("--top-k", type=int, default=50)
    evaluate.add_argument("--output-json", default="/kaggle/working/eval_metrics.json")
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
