"""
Download and preprocess training data for NEXUS-AURORA.

Downloads FineWeb-Edu (subset) or TinyStories from HuggingFace.
Trains a SentencePiece BPE tokenizer (vocab=8192) if not already present.
Pre-tokenizes datasets to binary memmap files (uint16).

Usage:
  # Train tokenizer and prepare FineWeb-Edu (1.5B tokens)
  python data/prepare.py --output_dir data/ --dataset fineweb_edu --n_tokens 1500000000

  # Prepare TinyStories (faster, good for debugging)
  python data/prepare.py --output_dir data/ --dataset tinystories --n_tokens 100000000
"""
import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm


def stream_texts(dataset_name: str):
    """Stream text samples from HuggingFace datasets."""
    from datasets import load_dataset
    if dataset_name == 'fineweb_edu':
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        field = 'text'
    elif dataset_name == 'tinystories':
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        field = 'text'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'fineweb_edu' or 'tinystories'.")

    for sample in ds:
        yield sample[field]


def train_tokenizer_on_dataset(
    dataset_name: str, output_path: str, vocab_size: int = 8192, n_texts: int = 100_000
):
    """Train BPE tokenizer on a sample of the dataset."""
    from data.tokenizer import train_tokenizer
    print(f"Collecting {n_texts:,} texts for tokenizer training...")
    texts = []
    for text in stream_texts(dataset_name):
        texts.append(text)
        if len(texts) >= n_texts:
            break
    print(f"Training tokenizer (vocab_size={vocab_size}) on {len(texts):,} texts...")
    train_tokenizer(texts, vocab_size=vocab_size, output_path=output_path)
    print(f"Tokenizer saved to {output_path}")


def prepare_dataset(
    dataset_name: str,
    output_dir: str,
    tokenizer_path: str,
    n_tokens_approx: int,
    val_fraction: float = 0.01,
):
    """Tokenize and write dataset to binary memmap files."""
    from data.tokenizer import load_tokenizer, encode
    sp = load_tokenizer(tokenizer_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_path = os.path.join(output_dir, f"{dataset_name}_train.bin")
    val_path = os.path.join(output_dir, f"{dataset_name}_val.bin")

    val_budget = int(n_tokens_approx * val_fraction)
    train_budget = n_tokens_approx - val_budget

    train_tokens = []
    val_tokens = []

    print(f"Tokenizing {dataset_name} (target: {n_tokens_approx:,} tokens)...")
    for text in tqdm(stream_texts(dataset_name)):
        ids = encode(sp, text) + [sp.eos_id()]
        if len(val_tokens) < val_budget:
            val_tokens.extend(ids)
        elif len(train_tokens) < train_budget:
            train_tokens.extend(ids)
        else:
            break

    print(f"Writing {len(train_tokens):,} train tokens to {train_path}")
    fp = np.memmap(train_path, dtype=np.uint16, mode='w+', shape=(len(train_tokens),))
    fp[:] = np.array(train_tokens, dtype=np.uint16)
    fp.flush()
    del fp

    print(f"Writing {len(val_tokens):,} val tokens to {val_path}")
    fp = np.memmap(val_path, dtype=np.uint16, mode='w+', shape=(len(val_tokens),))
    fp[:] = np.array(val_tokens, dtype=np.uint16)
    fp.flush()
    del fp

    print(f"Done. Train: {len(train_tokens):,} | Val: {len(val_tokens):,} tokens.")
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description='Prepare data for NEXUS-AURORA training')
    parser.add_argument('--output_dir', default='data/', help='Directory to write output files')
    parser.add_argument('--dataset', default='fineweb_edu',
                        choices=['fineweb_edu', 'tinystories'])
    parser.add_argument('--n_tokens', type=int, default=1_500_000_000,
                        help='Approximate number of tokens to prepare')
    parser.add_argument('--vocab_size', type=int, default=8192)
    parser.add_argument('--tokenizer_path', default='data/tokenizer.model')
    parser.add_argument('--train_tokenizer', action='store_true',
                        help='Force re-training the tokenizer even if it exists')
    args = parser.parse_args()

    if args.train_tokenizer or not Path(args.tokenizer_path).exists():
        train_tokenizer_on_dataset(args.dataset, args.tokenizer_path, args.vocab_size)

    prepare_dataset(
        args.dataset, args.output_dir, args.tokenizer_path, args.n_tokens
    )


if __name__ == '__main__':
    main()
