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
    dataset_name: str, output_path: str, vocab_size: int = 8192, n_texts: int = 200_000
):
    """
    Train BPE tokenizer by streaming texts directly — no list accumulation.
    Uses a generator so RAM stays ~constant regardless of n_texts.
    """
    from data.tokenizer import train_tokenizer

    def text_generator():
        count = 0
        for text in stream_texts(dataset_name):
            yield text
            count += 1
            if count >= n_texts:
                break

    print(f"Training tokenizer (vocab_size={vocab_size}) on up to {n_texts:,} streamed texts...")
    train_tokenizer(list(text_generator()), vocab_size=vocab_size, output_path=output_path)
    print(f"Tokenizer saved to {output_path}")


def prepare_dataset(
    dataset_name: str,
    output_dir: str,
    tokenizer_path: str,
    n_tokens_approx: int,
    val_fraction: float = 0.01,
    chunk_size: int = 100_000,
):
    """
    Tokenize and write dataset to binary memmap files.

    Streams text → tokenizes → writes directly to pre-allocated memmap in chunks.
    Never holds more than `chunk_size` tokens in RAM at once, so memory usage
    stays constant regardless of dataset size.
    """
    from data.tokenizer import load_tokenizer, encode
    sp = load_tokenizer(tokenizer_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_path = os.path.join(output_dir, f"{dataset_name}_train.bin")
    val_path = os.path.join(output_dir, f"{dataset_name}_val.bin")

    val_budget = int(n_tokens_approx * val_fraction)
    train_budget = n_tokens_approx - val_budget

    # Pre-allocate memmap files at full size (2 bytes per token via uint16)
    print(f"Pre-allocating {train_budget:,} token train file ({train_budget*2/1e9:.2f} GB)...")
    train_fp = np.memmap(train_path, dtype=np.uint16, mode='w+', shape=(train_budget,))

    print(f"Pre-allocating {val_budget:,} token val file ({val_budget*2/1e9:.2f} GB)...")
    val_fp = np.memmap(val_path, dtype=np.uint16, mode='w+', shape=(val_budget,))

    train_cursor = 0
    val_cursor = 0
    chunk: list = []

    def flush_chunk(buf: list, fp: np.memmap, cursor: int, budget: int):
        """Write buf to fp starting at cursor, respecting budget. Returns new cursor."""
        if not buf:
            return cursor
        arr = np.array(buf, dtype=np.uint16)
        space = budget - cursor
        write_n = min(len(arr), space)
        fp[cursor:cursor + write_n] = arr[:write_n]
        return cursor + write_n

    print(f"Tokenizing {dataset_name} (target: {n_tokens_approx:,} tokens)...")
    pbar = tqdm(total=n_tokens_approx, unit='tok')

    for text in stream_texts(dataset_name):
        ids = encode(sp, text) + [sp.eos_id()]

        for tok in ids:
            if val_cursor < val_budget:
                chunk.append(tok)
                if len(chunk) >= chunk_size:
                    val_cursor = flush_chunk(chunk, val_fp, val_cursor, val_budget)
                    pbar.update(len(chunk))
                    chunk = []
            elif train_cursor < train_budget:
                chunk.append(tok)
                if len(chunk) >= chunk_size:
                    train_cursor = flush_chunk(chunk, train_fp, train_cursor, train_budget)
                    pbar.update(len(chunk))
                    chunk = []
            else:
                break
        else:
            continue
        break

    # Flush remaining tokens
    if chunk:
        if val_cursor < val_budget:
            val_cursor = flush_chunk(chunk, val_fp, val_cursor, val_budget)
        else:
            train_cursor = flush_chunk(chunk, train_fp, train_cursor, train_budget)
        pbar.update(len(chunk))

    pbar.close()

    # Truncate to actual written size
    val_fp.flush()
    train_fp.flush()
    del val_fp, train_fp

    if train_cursor < train_budget:
        # Truncate file to actual written tokens
        actual = np.memmap(train_path, dtype=np.uint16, mode='r+', shape=(train_budget,))
        final = np.array(actual[:train_cursor])
        del actual
        fp2 = np.memmap(train_path, dtype=np.uint16, mode='w+', shape=(train_cursor,))
        fp2[:] = final
        fp2.flush()
        del fp2

    print(f"Done. Train: {train_cursor:,} | Val: {val_cursor:,} tokens.")
    print(f"  Train: {train_path}  ({train_cursor*2/1e6:.1f} MB)")
    print(f"  Val:   {val_path}  ({val_cursor*2/1e6:.1f} MB)")
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
