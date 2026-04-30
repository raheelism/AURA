from pathlib import Path

import numpy as np

from data.prepare import prepare_dataset_from_texts
from data.tokenizer import train_tokenizer


def test_prepare_dataset_from_texts_smoke(tmp_path):
    texts = [
        "hello world",
        "tiny smoke test",
        "nexus aurora",
        "streaming data prep",
        "reasoning slots",
        "halting and routing",
    ]

    tokenizer_path = str(tmp_path / "tok.model")
    train_tokenizer(texts, vocab_size=32, output_path=tokenizer_path)

    train_path = str(tmp_path / "train.bin")
    val_path = str(tmp_path / "val.bin")
    train_path, val_path, train_tokens, val_tokens = prepare_dataset_from_texts(
        texts,
        train_path=train_path,
        val_path=val_path,
        tokenizer_path=tokenizer_path,
        n_tokens_approx=64,
        val_fraction=0.25,
        chunk_size=8,
    )

    assert Path(train_path).exists()
    assert Path(val_path).exists()
    train = np.memmap(train_path, dtype=np.uint16, mode='r')
    val = np.memmap(val_path, dtype=np.uint16, mode='r')
    assert len(train) == train_tokens
    assert len(val) == val_tokens
    assert len(train) > 0
    assert len(val) > 0
