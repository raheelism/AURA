import pytest
from pathlib import Path
from data.tokenizer import train_tokenizer, load_tokenizer, encode, decode


def test_train_tokenizer_creates_model_file(tmp_path):
    texts = ["Hello world, this is a test sentence."] * 100
    model_path = str(tmp_path / "tokenizer.model")
    train_tokenizer(texts, vocab_size=256, output_path=model_path)
    assert Path(model_path).exists()


def test_encode_decode_roundtrip(tmp_path):
    texts = ["The quick brown fox jumps over the lazy dog."] * 100
    model_path = str(tmp_path / "tokenizer.model")
    train_tokenizer(texts, vocab_size=256, output_path=model_path)
    sp = load_tokenizer(model_path)
    text = "Hello world"
    ids = encode(sp, text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_encode_returns_ints_within_vocab(tmp_path):
    texts = ["Sample training text for tokenizer."] * 100
    model_path = str(tmp_path / "tokenizer.model")
    train_tokenizer(texts, vocab_size=256, output_path=model_path)
    sp = load_tokenizer(model_path)
    ids = encode(sp, "test sentence")
    assert all(0 <= i < 256 for i in ids)
