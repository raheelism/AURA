import sentencepiece as spm
from typing import Iterable, List


def train_tokenizer(texts: Iterable[str], vocab_size: int, output_path: str) -> None:
    """Train a BPE SentencePiece tokenizer on the given texts.

    `texts` may be any finite iterable; it is streamed into SentencePiece.
    """
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(texts),
        model_prefix=output_path.replace(".model", ""),
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        character_coverage=0.9995,
        num_threads=4,
        input_sentence_size=1_000_000,   # cap sentences loaded into RAM
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
    )


def load_tokenizer(model_path: str) -> spm.SentencePieceProcessor:
    """Load a trained SentencePiece tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def encode(sp: spm.SentencePieceProcessor, text: str) -> List[int]:
    """Encode text to token IDs."""
    return sp.encode(text, out_type=int)


def decode(sp: spm.SentencePieceProcessor, ids: List[int]) -> str:
    """Decode token IDs to text."""
    return sp.decode(ids)
