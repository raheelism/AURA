import numpy as np
import pytest
import torch
from pathlib import Path
from data.dataloader import MemmapDataset, create_memmap_file, get_dataloader


def test_create_and_load_memmap(tmp_path):
    token_ids = list(range(1000))
    bin_path = str(tmp_path / "test.bin")
    create_memmap_file(token_ids, bin_path)
    assert Path(bin_path).exists()
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    assert len(data) == 1000
    assert list(data[:5]) == [0, 1, 2, 3, 4]


def test_dataset_returns_correct_shapes(tmp_path):
    token_ids = list(range(2000))
    bin_path = str(tmp_path / "test.bin")
    create_memmap_file(token_ids, bin_path)
    dataset = MemmapDataset(bin_path, seq_len=512)
    x, y = dataset[0]
    assert x.shape == (512,)
    assert y.shape == (512,)
    assert x.dtype in (torch.int64, torch.int32, torch.long)


def test_dataset_y_is_x_shifted_by_one(tmp_path):
    token_ids = list(range(2000))
    bin_path = str(tmp_path / "test.bin")
    create_memmap_file(token_ids, bin_path)
    dataset = MemmapDataset(bin_path, seq_len=10)
    x, y = dataset[0]
    assert list(x.numpy()) == list(range(10))
    assert list(y.numpy()) == list(range(1, 11))


def test_dataloader_batches_correctly(tmp_path):
    token_ids = list(range(10000))
    bin_path = str(tmp_path / "test.bin")
    create_memmap_file(token_ids, bin_path)
    dataset = MemmapDataset(bin_path, seq_len=32)
    loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (4, 32)
    assert batch_y.shape == (4, 32)
