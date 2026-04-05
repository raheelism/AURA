import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


def create_memmap_file(token_ids: List[int], output_path: str) -> None:
    """Write token IDs to a memory-mapped binary file (uint16)."""
    arr = np.array(token_ids, dtype=np.uint16)
    fp = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(len(arr),))
    fp[:] = arr[:]
    fp.flush()


class MemmapDataset(Dataset):
    """Memory-mapped dataset for zero-copy token loading."""

    def __init__(self, bin_path: str, seq_len: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        self.n_samples = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.data[start: start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def get_dataloader(
    dataset: MemmapDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from a MemmapDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
