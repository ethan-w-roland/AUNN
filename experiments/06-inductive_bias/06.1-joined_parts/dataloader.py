"""
@author: ethan-w-roland
@date: 2025-07-20
@desc: DataLoader for memory-mapped token files
"""

import torch
import numpy as np
from typing import Optional, Union
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler


class TokenDataset(Dataset):
    """PyTorch Dataset for memory-mapped token files."""

    def __init__(self, filename: str, T: int):
        self.filename = filename
        self.T = T

        # Load tokens as memory map
        self.tokens = np.memmap(filename, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.tokens)

        # The number of sequences that can be formed
        self.num_sequences = (self.num_tokens - 1) // self.T

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Get sequence of T+1 tokens
        buf = self.tokens[idx * self.T : (idx + 1) * self.T]

        # Convert to tensor
        return torch.from_numpy(buf.astype(np.int64))


class DataLoader:
    """A simplified DataLoader that uses PyTorch's built-in functionality."""

    def __init__(
        self,
        filename: str,
        B: int,
        T: int,
        process_rank: int = 0,
        num_processes: int = 1,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        self.B = B
        self.T = T
        self.device = device
        self.dataset = TokenDataset(filename, T)

        self.sampler: DistributedSampler = DistributedSampler(
            self.dataset,
            num_replicas=num_processes,
            rank=process_rank,
            shuffle=False,
        )

        self.dataloader = TorchDataLoader(
            self.dataset,
            batch_size=B,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=True,
        )
        self.iterator = iter(self.dataloader)

    def reset(self, epoch: int = 0):
        """Resets the data iterator."""
        self.sampler.set_epoch(epoch)
        self.iterator = iter(self.dataloader)

    def next_batch(self) -> torch.Tensor:
        """Gets the next batch, automatically resetting the iterator if it's exhausted."""
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.reset()
            batch = next(self.iterator)

        if self.device is not None:
            batch = batch.to(self.device, non_blocking=True)
        return batch

    def __len__(self) -> int:
        return len(self.dataloader)
