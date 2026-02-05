from __future__ import annotations

import os
import threading
import queue
import warnings
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

from ..utils import try_import_mlx, as_mx_array


@dataclass
class PackingConfig:
    seq_len: int
    eos_id: int
    pad_id: int


class MMapTokenDataset:
    """Memory-mapped dataset of token ids stored in a plain .npy/.bin array.

    Supports iterating over fixed-length sequences with EOS padding and ignores
    final short tail unless `drop_last=False`.
    """

    def __init__(
        self,
        path: str,
        *,
        seq_len: int,
        eos_id: int,
        pad_id: int,
        drop_last: bool = True,
    ) -> None:
        import numpy as np

        self.seq_len = int(seq_len)
        self.eos_id = int(eos_id)
        self.pad_id = int(pad_id)
        self.drop_last = drop_last

        # accept .npy or raw int32 .bin with sidecar .shape
        if path.endswith(".npy"):
            self._arr = np.load(path, mmap_mode="r")
        else:
            # raw int32 tokens with .shape file containing length
            with open(path + ".shape", "r") as f:
                n = int(f.read().strip())
            self._arr = np.memmap(path, dtype="int32", mode="r", shape=(n,))

        self._n = len(self._arr)

    def __len__(self):
        import math

        n_seq = self._n // self.seq_len
        if not self.drop_last and (self._n % self.seq_len) != 0:
            n_seq += 1
        return n_seq

    def __iter__(self) -> Iterator[List[int]]:
        step = self.seq_len
        n = self._n
        for start in range(0, n, step):
            end = min(start + step, n)
            if end - start < step and self.drop_last:
                break
            chunk = self._arr[start:end].tolist()
            if end - start < step:
                chunk = chunk + [self.eos_id] + [self.pad_id] * (step - (end - start) - 1)
            yield chunk[:step]


class PrefetchDataLoader:
    """Background prefetch wrapper around an iterable of token batches.

    Example:
        ds = MMapTokenDataset(...)
        loader = PrefetchDataLoader(ds, batch_size=4)
        for batch in loader:  # yields mx.array [B, T]
            ...
    """

    def __init__(
        self,
        dataset: Iterable[List[int]],
        *,
        batch_size: int,
        prefetch: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self._is_sequence = hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")
        self.batch_size = int(batch_size)
        self.prefetch = max(1, int(prefetch))
        self.seed = seed
        self._q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        if not self._is_sequence:
            warnings.warn(
                "PrefetchDataLoader: dataset is iterable-only; shuffle is disabled without sequence semantics.",
                RuntimeWarning,
            )

    def _worker(self):
        import random

        rng = random.Random(self.seed)
        if self._is_sequence:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
            rng.shuffle(indices)
            seq_iter = (self.dataset[i] for i in indices)  # type: ignore[index]
        else:
            seq_iter = iter(self.dataset)
        mx, _ = try_import_mlx()
        batch: List[List[int]] = []
        for seq in seq_iter:
            batch.append(seq)
            if len(batch) == self.batch_size:
                arr = as_mx_array(batch, dtype=mx.int32)
                self._q.put(arr)
                batch = []
            if self._stop.is_set():
                break
        if batch:
            arr = as_mx_array(batch, dtype=mx.int32)
            self._q.put(arr)
        self._q.put(None)

    def __iter__(self):
        self._stop.clear()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()
        while True:
            item = self._q.get()
            if item is None:
                break
            yield item
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=1.0)
