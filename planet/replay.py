"""Fixed-size episode replay held in pre-allocated tensors.

Episodes all have the same length, so the store is a single block indexed by
(episode, time). By default the block lives in VRAM: the full 1000 episodes take
about 3.1 GB of the 4080 Super's 16 GB, and sampling becomes GPU indexing rather
than a 30 MB CPU-to-GPU copy on each update.

Alignment: actions[t] led to frames[t] and rewards[t]. Index 0 holds the reset
frame with a zero action and zero reward.
"""
import numpy as np
import torch


class Replay:
    def __init__(self, capacity: int, steps: int, action_dim: int, device: torch.device, storage: str):
        self.device = device
        self.size = 0
        self.length = steps + 1
        self.storage = self._allocate(capacity, action_dim, device, storage)

    def _allocate(self, capacity: int, action_dim: int, device: torch.device, storage: str) -> torch.device:
        target = device if storage == "auto" else torch.device(storage)
        try:
            self._create(capacity, action_dim, target)
        except torch.OutOfMemoryError:
            if storage != "auto":
                raise
            torch.cuda.empty_cache()
            print("Replay does not fit in VRAM right now, using RAM (slower sampling).")
            target = torch.device("cpu")
            self._create(capacity, action_dim, target)
        return target

    def _create(self, capacity: int, action_dim: int, target: torch.device) -> None:
        shape = (capacity, self.length)
        self.frames = torch.zeros(shape + (3, 64, 64), dtype=torch.uint8, device=target)
        self.actions = torch.zeros(shape + (action_dim,), device=target)
        self.rewards = torch.zeros(shape, device=target)

    def add(self, frames: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        index = self.size
        self.frames[index] = torch.from_numpy(frames).to(self.storage)
        self.actions[index] = torch.from_numpy(actions).to(self.storage)
        self.rewards[index] = torch.from_numpy(rewards).to(self.storage)
        self.size += 1

    def sample(self, batch_size: int, chunk_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform random chunks, returned on the training device."""
        target = self.storage
        episodes = torch.randint(self.size, (batch_size, 1), device=target)
        starts = torch.randint(self.length - chunk_length + 1, (batch_size, 1), device=target)
        steps = starts + torch.arange(chunk_length, device=target)
        return (
            self.frames[episodes, steps].to(self.device, non_blocking=True),
            self.actions[episodes, steps].to(self.device, non_blocking=True),
            self.rewards[episodes, steps].to(self.device, non_blocking=True),
        )
