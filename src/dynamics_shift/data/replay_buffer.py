"""NumPy ring storage; batches carry no assumptions about data provenance."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TransitionBatch:
    obs: NDArray
    action: NDArray
    reward: NDArray
    next_obs: NDArray
    terminated: NDArray
    truncated: NDArray


class ReplayBuffer:
    """Fixed-capacity replay with explicit terminal and time-limit flags."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int) -> None:
        if any(type(n) is not int or n <= 0 for n in (capacity, obs_dim, action_dim)):
            raise ValueError("capacity and dimensions must be positive integers")
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)
        self._arrays = {
            "obs": np.empty((capacity, obs_dim), dtype=np.float32),
            "action": np.empty((capacity, action_dim), dtype=np.float32),
            "reward": np.empty((capacity, 1), dtype=np.float32),
            "next_obs": np.empty((capacity, obs_dim), dtype=np.float32),
            "terminated": np.empty((capacity, 1), dtype=np.bool_),
            "truncated": np.empty((capacity, 1), dtype=np.bool_),
        }

    def __len__(self) -> int:
        return self.size

    def add(self, obs: NDArray, action: NDArray, reward: float,
            next_obs: NDArray, terminated: bool, truncated: bool) -> None:
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
            raise ValueError("terminated and truncated must be booleans")
        values = dict(obs=obs, action=action, reward=[reward], next_obs=next_obs,
                      terminated=[terminated], truncated=[truncated])
        converted = {}
        for name, value in values.items():
            array = np.asarray(value, dtype=self._arrays[name].dtype)
            if array.shape != self._arrays[name].shape[1:] or not np.isfinite(array).all():
                raise ValueError(f"Invalid shape or nonfinite value for {name}")
            converted[name] = array
        for name, value in converted.items():
            self._arrays[name][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample with replacement; returned arrays do not alias replay storage."""
        if type(batch_size) is not int or batch_size <= 0 or self.size == 0:
            raise ValueError("Need nonempty replay and positive batch_size")
        indices = self.rng.integers(self.size, size=batch_size)
        return TransitionBatch(**{name: value[indices] for name, value in self._arrays.items()})
