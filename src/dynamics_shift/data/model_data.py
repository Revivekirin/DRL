"""Separate replay roles and a real-only dynamics dataset snapshot."""
from dataclasses import dataclass
import numpy as np
from .replay_buffer import ReplayBuffer, TransitionBatch


class RealReplayBuffer(ReplayBuffer):
    source = "real_source"

    def sample_for_model(self, max_samples: int, rng: np.random.Generator) -> TransitionBatch:
        """Bounded snapshot without replacement; do not copy the entire replay."""
        if max_samples <= 0 or not len(self):
            raise ValueError("Need real samples and positive max_samples")
        indices = rng.choice(len(self), size=min(len(self), max_samples), replace=False)
        return TransitionBatch(**{key: values[indices] for key, values in self._arrays.items()})


class ModelReplayBuffer(ReplayBuffer):
    source = "synthetic_source_model"

    def clear(self) -> None:
        self.size = self.position = 0


@dataclass(frozen=True)
class ModelDataset:
    inputs: np.ndarray
    targets: np.ndarray
    train_indices: np.ndarray
    holdout_indices: np.ndarray

    @classmethod
    def from_real_replay(cls, replay: RealReplayBuffer, rng: np.random.Generator,
                         holdout_ratio: float, max_samples: int) -> "ModelDataset":
        if not isinstance(replay, RealReplayBuffer):
            raise TypeError("Dynamics fitting requires RealReplayBuffer, never synthetic replay")
        if len(replay) < 3 or not 0 < holdout_ratio < 1 or max_samples < 3:
            raise ValueError("Need at least 3 real transitions and a valid holdout ratio")
        batch = replay.sample_for_model(max_samples, rng)
        obs = batch.obs
        inputs = np.concatenate((obs, batch.action), axis=-1)
        targets = np.concatenate((batch.next_obs - obs, batch.reward), axis=-1)
        n_holdout = min(len(obs) - 2, max(1, int(len(obs) * holdout_ratio)))
        # Snapshot sampling itself is shuffled and without replacement.
        return cls(inputs, targets, np.arange(n_holdout, len(obs)), np.arange(n_holdout))
