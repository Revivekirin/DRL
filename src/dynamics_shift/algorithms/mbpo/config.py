"""Explicit MBPO model and rollout settings."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import yaml
from dynamics_shift.algorithms.sac.config import SACConfig
from dynamics_shift.config import EnvConfig, DynamicsConfig, _mapping
from dynamics_shift.experiments.config import TrainingConfig, RunConfig, TrackingConfig


@dataclass(frozen=True)
class MBPOConfig:
    ensemble_size: int = 7
    elite_size: int = 5
    model_hidden_dims: tuple[int, ...] = (200, 200, 200, 200)
    model_learning_rate: float = 0.001
    model_batch_size: int = 256
    model_train_frequency: int = 250
    model_max_epochs: int = 20
    model_patience: int = 5
    model_max_samples: int = 100000
    holdout_ratio: float = 0.2
    rollout_horizon: int = 1
    rollout_batch_size: int = 10000
    model_replay_capacity: int = 100000
    real_ratio: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_hidden_dims", tuple(self.model_hidden_dims))
        for name in ("ensemble_size", "elite_size", "model_batch_size", "model_train_frequency",
                     "model_max_epochs", "model_patience", "model_max_samples", "rollout_horizon",
                     "rollout_batch_size", "model_replay_capacity"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.model_hidden_dims or any(type(n) is not int or n <= 0 for n in self.model_hidden_dims):
            raise ValueError("model_hidden_dims must be positive integers")
        if self.elite_size > self.ensemble_size or self.model_max_samples < 3:
            raise ValueError("Invalid elite size or model_max_samples")
        if not math.isfinite(self.model_learning_rate) or self.model_learning_rate <= 0:
            raise ValueError("model_learning_rate must be positive")
        if not 0 < self.holdout_ratio < 1 or not 0 < self.real_ratio < 1:
            raise ValueError("holdout_ratio and real_ratio must lie in (0, 1)")


@dataclass(frozen=True)
class MBPORunConfig:
    name: str = "mbpo_halfcheetah_source"
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    algo: SACConfig = field(default_factory=SACConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mbpo: MBPOConfig = field(default_factory=MBPOConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def __post_init__(self) -> None:
        RunConfig(name=self.name, seed=self.seed, env=self.env, dynamics=self.dynamics,
                  algo=self.algo, training=self.training)
        if self.seed != 0:
            raise ValueError("This development milestone uses seed 0 only")
        if self.training.learning_starts < 3 or self.training.batch_size < 2:
            raise ValueError("MBPO requires learning_starts >= 3 and batch_size >= 2")

    def to_dict(self) -> dict:
        return yaml.safe_load(yaml.safe_dump(asdict(self)))

    @classmethod
    def from_dict(cls, raw: dict) -> "MBPORunConfig":
        values = _mapping(raw, set(cls.__dataclass_fields__))
        for key, kind in (("env", EnvConfig), ("dynamics", DynamicsConfig), ("algo", SACConfig),
                          ("training", TrainingConfig), ("mbpo", MBPOConfig)):
            values[key] = kind(**_mapping(values.get(key, {}), set(kind.__dataclass_fields__)))
        return cls(**values)


def load_mbpo_config(path: str | Path) -> MBPORunConfig:
    path = Path(path)
    with path.open() as stream:
        values = yaml.safe_load(stream)
    for key in ("algo", "mbpo"):
        if isinstance(values.get(key), str):
            with (path.parent / values[key]).open() as stream:
                values[key] = yaml.safe_load(stream)
    return MBPORunConfig.from_dict(values)
