"""Small, strict YAML configuration schema."""
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from collections.abc import Mapping
import yaml


def validate_scale(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("actuator_scale must be a finite positive number")
    if not isfinite(value) or value <= 0:
        raise ValueError("actuator_scale must be a finite positive number")
    return float(value)


@dataclass(frozen=True)
class EnvConfig:
    id: str = "HalfCheetah-v5"

    def __post_init__(self) -> None:
        if self.id != "HalfCheetah-v5":
            raise ValueError("Only HalfCheetah-v5 is supported")


@dataclass(frozen=True)
class DynamicsConfig:
    actuator_scale: float = 1.0

    def __post_init__(self) -> None:
        validate_scale(self.actuator_scale)


@dataclass(frozen=True)
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _mapping(value: object, allowed: set[str]) -> dict:
    if not isinstance(value, Mapping) or set(value) - allowed:
        raise ValueError(f"Expected a mapping with only these keys: {sorted(allowed)}")
    return dict(value)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open() as stream:
        raw = _mapping(yaml.safe_load(stream), {"env", "dynamics", "seed"})
    return ExperimentConfig(
        env=EnvConfig(**_mapping(raw.get("env", {}), {"id"})),
        dynamics=DynamicsConfig(**_mapping(raw.get("dynamics", {}), {"actuator_scale"})),
        seed=raw.get("seed", 0),
    )
