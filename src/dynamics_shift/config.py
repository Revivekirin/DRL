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
    backend: str = "mujoco"
    obs_mode: str | None = None
    robot_uids: str | None = None
    control_mode: str | None = None
    reward_mode: str | None = None
    sim_backend: str | None = None
    num_envs: int | None = None

    def __post_init__(self) -> None:
        options = ("obs_mode", "robot_uids", "control_mode", "reward_mode", "sim_backend", "num_envs")
        if self.backend == "mujoco":
            if self.id != "HalfCheetah-v5":
                raise ValueError("MuJoCo backend only supports HalfCheetah-v5")
            if any(getattr(self, key) is not None for key in options):
                raise ValueError("ManiSkill options are not supported by the MuJoCo backend")
        elif self.backend == "maniskill":
            expected = dict(id="PushCube-v1", obs_mode="state", robot_uids="panda",
                            control_mode="pd_joint_delta_pos", reward_mode="normalized_dense",
                            sim_backend="cpu", num_envs=1)
            for key, value in expected.items():
                if getattr(self, key) != value or (key == "num_envs" and type(self.num_envs) is not int):
                    raise ValueError(f"Stage-2 ManiSkill requires env.{key}={value!r}")
        else:
            raise ValueError(f"Unsupported environment backend: {self.backend}")


@dataclass(frozen=True)
class DynamicsConfig:
    actuator_scale: float = 1.0

    def __post_init__(self) -> None:
        validate_scale(self.actuator_scale)


@dataclass(frozen=True)
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    dynamics: DynamicsConfig | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.env.backend == "mujoco":
            if self.dynamics is None:
                object.__setattr__(self, "dynamics", DynamicsConfig())
            elif not isinstance(self.dynamics, DynamicsConfig):
                raise ValueError("MuJoCo dynamics must be DynamicsConfig")
        else:
            if self.dynamics is not None:
                raise ValueError("ManiSkill does not support actuator dynamics settings; omit dynamics")
            if self.seed != 0:
                raise ValueError("Stage-2 ManiSkill smoke uses seed 0 only")


def _mapping(value: object, allowed: set[str]) -> dict:
    if not isinstance(value, Mapping) or set(value) - allowed:
        raise ValueError(f"Expected a mapping with only these keys: {sorted(allowed)}")
    return dict(value)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open() as stream:
        raw = _mapping(yaml.safe_load(stream), {"env", "dynamics", "seed"})
    env = EnvConfig(**_mapping(raw.get("env", {}), set(EnvConfig.__dataclass_fields__)))
    if env.backend == "maniskill" and raw.get("dynamics") is not None:
        raise ValueError("ManiSkill does not support actuator dynamics settings; omit dynamics")
    return ExperimentConfig(
        env=env,
        dynamics=(None if env.backend == "maniskill" else
                  DynamicsConfig(**_mapping(raw.get("dynamics", {}), {"actuator_scale"}))),
        seed=raw.get("seed", 0),
    )
