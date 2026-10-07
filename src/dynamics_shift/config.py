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

    # ManiSkill-only options
    obs_mode: str | None = None
    robot_uids: str | None = None
    control_mode: str | None = None
    reward_mode: str | None = None
    sim_backend: str | None = None
    num_envs: int | None = None

    def __post_init__(self) -> None:
        maniskill_options = (
            "obs_mode",
            "robot_uids",
            "control_mode",
            "reward_mode",
            "sim_backend",
            "num_envs",
        )

        if self.backend == "mujoco":
            if self.id != "HalfCheetah-v5":
                raise ValueError(
                    "MuJoCo backend only supports HalfCheetah-v5"
                )

            if any(
                getattr(self, key) is not None
                for key in maniskill_options
            ):
                raise ValueError(
                    "ManiSkill options are not supported "
                    "by the MuJoCo backend"
                )

            return

        if self.backend == "maniskill":
            from dynamics_shift.envs.maniskill_tasks import state_task
            state_task(self.id)

            if self.obs_mode != "state":
                raise ValueError(
                    "ManiSkill state runner currently requires env.obs_mode='state'"
                )

            if self.robot_uids != "panda":
                raise ValueError(
                    "ManiSkill state runner currently requires env.robot_uids='panda'"
                )

            allowed_control_modes = {
                "pd_joint_delta_pos",
                "pd_ee_delta_pos",
            }

            if self.control_mode not in allowed_control_modes:
                raise ValueError(
                    "Unsupported ManiSkill control mode. "
                    f"Expected one of {sorted(allowed_control_modes)}, "
                    f"got {self.control_mode!r}"
                )

            if self.reward_mode != "normalized_dense":
                raise ValueError(
                    "ManiSkill state runner currently requires "
                    "env.reward_mode='normalized_dense'"
                )

            allowed_sim_backends = {
                "cpu",
                "gpu",
            }

            if self.sim_backend not in allowed_sim_backends:
                raise ValueError(
                    "Unsupported ManiSkill simulation backend. "
                    f"Expected one of {sorted(allowed_sim_backends)}, "
                    f"got {self.sim_backend!r}"
                )

            if type(self.num_envs) is not int or self.num_envs < 1:
                raise ValueError(
                    "env.num_envs must be a positive integer"
                )

            # Keep the old CPU smoke path deliberately single-env.
            if self.sim_backend == "cpu" and self.num_envs != 1:
                raise ValueError(
                    "The current CPU ManiSkill path supports num_envs=1 only. "
                    "Use sim_backend='gpu' for vectorized environments."
                )

            return

        raise ValueError(
            f"Unsupported environment backend: {self.backend}"
        )


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
            raise ValueError(
                "seed must be a nonnegative integer"
            )

        if self.env.backend == "mujoco":
            if self.dynamics is None:
                object.__setattr__(
                    self,
                    "dynamics",
                    DynamicsConfig(),
                )
            elif not isinstance(
                self.dynamics,
                DynamicsConfig,
            ):
                raise ValueError(
                    "MuJoCo dynamics must be DynamicsConfig"
                )

        else:
            if self.dynamics is not None:
                raise ValueError(
                    "ManiSkill does not support actuator dynamics "
                    "settings; omit dynamics"
                )


def _mapping(
    value: object,
    allowed: set[str],
) -> dict:
    if (
        not isinstance(value, Mapping)
        or set(value) - allowed
    ):
        raise ValueError(
            "Expected a mapping with only these keys: "
            f"{sorted(allowed)}"
        )

    return dict(value)


def load_config(
    path: str | Path,
) -> ExperimentConfig:
    with Path(path).open() as stream:
        raw = _mapping(
            yaml.safe_load(stream),
            {
                "env",
                "dynamics",
                "seed",
            },
        )

    env = EnvConfig(
        **_mapping(
            raw.get("env", {}),
            set(
                EnvConfig.__dataclass_fields__
            ),
        )
    )

    if (
        env.backend == "maniskill"
        and raw.get("dynamics") is not None
    ):
        raise ValueError(
            "ManiSkill does not support actuator dynamics "
            "settings; omit dynamics"
        )

    return ExperimentConfig(
        env=env,
        dynamics=(
            None
            if env.backend == "maniskill"
            else DynamicsConfig(
                **_mapping(
                    raw.get(
                        "dynamics",
                        {},
                    ),
                    {
                        "actuator_scale",
                    },
                )
            )
        ),
        seed=raw.get(
            "seed",
            0,
        ),
    )