"""Resolved source-training configuration, separate from environment configs."""

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path

import yaml

from dynamics_shift.algorithms.sac.config import SACConfig
from dynamics_shift.config import (
    DynamicsConfig,
    EnvConfig,
    ExperimentConfig,
    _mapping,
)


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cuda:0"

    # Count of real environment transitions, not vector-env calls.
    real_env_steps: int = 1_000_000

    batch_size: int = 256
    replay_capacity: int = 1_000_000
    learning_starts: int = 10_000

    # Update-to-data ratio:
    #
    #   UTD = gradient updates / newly collected real transitions
    #
    # Examples:
    #   num_envs=1,  utd=1.0 -> 1 update per transition
    #   num_envs=32, utd=0.5 -> 16 updates per vector env step
    #
    # The trainer must implement this using an update-budget accumulator.
    utd: float = 1.0

    log_every: int = 1000
    torch_threads: int = 1
    checkpoint_every: int = 10000

    def __post_init__(self) -> None:
        import re

        if (
            not isinstance(self.device, str)
            or not re.fullmatch(
                r"cpu|cuda(?::[0-9]+)?",
                self.device,
            )
        ):
            raise ValueError(
                "device must be cpu, cuda, or cuda:N"
            )

        integer_settings = {
            "real_env_steps": self.real_env_steps,
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
            "learning_starts": self.learning_starts,
            "log_every": self.log_every,
            "torch_threads": self.torch_threads,
            "checkpoint_every": self.checkpoint_every,
        }

        for name, value in integer_settings.items():
            minimum = 0 if name == "learning_starts" else 1

            if type(value) is not int or value < minimum:
                raise ValueError(
                    f"Invalid integer setting: {name}"
                )

        if (
            isinstance(self.utd, bool)
            or not isinstance(self.utd, (int, float))
            or not isfinite(self.utd)
            or self.utd <= 0
        ):
            raise ValueError(
                "training.utd must be a finite positive number"
            )

        object.__setattr__(
            self,
            "utd",
            float(self.utd),
        )

        if self.batch_size > self.replay_capacity:
            raise ValueError(
                "batch_size exceeds replay_capacity"
            )

        if self.learning_starts > self.real_env_steps:
            raise ValueError(
                "learning_starts exceeds real_env_steps"
            )


@dataclass(frozen=True)
class EvaluationConfig:
    seeds: tuple[int, ...] = (
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
    )

    target_actuator_scale: float = 0.7

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seeds",
            tuple(self.seeds),
        )

        if (
            not self.seeds
            or any(
                type(seed) is not int or seed < 0
                for seed in self.seeds
            )
        ):
            raise ValueError(
                "Evaluation needs explicit nonnegative seeds"
            )

        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(
                "Evaluation seeds must be distinct"
            )

        DynamicsConfig(
            self.target_actuator_scale
        )


@dataclass(frozen=True)
class PushCubeEvaluationConfig:
    episodes: int = 3
    interval: int = 0
    seed: int = 0
    termination_policy: str = "terminate_on_success"

    def __post_init__(self) -> None:
        if (
            type(self.interval) is not int
            or self.interval < 0
        ):
            raise ValueError(
                "evaluation.interval must be a "
                "nonnegative real-transition count"
            )

        if (
            type(self.episodes) is not int
            or self.episodes < 1
        ):
            raise ValueError(
                "evaluation.episodes must be positive"
            )

        if (
            type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError(
                "evaluation.seed must be a "
                "nonnegative integer"
            )

        if (
            self.termination_policy
            != "terminate_on_success"
        ):
            raise ValueError(
                "PushCube training and evaluation "
                "preserve success termination"
            )


@dataclass(frozen=True)
class TrackingConfig:
    mode: str = "disabled"
    project: str = "dynamics-shift"
    entity: str | None = None
    video_every: int = 50000
    video_steps: int = 1000

    def __post_init__(self) -> None:
        if self.mode not in (
            "disabled",
            "online",
            "offline",
        ):
            raise ValueError(
                "tracking.mode must be "
                "disabled, online or offline"
            )

        if not self.project:
            raise ValueError(
                "tracking.project must be nonempty"
            )

        if (
            type(self.video_every) is not int
            or self.video_every < 0
        ):
            raise ValueError(
                "video_every must be nonnegative; "
                "0 disables video"
            )

        if (
            type(self.video_steps) is not int
            or not 1 <= self.video_steps <= 1000
        ):
            raise ValueError(
                "video_steps must be in [1, 1000]"
            )


@dataclass(frozen=True)
class RunConfig:
    name: str = "sac_halfcheetah_source"
    seed: int = 0

    env: EnvConfig = field(
        default_factory=EnvConfig
    )

    dynamics: DynamicsConfig | None = None

    algo: SACConfig = field(
        default_factory=SACConfig
    )

    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )

    evaluation: (
        EvaluationConfig
        | PushCubeEvaluationConfig
    ) = field(
        default_factory=EvaluationConfig
    )

    tracking: TrackingConfig = field(
        default_factory=TrackingConfig
    )

    def __post_init__(self) -> None:
        if (
            not self.name
            or any(
                c not in (
                    "abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789_-"
                )
                for c in self.name
            )
        ):
            raise ValueError(
                "name must contain only letters, "
                "digits, underscores or hyphens"
            )

        resolved = ExperimentConfig(
            self.env,
            self.dynamics,
            self.seed,
        )

        object.__setattr__(
            self,
            "dynamics",
            resolved.dynamics,
        )

        if self.env.backend == "maniskill":
            if not isinstance(
                self.evaluation,
                PushCubeEvaluationConfig,
            ):
                raise ValueError(
                    "PushCube requires "
                    "PushCubeEvaluationConfig"
                )

            # Training with a vectorized GPU environment is
            # allowed. Video support still depends on the
            # downstream ManiSkill adapter/recorder.
            if (
                self.tracking.mode == "disabled"
                and self.tracking.video_every != 0
            ):
                raise ValueError(
                    "tracking.video_every must be 0 "
                    "when tracking.mode='disabled'"
                )

        else:
            if (
                self.dynamics is None
                or self.dynamics.actuator_scale
                != 1.0
            ):
                raise ValueError(
                    "Source SAC training requires "
                    "nominal actuator_scale=1.0"
                )

    def to_dict(self) -> dict:
        # YAML roundtrip normalizes tuples to
        # plain serializable lists.
        return yaml.safe_load(
            yaml.safe_dump(
                asdict(self)
            )
        )

    @classmethod
    def from_dict(
        cls,
        raw: dict,
    ) -> "RunConfig":
        raw = _mapping(
            raw,
            set(cls.__dataclass_fields__),
        )

        values = dict(raw)

        env_raw = _mapping(
            raw.get("env", {}),
            set(EnvConfig.__dataclass_fields__),
        )

        is_maniskill = (
            env_raw.get("backend")
            == "maniskill"
        )

        for name, kind in (
            ("env", EnvConfig),
            ("dynamics", DynamicsConfig),
            ("algo", SACConfig),
            ("training", TrainingConfig),
            ("evaluation", EvaluationConfig),
            ("tracking", TrackingConfig),
        ):
            if (
                is_maniskill
                and name == "dynamics"
            ):
                if raw.get(name) is not None:
                    raise ValueError(
                        "ManiSkill does not support "
                        "actuator dynamics settings; "
                        "omit dynamics"
                    )

                values[name] = None
                continue

            if (
                is_maniskill
                and name == "evaluation"
            ):
                kind = PushCubeEvaluationConfig

            values[name] = kind(
                **_mapping(
                    raw.get(name, {}),
                    set(
                        kind.__dataclass_fields__
                    ),
                )
            )

        return cls(**values)


def load_run_config(
    path: str | Path,
) -> RunConfig:
    path = Path(path)

    with path.open() as stream:
        raw = _mapping(
            yaml.safe_load(stream),
            set(
                RunConfig.__dataclass_fields__
            ),
        )

    if isinstance(
        raw.get("algo"),
        str,
    ):
        with (
            path.parent / raw["algo"]
        ).open() as stream:
            raw["algo"] = yaml.safe_load(
                stream
            )

    return RunConfig.from_dict(raw)