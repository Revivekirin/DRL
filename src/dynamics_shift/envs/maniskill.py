"""Nominal PushCube ManiSkill adapter.

Supports:
- CPU single-environment smoke/evaluation path via CPUGymWrapper.
- GPU vectorized training path via ManiSkillVectorEnv.
"""

from __future__ import annotations

from dynamics_shift.config import ExperimentConfig


def make_maniskill_env(
    config: ExperimentConfig,
    *,
    render_mode=None,
    _video_backend=None,
):
    if (
        config.env.backend != "maniskill"
        or config.dynamics is not None
    ):
        raise ValueError(
            "Expected nominal ManiSkill config without actuator dynamics"
        )

    if render_mode is not None and _video_backend is None:
        raise ValueError(
            "Current ManiSkill adapter supports headless state observations only"
        )

    try:
        import gymnasium as gym
        import mani_skill.envs

        from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    except ImportError as error:
        raise ImportError(
            "ManiSkill optional dependencies are missing or incompatible. "
            "Expected both CPUGymWrapper and ManiSkillVectorEnv to be available. "
            "Install the optional extra from the repository root with: "
            "pip install -e '.[maniskill]'. "
            f"Original import error: {error}"
        ) from error

    settings = config.env
    if _video_backend is not None and (settings.sim_backend != "cpu" or settings.num_envs != 1):
        raise ValueError("Video evaluation requires a separate single CPU environment")

    env = gym.make(
        settings.id,
        obs_mode=settings.obs_mode,
        robot_uids=settings.robot_uids,
        control_mode=settings.control_mode,
        reward_mode=settings.reward_mode,
        sim_backend=settings.sim_backend,
        num_envs=settings.num_envs,
        render_mode=render_mode,
        render_backend=_video_backend or "none",
    )

    try:
        # ---------------------------------------------------------------
        # CPU single-env path
        # ---------------------------------------------------------------
        if settings.sim_backend == "cpu":
            if settings.num_envs != 1:
                raise ValueError(
                    "CPU ManiSkill adapter currently requires num_envs=1"
                )

            env = CPUGymWrapper(
                env,
                ignore_terminations=False,
                record_metrics=True,
            )

            env.action_space.seed(config.seed)
            env.observation_space.seed(config.seed)

            return env

        # ---------------------------------------------------------------
        # GPU vectorized path
        # ---------------------------------------------------------------
        if settings.sim_backend == "gpu":
            if settings.num_envs is None or settings.num_envs < 1:
                raise ValueError(
                    "GPU ManiSkill adapter requires num_envs >= 1"
                )

            env = ManiSkillVectorEnv(
                env,
                settings.num_envs,
                ignore_terminations=True,
                record_metrics=True,
            )

            return env

        raise ValueError(
            f"Unsupported ManiSkill sim_backend: "
            f"{settings.sim_backend!r}"
        )

    except Exception:
        env.close()
        raise


def make_maniskill_video_env(config):
    """Explicit offscreen evaluation only; regular training factory stays headless."""
    return make_maniskill_env(config, render_mode="rgb_array", _video_backend="gpu")
