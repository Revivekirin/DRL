"""Nominal PushCube ManiSkill adapter.

Supports:
- CPU single-environment smoke/evaluation path via CPUGymWrapper.
- GPU vectorized training path via ManiSkillVectorEnv.

No dynamics shifts are applied here.
"""

from __future__ import annotations

from dynamics_shift.config import ExperimentConfig


def make_maniskill_env(
    config: ExperimentConfig,
    *,
    render_mode=None,
):
    if (
        config.env.backend != "maniskill"
        or config.dynamics is not None
    ):
        raise ValueError(
            "Expected nominal ManiSkill config without actuator dynamics"
        )

    if render_mode is not None:
        raise ValueError(
            "Current ManiSkill adapter supports headless state observations only"
        )

    try:
        import gymnasium as gym
        import mani_skill.envs  # Registers PushCube-v1 with Gymnasium.

        from mani_skill.utils.wrappers.gymnasium import (
            CPUGymWrapper,
            ManiSkillVectorEnv,
        )

    except ImportError as error:
        raise ImportError(
            "ManiSkill optional dependencies are missing or could not be imported. "
            "On the server, install the appropriate PyTorch build, then run "
            "python -m pip install -e '.[maniskill]' from the repository root. "
            "See docs/maniskill.md for compatibility checks. "
            f"Original import error: {error}"
        ) from error

    settings = config.env

    env = gym.make(
        settings.id,
        obs_mode=settings.obs_mode,
        robot_uids=settings.robot_uids,
        control_mode=settings.control_mode,
        reward_mode=settings.reward_mode,
        sim_backend=settings.sim_backend,
        num_envs=settings.num_envs,
        render_mode=None,
        render_backend="none",
    )

    try:
        # ---------------------------------------------------------------
        # CPU single-env path
        # ---------------------------------------------------------------
        #
        # Keep the already validated smoke/evaluation behavior:
        # - removes ManiSkill's leading batch dimension,
        # - converts tensors to NumPy,
        # - preserves success termination,
        # - does not automatically reset through this wrapper.
        #
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
        #
        # Keep tensors on device and preserve the vector dimension.
        # train_pushcube_sac.py handles:
        # - batched observations/actions,
        # - replay insertion,
        # - per-env episode statistics,
        # - final observations after automatic reset,
        # - UTD scheduling.
        #
        if settings.sim_backend == "gpu":
            if settings.num_envs is None or settings.num_envs < 1:
                raise ValueError(
                    "GPU ManiSkill adapter requires num_envs >= 1"
                )

            env = ManiSkillVectorEnv(
                env,
                num_envs=settings.num_envs,

                # Keep the same project semantics as the existing
                # PushCube smoke path: successful task termination is real.
                #
                # This is intentionally False rather than matching some
                # ManiSkill SAC examples that ignore task termination.
                ignore_terminations=False,

                # Let the vector wrapper automatically reset finished envs.
                # The trainer must use final_observation for replay targets.
                auto_reset=True,

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