"""Nominal PushCube adapter; no policies, shifts, rendering or automatic reset."""
from dynamics_shift.config import ExperimentConfig


def make_maniskill_env(config: ExperimentConfig, *, render_mode=None):
    if config.env.backend != "maniskill" or config.dynamics is not None:
        raise ValueError("Expected nominal ManiSkill config without actuator dynamics")
    if render_mode is not None:
        raise ValueError("Stage-2 ManiSkill adapter supports headless state observations only")
    try:
        import gymnasium as gym
        import mani_skill.envs  # Registers PushCube-v1 with Gymnasium.
        from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
    except ImportError as error:
        raise ImportError(
            "ManiSkill optional dependencies are missing or could not be imported. "
            "On the server, install the appropriate PyTorch build, then run "
            "python -m pip install -e '.[maniskill]' from the repository root. "
            "See docs/maniskill.md for compatibility checks. "
            f"Original import error: {error}"
        ) from error
    settings = config.env
    env = gym.make(settings.id, obs_mode=settings.obs_mode,
                   robot_uids=settings.robot_uids, control_mode=settings.control_mode,
                   reward_mode=settings.reward_mode, sim_backend=settings.sim_backend,
                   num_envs=settings.num_envs, render_mode=None, render_backend="none")
    try:
        # Official wrapper removes the batch dimension and converts tensors to NumPy.
        # It neither resets on step nor suppresses task terminations.
        env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)
        env.action_space.seed(config.seed)
        env.observation_space.seed(config.seed)
        return env
    except Exception:
        env.close()
        raise
