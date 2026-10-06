"""Backend dispatch with optional simulator imports deferred until construction."""
import gymnasium as gym
from dynamics_shift.config import ExperimentConfig


def make_env(config: ExperimentConfig, *, render_mode: str | None = None) -> gym.Env:
    """Construct an environment; caller explicitly resets with config.seed.

    ManiSkill may initialize its scene internally during construction.
    """
    if config.env.backend == "maniskill":
        from .maniskill import make_maniskill_env
        return make_maniskill_env(config, render_mode=render_mode)
    from .dynamics import DynamicsController

    env = gym.make(config.env.id, render_mode=render_mode)
    try:
        controller = DynamicsController(env)
        controller.set_actuator_scale(config.dynamics.actuator_scale)
        controller.action_space.seed(config.seed)
        controller.observation_space.seed(config.seed)
        return controller
    except Exception:
        env.close()
        raise
