"""Single construction path for the validated benchmark."""
import gymnasium as gym
from dynamics_shift.config import ExperimentConfig
from .dynamics import DynamicsController


def make_env(config: ExperimentConfig) -> DynamicsController:
    """Construct an unreset environment; caller uses reset(seed=config.seed)."""
    env = gym.make(config.env.id)
    try:
        controller = DynamicsController(env)
        controller.set_actuator_scale(config.dynamics.actuator_scale)
        controller.action_space.seed(config.seed)
        return controller
    except Exception:
        env.close()
        raise
