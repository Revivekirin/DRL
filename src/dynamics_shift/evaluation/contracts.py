"""Environment-independent space checks and explicit ManiSkill checkpoint contract."""
import numpy as np


def episode_horizon(env):
    if env.spec is not None and env.spec.max_episode_steps is not None:
        return env.spec.max_episode_steps
    from mani_skill.utils.gym_utils import find_max_episode_steps_value
    horizon = find_max_episode_steps_value(env)
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("Environment has no valid episode horizon")
    return horizon


def assert_common_contract(source, target):
    if source.observation_space != target.observation_space or source.action_space != target.action_space:
        raise ValueError("Source/target spaces differ")
    if source.spec.id != target.spec.id or episode_horizon(source) != episode_horizon(target):
        raise ValueError("Source/target environment or horizon differs")


def assert_learner_contract(learner, env):
    if learner.obs_dim != env.observation_space.shape[0] or not (
        np.array_equal(learner.action_low, env.action_space.low)
        and np.array_equal(learner.action_high, env.action_space.high)
    ):
        raise ValueError("Checkpoint and evaluation spaces differ")


def maniskill_contract(config, env):
    if env.ignore_terminations or env.unwrapped.num_envs != 1 or env.unwrapped.gpu_sim_enabled:
        raise ValueError("Expected single CPU ManiSkill preserving success termination")
    from dynamics_shift.envs.maniskill_tasks import task_contract
    return dict(**task_contract(env), env_id=config.env.id, backend=config.env.backend,
                observation_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
                observation_dtype=str(env.observation_space.dtype), action_dtype=str(env.action_space.dtype),
                action_low=env.action_space.low.tolist(), action_high=env.action_space.high.tolist(),
                obs_mode=config.env.obs_mode, robot_uids=config.env.robot_uids,
                control_mode=config.env.control_mode, reward_mode=config.env.reward_mode,
                sim_backend=config.env.sim_backend, num_envs=config.env.num_envs,
                horizon=episode_horizon(env), termination_policy="terminate_on_success",
                truncation_policy="bootstrap_from_final_observation", automatic_reset=False)


def halfcheetah_contract(env):
    return dict(env_id=env.spec.id, backend="mujoco",
                observation_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
                action_low=env.action_space.low.tolist(), action_high=env.action_space.high.tolist(),
                control_mode="stock_joint_torque", reward_mode="stock_halfcheetah",
                horizon=episode_horizon(env), termination_policy="no_physical_termination",
                truncation_policy="bootstrap_from_final_observation", automatic_reset=False)


def success_flag(info, terminated):
    success = info.get("success")
    if not isinstance(success, (bool, np.bool_)):
        raise ValueError("ManiSkill must supply a scalar boolean success")
    if bool(terminated) != bool(success):
        raise ValueError("ManiSkill success termination contract violated")
    return bool(success)

# Backwards-compatible imports for existing integrations.
pushcube_contract = maniskill_contract
