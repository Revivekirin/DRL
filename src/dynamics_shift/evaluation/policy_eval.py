"""Paired deterministic evaluation without learner updates."""
from dataclasses import asdict, dataclass
import gymnasium as gym
import numpy as np
from dynamics_shift.algorithms.sac.learner import SACLearner
from dynamics_shift.config import DynamicsConfig, EnvConfig, ExperimentConfig
from dynamics_shift.envs import make_env


@dataclass(frozen=True)
class EpisodeResult:
    condition: str
    actuator_scale: float
    seed: int
    per_episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool


def assert_matching_contract(source: gym.Env, target: gym.Env) -> None:
    """The stock factory fixes all settings except the explicit actuator gear."""
    if source.observation_space != target.observation_space or source.action_space != target.action_space:
        raise ValueError("Source/target spaces differ")
    if source.spec.id != target.spec.id or source.spec.max_episode_steps != target.spec.max_episode_steps:
        raise ValueError("Source/target environment or horizon differs")
    for name in ("_forward_reward_weight", "_ctrl_cost_weight", "_exclude_current_positions_from_observation", "frame_skip", "_reset_noise_scale"):
        if getattr(source.unwrapped, name) != getattr(target.unwrapped, name):
            raise ValueError(f"Source/target setting differs: {name}")
    if source.unwrapped.dt != target.unwrapped.dt:
        raise ValueError("Source/target timestep differs")


def evaluate_policy(learner: SACLearner, env: gym.Env, seeds: tuple[int, ...],
                    condition: str, actuator_scale: float) -> list[EpisodeResult]:
    results = []
    was_training = learner.actor.training
    learner.actor.eval()
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            total_return = 0.0
            for length in range(1, env.spec.max_episode_steps + 1):
                obs, reward, terminated, truncated, _ = env.step(learner.act(obs, deterministic=True))
                total_return += float(reward)
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("Environment did not finish at its configured horizon")
            results.append(EpisodeResult(condition, actuator_scale, seed, total_return,
                                         length, bool(terminated), bool(truncated)))
    finally:
        learner.actor.train(was_training)
    return results


def evaluate_shift(learner: SACLearner, env_config: EnvConfig, seeds: tuple[int, ...],
                   target_scale: float) -> tuple[list[dict], dict]:
    if not seeds:
        raise ValueError("Evaluation needs at least one explicit seed")
    source = make_env(ExperimentConfig(env_config, DynamicsConfig(1.0), seeds[0]))
    try:
        target = make_env(ExperimentConfig(env_config, DynamicsConfig(target_scale), seeds[0]))
        try:
            assert_matching_contract(source, target)
            if learner.obs_dim != source.observation_space.shape[0] or not (
                np.array_equal(learner.action_low, source.action_space.low)
                and np.array_equal(learner.action_high, source.action_space.high)
            ):
                raise ValueError("Checkpoint and evaluation spaces differ")
            results = evaluate_policy(learner, source, seeds, "source", 1.0)
            results += evaluate_policy(learner, target, seeds, "target", target_scale)
        finally:
            target.close()
    finally:
        source.close()
    summary = {}
    for condition in ("source", "target"):
        returns = np.asarray([r.per_episode_return for r in results if r.condition == condition])
        summary[condition] = {"mean_return": float(returns.mean()), "std_return": float(returns.std(ddof=0)),
                              "median_return": float(np.median(returns)), "episodes": len(returns)}
    summary["delta_return"] = summary["target"]["mean_return"] - summary["source"]["mean_return"]
    return [asdict(result) for result in results], summary
