"""Paired deterministic evaluation without learner updates."""
from dataclasses import asdict, dataclass
import gymnasium as gym
import numpy as np
from dynamics_shift.algorithms.sac.learner import SACLearner
from dynamics_shift.config import DynamicsConfig, EnvConfig, ExperimentConfig
from dynamics_shift.envs import make_env
from .contracts import assert_common_contract, assert_learner_contract


@dataclass(frozen=True)
class EpisodeResult:
    condition: str
    actuator_scale: float
    seed: int
    per_episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    success_once: bool | None = None
    success_at_end: bool | None = None


def assert_matching_contract(source: gym.Env, target: gym.Env) -> None:
    """The stock factory fixes all settings except the explicit actuator gear."""
    assert_common_contract(source, target)
    assert_halfcheetah_contract(source, target)


def assert_halfcheetah_contract(source, target):
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
            assert_learner_contract(learner, source)
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
