"""Constant-condition frozen evaluation and paired descriptive statistics."""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import torch
import yaml
from dynamics_shift.config import ExperimentConfig, _mapping, validate_scale
from dynamics_shift.envs import make_env
from .frozen_actor import FrozenActor
from .policy_eval import evaluate_policy


@dataclass(frozen=True)
class SweepConfig:
    checkpoint: str
    actuator_strengths: tuple[float, ...] = (1., .95, .9, .85, .8, .75, .7, .6)
    evaluation_seeds: tuple[int, ...] = tuple(range(100, 110))
    device: str = "cuda:0"
    torch_threads: int = 1
    output_root: str = "outputs/shift_sweeps/sac_halfcheetah_actuator/seed_0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "actuator_strengths", tuple(validate_scale(v) for v in self.actuator_strengths))
        object.__setattr__(self, "evaluation_seeds", tuple(self.evaluation_seeds))
        if not self.checkpoint or 1.0 not in self.actuator_strengths:
            raise ValueError("Require a checkpoint and nominal strength 1.0")
        if len(set(self.actuator_strengths)) != len(self.actuator_strengths):
            raise ValueError("Duplicate actuator conditions")
        if (not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(self.evaluation_seeds)
                or any(type(s) is not int or s < 0 for s in self.evaluation_seeds)):
            raise ValueError("Require unique nonnegative explicit evaluation seeds")
        if type(self.torch_threads) is not int or self.torch_threads <= 0:
            raise ValueError("torch_threads must be positive")


def load_sweep_config(path: str | Path) -> SweepConfig:
    with Path(path).open() as stream:
        return SweepConfig(**_mapping(yaml.safe_load(stream), set(SweepConfig.__dataclass_fields__)))


def collect_sweep(policy: FrozenActor, config: SweepConfig) -> list[dict]:
    """Each environment retains one scale; no temporal event controller is used."""
    before = {key: tensor.detach().clone() for key, tensor in policy.actor.state_dict().items()}
    rows = []
    try:
        for scale in config.actuator_strengths:
            env = make_env(ExperimentConfig(seed=config.evaluation_seeds[0]))
            try:
                env.apply_dynamics_shift(parameter="actuator_strength", value=scale)
                if env.get_shift_parameter("actuator_strength") != scale:
                    raise RuntimeError("Environment did not resolve the requested actuator strength")
                if (env.observation_space.shape != (policy.obs_dim,)
                        or not np.array_equal(env.action_space.low, policy.action_low)
                        or not np.array_equal(env.action_space.high, policy.action_high)):
                    raise ValueError("Policy and environment spaces differ")
                # One factory and identical defaults fix all non-dynamics settings.
                episodes = evaluate_policy(policy, env, config.evaluation_seeds, "constant", scale)
                for episode_id, episode in enumerate(episodes):
                    rows.append({"actuator_strength": scale, "episode_id": episode_id,
                                 "evaluation_seed": episode.seed, "episode_return": episode.per_episode_return,
                                 "episode_length": episode.episode_length, "terminated": episode.terminated,
                                 "truncated": episode.truncated})
                if env.get_shift_parameter("actuator_strength") != scale:
                    raise RuntimeError("Dynamics changed during constant-condition evaluation")
            finally:
                env.close()
    finally:
        after = policy.actor.state_dict()
        if before.keys() != after.keys() or any(not torch.equal(before[k], after[k]) for k in before):
            raise RuntimeError("Frozen actor changed during severity sweep")
    return rows


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Population SD across episodes; ratios undefined for near-zero nominal mean."""
    groups = defaultdict(dict)
    for row in rows:
        scale, seed, value = float(row["actuator_strength"]), int(row["evaluation_seed"]), float(row["episode_return"])
        if seed in groups[scale] or not math.isfinite(value):
            raise ValueError("Duplicate condition/seed or nonfinite return")
        groups[scale][seed] = value
    if 1.0 not in groups:
        raise ValueError("Missing nominal condition")
    nominal = groups[1.0]
    nominal_mean = float(np.mean(list(nominal.values())))
    summary, paired = [], []
    for scale in sorted(groups, reverse=True):
        values = groups[scale]
        if values.keys() != nominal.keys():
            raise ValueError("Every condition must contain the same evaluation seeds")
        returns = np.array(list(values.values()))
        mean = float(returns.mean())
        relative = mean / nominal_mean if abs(nominal_mean) > 1e-12 else None
        deltas = []
        for seed in sorted(nominal):
            delta = values[seed] - nominal[seed]
            deltas.append(delta)
            paired.append(dict(actuator_strength=scale, evaluation_seed=seed, nominal_return=nominal[seed],
                               shifted_return=values[seed], paired_delta_return=delta))
        summary.append(dict(actuator_strength=scale, n_episodes=len(returns), mean_return=mean,
                            std_return=float(returns.std(ddof=0)), median_return=float(np.median(returns)),
                            min_return=float(returns.min()), max_return=float(returns.max()),
                            absolute_drop=mean - nominal_mean, relative_return=relative,
                            relative_drop=1 - relative if relative is not None else None,
                            paired_mean_delta=float(np.mean(deltas)), paired_std_delta=float(np.std(deltas, ddof=0))))
    adjacent = [{"from_strength": a["actuator_strength"], "to_strength": b["actuator_strength"],
                 "mean_return_change": b["mean_return"] - a["mean_return"],
                 "return_change_per_unit_strength_decrease": (b["mean_return"] - a["mean_return"]) /
                    (a["actuator_strength"] - b["actuator_strength"])} for a, b in zip(summary, summary[1:])]
    analysis = {"monotonic_nondecreasing_with_strength": all(a["mean_return_change"] <= 0 for a in adjacent),
                "adjacent_changes": adjacent,
                "steepest_decrease_region": min((a for a in adjacent if a["mean_return_change"] < 0),
                    key=lambda a: a["return_change_per_unit_strength_decrease"], default=None),
                "interpretation": "Descriptive seed-0 mean curve; no severity labels or threshold significance inferred."}
    return summary, paired, analysis


def plot_curve(summary: list[dict], path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = sorted(summary, key=lambda r: r["actuator_strength"])
    x = [r["actuator_strength"] for r in ordered]
    nominal = next(r["mean_return"] for r in ordered if r["actuator_strength"] == 1.)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    try:
        axes[0].errorbar(x, [r["mean_return"] for r in ordered],
                         yerr=[r["std_return"] for r in ordered], marker="o", capsize=3)
        axes[0].set_ylabel("Frozen-policy mean return")
        if abs(nominal) > 1e-12:
            axes[1].errorbar(x, [r["relative_return"] for r in ordered],
                             yerr=[r["std_return"] / abs(nominal) for r in ordered], marker="o", capsize=3)
        else:
            axes[1].text(.5, .5, "Undefined: nominal mean near zero", ha="center", transform=axes[1].transAxes)
        axes[1].axhline(1., color="gray", linestyle="--")
        axes[1].set_ylabel("J(strength) / J(1.0)")
        for ax in axes:
            ax.set_xlabel("Actuator strength")
            ax.grid(alpha=.2)
        fig.suptitle("Frozen SAC, training seed 0 | bars: episode population SD")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
