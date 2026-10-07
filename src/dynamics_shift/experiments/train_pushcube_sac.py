"""ManiSkill PushCube SAC training.

Supports:
- legacy single-CPU ManiSkill collection,
- GPU-vectorized ManiSkill collection,
- transition-count based training budgets,
- fractional update-to-data ratio (UTD),
- correct final-observation storage for auto-reset vector environments.

Exact simulator/replay resume remains unsupported.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dynamics_shift.algorithms.sac.learner import SACLearner
from dynamics_shift.config import ExperimentConfig
from dynamics_shift.data.replay_buffer import ReplayBuffer
from dynamics_shift.envs import make_env
from dynamics_shift.evaluation.contracts import episode_horizon
from dynamics_shift.utils.tracking import Tracker
from dynamics_shift.utils.checkpoint import isolated_rng
from dynamics_shift.utils.checkpoint import (
    save_checkpoint,
)
from dynamics_shift.utils.device import check_device


# ---------------------------------------------------------------------------
# Generic scalar / vector helpers
# ---------------------------------------------------------------------------


def _to_numpy(value) -> np.ndarray:
    """Detach tensors and return CPU NumPy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _single_observation_space(env):
    if hasattr(env, "single_observation_space"):
        return env.single_observation_space
    return env.observation_space


def _single_action_space(env):
    if hasattr(env, "single_action_space"):
        return env.single_action_space
    return env.action_space


def _num_envs(env, config) -> int:
    if hasattr(env, "num_envs"):
        return int(env.num_envs)

    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "num_envs"):
        return int(unwrapped.num_envs)

    return int(config.env.num_envs or 1)


def _horizon(env) -> int:
    """Find horizon through either scalar or ManiSkill vector wrappers."""
    candidates = [
        env,
        getattr(env, "_env", None),
        getattr(env, "unwrapped", None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        try:
            return episode_horizon(candidate)
        except Exception:
            pass

    raise ValueError("Could not determine ManiSkill episode horizon")


def _batch_size_from_obs(obs, num_envs: int) -> int:
    array = obs if isinstance(obs, torch.Tensor) else np.asarray(obs)

    if num_envs == 1:
        return 1

    if array.ndim != 2 or array.shape[0] != num_envs:
        raise ValueError(
            f"Expected batched observations with shape "
            f"[{num_envs}, obs_dim], got {tuple(array.shape)}"
        )

    return num_envs


def _deterministic_action(
    learner: SACLearner,
    obs,
):
    """Run deterministic actor without forcing GPU observations through NumPy."""
    if isinstance(obs, torch.Tensor):
        tensor = obs.to(
            device=learner.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            return learner.actor.deterministic(tensor)

    return learner.act(
        np.asarray(obs),
        deterministic=True,
    )


def _policy_action(
    learner: SACLearner,
    obs,
):
    """Sample from SAC actor while preserving environment tensor type."""
    if isinstance(obs, torch.Tensor):
        tensor = obs.to(
            device=learner.device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            action, _ = learner.actor.sample(tensor)

        return action

    return learner.act(
        np.asarray(obs),
        deterministic=False,
    )


def _random_action(
    env,
    obs,
    *,
    num_envs: int,
    device: torch.device,
):
    """Generate uniformly distributed actions for warm-up collection."""
    single_space = _single_action_space(env)

    low = np.asarray(
        single_space.low,
        dtype=np.float32,
    )
    high = np.asarray(
        single_space.high,
        dtype=np.float32,
    )

    if num_envs == 1:
        return env.action_space.sample()

    # ManiSkill GPU env expects torch tensors.
    if isinstance(obs, torch.Tensor):
        low_t = torch.as_tensor(
            low,
            dtype=torch.float32,
            device=device,
        )
        high_t = torch.as_tensor(
            high,
            dtype=torch.float32,
            device=device,
        )

        random01 = torch.rand(
            (num_envs, low.shape[0]),
            dtype=torch.float32,
            device=device,
        )

        return low_t + random01 * (high_t - low_t)

    rng = np.random.random(
        (num_envs, low.shape[0])
    ).astype(np.float32)

    return low + rng * (high - low)


def _extract_vector_success(
    info: dict,
    terminated,
    num_envs: int,
) -> np.ndarray:
    """Return success flags for the transition that just occurred.

    ManiSkillVectorEnv may auto-reset completed environments. In that case,
    terminal-step metrics must be recovered from ``final_info`` using the
    ``_final_info`` mask rather than relying only on ``info["success"]``.
    """

    terminated_np = (
        _to_numpy(terminated)
        .astype(np.bool_, copy=False)
        .reshape(-1)
    )

    if terminated_np.shape != (num_envs,):
        raise ValueError(
            "terminated has wrong vector shape: "
            f"{terminated_np.shape}; expected ({num_envs},)"
        )

    # Current observations / non-final transitions.
    current_success = info.get("success")

    if current_success is None:
        success_np = np.zeros(
            num_envs,
            dtype=np.bool_,
        )
    else:
        success_np = (
            _to_numpy(current_success)
            .astype(np.bool_, copy=False)
            .reshape(-1)
            .copy()
        )

        if success_np.shape != (num_envs,):
            raise ValueError(
                "PushCube success has wrong vector shape: "
                f"{success_np.shape}; expected ({num_envs},)"
            )

    # ManiSkillVectorEnv auto-reset path.
    #
    # For environments that terminated/truncated on this step,
    # info["success"] can already correspond to the reset state.
    # The transition that actually ended the episode is recorded
    # in info["final_info"].
    final_info = info.get("final_info")
    final_mask = info.get("_final_info")

    if final_info is not None and final_mask is not None:
        final_mask_np = (
            _to_numpy(final_mask)
            .astype(np.bool_, copy=False)
            .reshape(-1)
        )

        if final_mask_np.shape != (num_envs,):
            raise ValueError(
                "_final_info has wrong vector shape: "
                f"{final_mask_np.shape}; expected ({num_envs},)"
            )

        final_success = final_info.get("success")

        if final_success is not None:
            final_success_np = (
                _to_numpy(final_success)
                .astype(np.bool_, copy=False)
                .reshape(-1)
            )

            if final_success_np.shape != (num_envs,):
                raise ValueError(
                    "final_info['success'] has wrong vector shape: "
                    f"{final_success_np.shape}; "
                    f"expected ({num_envs},)"
                )

            success_np[final_mask_np] = (
                final_success_np[final_mask_np]
            )

    # PushCube has success as its task termination condition.
    #
    # Validate only the implication needed by this project:
    #
    #     terminated -> success
    #
    # Do not require the raw current info["success"] tensor to be
    # elementwise identical to terminated because vector auto-reset
    # can replace current info with reset-state information.
    invalid = terminated_np & (~success_np)

    if invalid.any():
        raise ValueError(
            "PushCube reported terminated=True without terminal "
            "success for "
            f"{int(invalid.sum())} environment(s). "
            "Inspect final_info/_final_info semantics."
        )

    return success_np


def _true_next_observation(
    next_obs,
    terminated,
    truncated,
    info: dict,
):
    """Recover final observations before vector-env automatic resets."""
    if not isinstance(next_obs, torch.Tensor):
        return np.asarray(next_obs)

    real_next_obs = next_obs.clone()

    terminated_t = torch.as_tensor(
        terminated,
        dtype=torch.bool,
        device=next_obs.device,
    )
    truncated_t = torch.as_tensor(
        truncated,
        dtype=torch.bool,
        device=next_obs.device,
    )

    done = terminated_t | truncated_t

    if not bool(done.any()):
        return real_next_obs

    if "final_observation" not in info:
        raise ValueError(
            "Vector ManiSkill environment auto-reset a completed "
            "episode but did not expose info['final_observation']."
        )

    final_obs = info["final_observation"]

    if not isinstance(final_obs, torch.Tensor):
        final_obs = torch.as_tensor(
            final_obs,
            dtype=real_next_obs.dtype,
            device=real_next_obs.device,
        )

    if final_obs.shape != real_next_obs.shape:
        raise ValueError(
            "final_observation shape mismatch: "
            f"{tuple(final_obs.shape)} vs "
            f"{tuple(real_next_obs.shape)}"
        )

    real_next_obs[done] = final_obs[done]

    return real_next_obs


def _add_replay_batch(
    replay: ReplayBuffer,
    obs,
    action,
    reward,
    next_obs,
    terminated,
    truncated,
    *,
    num_envs: int,
) -> None:
    """Insert scalar or vector transitions into existing NumPy replay."""
    obs_np = _to_numpy(obs).astype(
        np.float32,
        copy=False,
    )
    action_np = _to_numpy(action).astype(
        np.float32,
        copy=False,
    )
    reward_np = _to_numpy(reward).astype(
        np.float32,
        copy=False,
    )
    next_obs_np = _to_numpy(next_obs).astype(
        np.float32,
        copy=False,
    )
    terminated_np = _to_numpy(terminated).astype(
        np.bool_,
        copy=False,
    )
    truncated_np = _to_numpy(truncated).astype(
        np.bool_,
        copy=False,
    )

    if num_envs == 1:
        replay.add(
            np.asarray(obs_np).reshape(-1),
            np.asarray(action_np).reshape(-1),
            float(np.asarray(reward_np).reshape(-1)[0]),
            np.asarray(next_obs_np).reshape(-1),
            bool(np.asarray(terminated_np).reshape(-1)[0]),
            bool(np.asarray(truncated_np).reshape(-1)[0]),
        )
        return

    expected_n = num_envs

    obs_np = obs_np.reshape(expected_n, -1)
    action_np = action_np.reshape(expected_n, -1)
    reward_np = reward_np.reshape(expected_n)
    next_obs_np = next_obs_np.reshape(expected_n, -1)
    terminated_np = terminated_np.reshape(expected_n)
    truncated_np = truncated_np.reshape(expected_n)

    # Keep ReplayBuffer unchanged for now.
    # This can later be replaced by ReplayBuffer.add_batch().
    for env_index in range(expected_n):
        replay.add(
            obs_np[env_index],
            action_np[env_index],
            float(reward_np[env_index]),
            next_obs_np[env_index],
            bool(terminated_np[env_index]),
            bool(truncated_np[env_index]),
        )


def _training_contract(
    config,
    env,
    *,
    num_envs: int,
) -> dict:
    obs_space = _single_observation_space(env)
    action_space = _single_action_space(env)

    return {
        "env_id": config.env.id,
        "backend": config.env.backend,
        "observation_dim": int(obs_space.shape[0]),
        "action_dim": int(action_space.shape[0]),
        "observation_dtype": str(obs_space.dtype),
        "action_dtype": str(action_space.dtype),
        "action_low": np.asarray(
            action_space.low
        ).tolist(),
        "action_high": np.asarray(
            action_space.high
        ).tolist(),
        "obs_mode": config.env.obs_mode,
        "robot_uids": config.env.robot_uids,
        "control_mode": config.env.control_mode,
        "reward_mode": config.env.reward_mode,
        "sim_backend": config.env.sim_backend,
        "num_envs": num_envs,
        "horizon": _horizon(env),
        "termination_policy":
            "ignore_terminations" if config.env.sim_backend == "gpu" else "terminate_on_success",
        "truncation_policy":
            "bootstrap_from_final_observation",
        "automatic_reset": config.env.sim_backend == "gpu",
    }


# ---------------------------------------------------------------------------
# Dedicated CPU evaluation for vector-trained checkpoints
# ---------------------------------------------------------------------------


def _evaluate_saved_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    config,
    *,
    device: torch.device,
) -> dict:
    """Use the same verified evaluator as the standalone CLI, isolating all RNG."""
    from dynamics_shift.experiments.evaluate_sac_shift import evaluate_checkpoint
    from dynamics_shift.utils.checkpoint import isolated_rng
    with isolated_rng(device):
        return evaluate_checkpoint(checkpoint, output_dir, device=device,
                                   evaluation_overrides={"sim_backend": "cpu", "num_envs": 1})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_pushcube_sac(
    config,
    output_root,
    *,
    show_progress=True,
    resume=None,
):
    if resume is not None:
        raise ValueError(
            "Exact ManiSkill training resume is unsupported: "
            "replay, simulator, controller and environment RNG "
            "are not persisted."
        )

    device = check_device(
        config.training.device
    )

    run = (
        Path(output_root)
        / config.name
        / f"seed_{config.seed}"
        / (
            datetime.now(
                timezone.utc
            ).strftime("%Y%m%dT%H%M%S")
            + "_"
            + uuid4().hex[:8]
        )
    )

    run.mkdir(
        parents=True,
        exist_ok=False,
    )
    (run / "metrics").mkdir()

    (
        run / "config.yaml"
    ).write_text(
        yaml.safe_dump(
            config.to_dict(),
            sort_keys=False,
        )
    )

    counters = {
        "real_env_steps": 0,
        "policy_gradient_steps": 0,
        "episodes": 0,
    }

    metadata = {
        "status": "running",
        "algorithm": "sac",
        "seed": config.seed,
        "backend": "maniskill",
        "device": str(device),
        "sim_backend": config.env.sim_backend,
        "num_envs": config.env.num_envs,
        "utd": config.training.utd,
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "cuda_build": torch.version.cuda,
        "replay_persisted": False,
        "simulator_persisted": False,
        "training_resume_supported": False,
        "versions": {
            name: version(name)
            for name in (
                "mani-skill",
                "sapien",
                "torch",
                "numpy",
                "gymnasium",
            )
        },
    }

    previous_threads = torch.get_num_threads()

    env = None
    tracker = None
    start = time.perf_counter()

    success_terminations = 0
    evaluation_seconds = 0.0

    try:
        torch.set_num_threads(
            config.training.torch_threads
        )

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(
                config.seed
            )

        env = make_env(
            ExperimentConfig(
                config.env,
                None,
                config.seed,
            )
        )

        num_envs = _num_envs(
            env,
            config,
        )

        if (
            config.training.real_env_steps
            % num_envs
            != 0
        ):
            raise ValueError(
                "training.real_env_steps must be divisible "
                f"by num_envs={num_envs}; got "
                f"{config.training.real_env_steps}"
            )

        obs_space = _single_observation_space(
            env
        )
        action_space = _single_action_space(
            env
        )

        if len(obs_space.shape) != 1:
            raise ValueError(
                "State SAC requires flat observations; "
                f"got {obs_space.shape}"
            )

        if len(action_space.shape) != 1:
            raise ValueError(
                "SAC requires flat continuous actions; "
                f"got {action_space.shape}"
            )

        contract = _training_contract(
            config,
            env,
            num_envs=num_envs,
        )

        metadata[
            "environment_contract"
        ] = contract
        metadata["training_environment_contract"] = contract
        metadata["evaluation_environment_contract"] = {**contract, "sim_backend": "cpu",
            "num_envs": 1, "automatic_reset": False, "termination_policy": "terminate_on_success"}

        from dynamics_shift.experiments.train_sac_source import _git_metadata
        metadata.update(_git_metadata())
        with isolated_rng(device):
            tracker = Tracker(config, run)
            tracker.metadata(metadata)

        learner = SACLearner(
            obs_dim=contract[
                "observation_dim"
            ],
            action_low=action_space.low,
            action_high=action_space.high,
            config=config.algo,
            device=device,
        )

        for module in (
            learner.actor,
            learner.critic,
            learner.target_critic,
        ):
            if any(
                parameter.device != device
                for parameter
                in module.parameters()
            ):
                raise RuntimeError(
                    "Learner parameter device mismatch"
                )

        if learner.log_alpha.device != device:
            raise RuntimeError(
                "Learner alpha device mismatch"
            )

        replay = ReplayBuffer(
            config.training.replay_capacity,
            learner.obs_dim,
            contract["action_dim"],
            seed=config.seed,
        )

        obs, _ = env.reset(
            seed=config.seed
        )

        _batch_size_from_obs(
            obs,
            num_envs,
        )

        # Per-environment statistics.
        episode_returns = np.zeros(
            num_envs,
            dtype=np.float64,
        )
        episode_lengths = np.zeros(
            num_envs,
            dtype=np.int64,
        )
        episode_success_once = np.zeros(
            num_envs,
            dtype=np.bool_,
        )

        # Fractional UTD accumulator.
        update_budget = 0.0

        def persist(
            name: str,
            *,
            overwrite: bool = False,
        ) -> None:
            if num_envs == 1:
                probe_obs = (
                    _to_numpy(obs)
                    .astype(
                        np.float32,
                        copy=False,
                    )
                    .reshape(-1)
                )
            else:
                probe_obs = (
                    _to_numpy(obs)[0]
                    .astype(
                        np.float32,
                        copy=False,
                    )
                    .reshape(-1)
                )

            probe_action = learner.act(
                probe_obs,
                deterministic=True,
            )

            probe = {
                "observation":
                    probe_obs.tolist(),
                "action":
                    probe_action.tolist(),
            }

            save_checkpoint(
                run
                / "checkpoints"
                / name,
                learner,
                counters,
                config.to_dict(),
                environment_contract=contract,
                training_environment_contract=contract,
                evaluation_environment_contract={**contract, "sim_backend": "cpu", "num_envs": 1,
                    "automatic_reset": False, "termination_policy": "terminate_on_success"},
                learner_probe=probe,
                overwrite=overwrite,
            )

        def evaluate_saved(
            name: str,
            output_name: str,
        ) -> dict:
            nonlocal evaluation_seconds

            began = time.perf_counter()

            summary = _evaluate_saved_checkpoint(
                run
                / "checkpoints"
                / name,
                run
                / "metrics"
                / output_name,
                config,
                device=device,
            )

            evaluation_seconds += (
                time.perf_counter()
                - began
            )

            record = {
                "real_env_steps":
                    counters[
                        "real_env_steps"
                    ],
                "policy_gradient_steps":
                    counters[
                        "policy_gradient_steps"
                    ],
                "wall_time_seconds":
                    time.perf_counter()
                    - start,
                **summary,
            }

            with (
                run
                / "metrics"
                / "evaluation_history.jsonl"
            ).open("a") as stream:
                stream.write(
                    json.dumps(record)
                    + "\n"
                )

            print(
                json.dumps(
                    {
                        "event":
                            "pushcube_evaluation",
                        "real_env_steps":
                            counters[
                                "real_env_steps"
                            ],
                        "mean_return":
                            summary[
                                "mean_return"
                            ],
                        "success_once":
                            summary[
                                "success_once"
                            ],
                        "success_at_end":
                            summary[
                                "success_at_end"
                            ],
                    }
                ),
                flush=True,
            )

            with isolated_rng(device):
                tracker.scalars("eval", summary, counters["real_env_steps"])
                tracker.checkpoint_video(run / "checkpoints" / name, config, counters["real_env_steps"])
            return summary

        persist("initial.pt")
        persist(
            "latest.pt",
            overwrite=True,
        )

        if config.evaluation.interval:
            evaluate_saved(
                "initial.pt",
                "evaluation_step_0",
            )

        fields = [
            *counters,
            "replay_size",
            "finished_episodes",
            "episode_return",
            "episode_length",
            "success_once",
            "success_at_end",
            "terminated",
            "truncated",
            "actor_loss",
            "critic_loss",
            "alpha_loss",
            "alpha",
            "utd",
            "update_budget",
            "wall_time_seconds",
            "evaluation_seconds",
            "transitions_per_second",
            "training_transitions_per_second",
        ]

        next_log = config.training.log_every
        next_checkpoint = (
            config.training.checkpoint_every
        )

        next_eval = (
            config.evaluation.interval
            if config.evaluation.interval > 0
            else None
        )

        with (
            run / "metrics" / "train.csv"
        ).open(
            "x",
            newline="",
        ) as stream, tqdm(
            total=
                config.training.real_env_steps,
            desc="PushCube SAC",
            disable=not show_progress,
        ) as progress:

            writer = csv.DictWriter(
                stream,
                fieldnames=fields,
            )
            writer.writeheader()

            while (
                counters["real_env_steps"]
                < config.training.real_env_steps
            ):
                current_steps = counters[
                    "real_env_steps"
                ]

                if (
                    current_steps
                    < config.training.learning_starts
                ):
                    action = _random_action(
                        env,
                        obs,
                        num_envs=num_envs,
                        device=device,
                    )
                else:
                    action = _policy_action(
                        learner,
                        obs,
                    )

                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                real_next_obs = (
                    _true_next_observation(
                        next_obs,
                        terminated,
                        truncated,
                        info,
                    )
                    if num_envs > 1
                    else next_obs
                )

                _add_replay_batch(
                    replay,
                    obs,
                    action,
                    reward,
                    real_next_obs,
                    terminated,
                    truncated,
                    num_envs=num_envs,
                )

                reward_np = (
                    _to_numpy(reward)
                    .astype(
                        np.float64,
                        copy=False,
                    )
                    .reshape(num_envs)
                )

                terminated_np = (
                    _to_numpy(terminated)
                    .astype(
                        np.bool_,
                        copy=False,
                    )
                    .reshape(num_envs)
                )

                truncated_np = (
                    _to_numpy(truncated)
                    .astype(
                        np.bool_,
                        copy=False,
                    )
                    .reshape(num_envs)
                )

                success_np = (
                    _extract_vector_success(
                        info,
                        terminated,
                        num_envs,
                    )
                )

                episode_returns += reward_np
                episode_lengths += 1
                episode_success_once |= (
                    success_np
                )

                done_np = (
                    terminated_np
                    | truncated_np
                )

                finished_indices = np.flatnonzero(
                    done_np
                )

                finished_episode_count = (
                    len(finished_indices)
                )

                completed_return = None
                completed_length = None
                completed_success_once = None
                completed_success_at_end = None
                completed_terminated = None
                completed_truncated = None

                if finished_episode_count:
                    completed_return = float(
                        np.mean(
                            episode_returns[
                                finished_indices
                            ]
                        )
                    )

                    completed_length = float(
                        np.mean(
                            episode_lengths[
                                finished_indices
                            ]
                        )
                    )

                    completed_success_once = float(
                        np.mean(
                            episode_success_once[
                                finished_indices
                            ]
                        )
                    )

                    completed_success_at_end = float(
                        np.mean(
                            success_np[
                                finished_indices
                            ]
                        )
                    )

                    completed_terminated = int(
                        terminated_np[
                            finished_indices
                        ].sum()
                    )

                    completed_truncated = int(
                        truncated_np[
                            finished_indices
                        ].sum()
                    )

                    counters[
                        "episodes"
                    ] += finished_episode_count

                    success_terminations += int(
                        np.logical_and(
                            terminated_np,
                            success_np,
                        ).sum()
                    )

                    episode_returns[
                        finished_indices
                    ] = 0.0

                    episode_lengths[
                        finished_indices
                    ] = 0

                    episode_success_once[
                        finished_indices
                    ] = False

                previous_steps = counters[
                    "real_env_steps"
                ]

                counters[
                    "real_env_steps"
                ] += num_envs

                # ----------------------------------------------------------
                # UTD scheduler
                # ----------------------------------------------------------
                losses = {}

                if (
                    counters[
                        "real_env_steps"
                    ]
                    >= config.training.learning_starts
                    and len(replay)
                    >= config.training.batch_size
                ):
                    # UTD is gradient updates per newly collected transition.
                    update_budget += (
                        num_envs
                        * config.training.utd
                    )

                    while update_budget >= 1.0:
                        losses = learner.update(
                            replay.sample(
                                config.training.batch_size
                            )
                        )

                        if not all(
                            np.isfinite(value)
                            for value
                            in losses.values()
                        ):
                            raise FloatingPointError(
                                "Nonfinite SAC losses at "
                                f"transition "
                                f"{counters['real_env_steps']}: "
                                f"{losses}"
                            )

                        update_budget -= 1.0

                counters[
                    "policy_gradient_steps"
                ] = (
                    learner.policy_gradient_steps
                )

                elapsed = (
                    time.perf_counter()
                    - start
                )

                log_due = (
                    counters[
                        "real_env_steps"
                    ]
                    >= next_log
                )

                if (
                    log_due
                    or finished_episode_count
                ):
                    with isolated_rng(device):
                        tracker.scalars("train", {**counters, **losses, "episode_return": completed_return,
                            "episode_length": completed_length, "success_once": completed_success_once,
                            "success_at_end": completed_success_at_end}, counters["real_env_steps"])
                    writer.writerow(
                        {
                            **counters,
                            "replay_size":
                                len(replay),
                            "finished_episodes":
                                finished_episode_count,
                            "episode_return":
                                completed_return,
                            "episode_length":
                                completed_length,
                            "success_once":
                                completed_success_once,
                            "success_at_end":
                                completed_success_at_end,
                            "terminated":
                                completed_terminated,
                            "truncated":
                                completed_truncated,
                            **losses,
                            "utd":
                                config.training.utd,
                            "update_budget":
                                update_budget,
                            "wall_time_seconds":
                                elapsed,
                            "evaluation_seconds":
                                evaluation_seconds,
                            "transitions_per_second":
                                counters[
                                    "real_env_steps"
                                ]
                                / max(
                                    elapsed,
                                    1e-9,
                                ),
                            "training_transitions_per_second":
                                counters[
                                    "real_env_steps"
                                ]
                                / max(
                                    elapsed
                                    - evaluation_seconds,
                                    1e-9,
                                ),
                        }
                    )

                    stream.flush()

                while (
                    counters[
                        "real_env_steps"
                    ]
                    >= next_log
                ):
                    next_log += (
                        config.training.log_every
                    )

                # ManiSkillVectorEnv automatically resets completed envs.
                obs = next_obs

                # Legacy CPU wrapper does not auto-reset.
                if (
                    num_envs == 1
                    and bool(done_np[0])
                ):
                    obs, _ = env.reset()

                # ----------------------------------------------------------
                # Checkpoint scheduling based on transition count.
                # Vector collection can cross a requested boundary by <N.
                # ----------------------------------------------------------
                checkpoint_due = (
                    counters[
                        "real_env_steps"
                    ]
                    >= next_checkpoint
                )

                eval_due = (
                    next_eval is not None
                    and counters[
                        "real_env_steps"
                    ]
                    >= next_eval
                    and counters[
                        "real_env_steps"
                    ]
                    < config.training.real_env_steps
                )

                checkpoint_name = (
                    f"step_"
                    f"{counters['real_env_steps']}.pt"
                )

                if checkpoint_due:
                    persist(
                        "latest.pt",
                        overwrite=True,
                    )

                if checkpoint_due or eval_due:
                    checkpoint_path = (
                        run
                        / "checkpoints"
                        / checkpoint_name
                    )

                    if not checkpoint_path.exists():
                        persist(
                            checkpoint_name
                        )

                if eval_due:
                    evaluate_saved(
                        checkpoint_name,
                        "evaluation_step_"
                        f"{counters['real_env_steps']}",
                    )

                while (
                    counters[
                        "real_env_steps"
                    ]
                    >= next_checkpoint
                ):
                    next_checkpoint += (
                        config.training.checkpoint_every
                    )

                if next_eval is not None:
                    while (
                        counters[
                            "real_env_steps"
                        ]
                        >= next_eval
                    ):
                        next_eval += (
                            config.evaluation.interval
                        )

                progress.update(
                    counters[
                        "real_env_steps"
                    ]
                    - previous_steps
                )

        metadata[
            "partial_episodes"
        ] = {
            "num_envs": num_envs,
            "lengths":
                episode_lengths.tolist(),
            "returns":
                episode_returns.tolist(),
            "success_once":
                episode_success_once.tolist(),
        }

        persist("final.pt")
        persist(
            "latest.pt",
            overwrite=True,
        )

        summary = evaluate_saved(
            "final.pt",
            "evaluation",
        )

        metadata.update(
            status="complete",
            evaluation=summary,
            success_termination_episodes=
                success_terminations,
            success_termination=(
                "verified"
                if success_terminations
                else (
                    "DISABLED: training suppresses termination; see success_once/success_at_end"
                    if config.env.sim_backend == "gpu" else
                    "UNVERIFIED: no successful termination observed"
                )
            ),
        )

        print(
            json.dumps(
                {
                    "event":
                        "pushcube_sac_complete",
                    "run_dir":
                        str(run.resolve()),
                    **counters,
                    "success_termination":
                        metadata[
                            "success_termination"
                        ],
                }
            ),
            flush=True,
        )

    except Exception as error:
        metadata.update(
            status="failed",
            error=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )
        raise

    finally:
        if env is not None:
            env.close()

        torch.set_num_threads(
            previous_threads
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        metadata.update(
            counters,
            elapsed_seconds=elapsed,
            wall_time_seconds=elapsed,
            evaluation_seconds=evaluation_seconds,
            transitions_per_second=(
                counters["real_env_steps"]
                / max(elapsed, 1e-9)
            ),
            training_transitions_per_second=(
                counters["real_env_steps"]
                / max(
                    elapsed
                    - evaluation_seconds,
                    1e-9,
                )
            ),
        )

        if tracker is not None:
            tracker.metadata(metadata)
            tracker.finish(failed=metadata["status"] == "failed")
        (
            run / "metadata.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
            )
            + "\n"
        )

    return run