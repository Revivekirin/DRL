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
from .dispatch import save_resolved_config
from .records import record_update, record_transition
from .artifacts import save_learner_artifact

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


from .interaction import (
    _to_numpy, _single_observation_space, _single_action_space, _num_envs, _horizon, _batch_size_from_obs, _deterministic_action, _policy_action, _random_action, _extract_vector_success, _true_next_observation, _add_replay_batch, _training_contract, _evaluate_saved_checkpoint, collect_transition,
)

def train_sac(
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

    save_resolved_config(run, config, 'sac')

    counters = {
        "real_env_steps": 0,
        "policy_gradient_steps": 0,
        "episodes": 0,
        "vector_steps": 0,
        "real_policy_samples": 0,
        "synthetic_policy_samples": 0,
        "synthetic_transition_count": 0,
        "dynamics_model_refit_count": 0,
        "dynamics_model_train_steps": 0,
        "real_truncations": 0,
        "final_observation_checks": 0,
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

        from dynamics_shift.experiments.provenance import _git_metadata
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

        learner.record_update_diagnostics = True

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
            save_learner_artifact(run/'checkpoints'/name, learner, counters,
                                  config, contract, obs, overwrite=overwrite)

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
            "entropy_estimate",
            "policy_q_min_mean",
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

                next_obs, reward, terminated, truncated, info = collect_transition(
                    env, replay, obs, action, num_envs=num_envs,
                    automatic_reset=contract['automatic_reset'])

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
                    with (run/'metrics/episodes.jsonl').open('a') as episode_file:
                        for index in finished_indices:
                            episode_file.write(json.dumps(dict(
                                real_env_steps=counters['real_env_steps']+num_envs,
                                env_index=int(index), episode_return=float(episode_returns[index]),
                                episode_length=int(episode_lengths[index]),
                                success_once=int(episode_success_once[index]),
                                success_at_end=int(success_np[index]),
                                terminated=bool(terminated_np[index]), truncated=bool(truncated_np[index])))+'\n')
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

                counters['vector_steps'] += 1
                counters['real_truncations'] += int(truncated_np.sum())
                counters['final_observation_checks'] += int(done_np.sum()) if contract['automatic_reset'] else 0

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

                        counters['policy_gradient_steps'] = learner.policy_gradient_steps
                        counters['real_policy_samples'] += config.training.batch_size
                        record_update(tracker, counters, losses,
                            real_count=config.training.batch_size, synthetic_count=0,
                            requested_real_ratio=1.0, real_replay_size=len(replay), model_replay_size=0,
                            wall_time=time.perf_counter()-start)
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

                record_transition(tracker, run/'metrics/train.jsonl', counters, losses, elapsed,
                    replay_size=len(replay), update_budget=update_budget,
                    loss_aggregation='last update in this vector call')

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
                    if finished_episode_count:
                        with isolated_rng(device):
                            tracker.scalars("train", {"episode_return": completed_return,
                                "episode_length": completed_length, "success_once": completed_success_once,
                                "success_at_end": completed_success_at_end,
                                "finished_episodes": int(finished_episode_count)}, counters["real_env_steps"])
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
# Historical Python API alias.
train_pushcube_sac = train_sac
