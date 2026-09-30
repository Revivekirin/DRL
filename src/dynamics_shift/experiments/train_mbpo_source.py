"""Nominal MBPO orchestration; no shift events or target environments."""

import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import random
import signal
import time
from uuid import uuid4

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dynamics_shift.algorithms.sac.learner import SACLearner
from dynamics_shift.algorithms.mbpo.config import MBPORunConfig
from dynamics_shift.algorithms.mbpo.rollouts import generate_rollouts, mixed_batch
from dynamics_shift.algorithms.mbpo.checkpoint import (
    save_mbpo_checkpoint,
    load_mbpo_checkpoint,
)
from dynamics_shift.data.model_data import (
    RealReplayBuffer,
    ModelReplayBuffer,
    ModelDataset,
)
from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble
from dynamics_shift.config import ExperimentConfig
from dynamics_shift.envs import make_env
from dynamics_shift.utils.device import check_device
from dynamics_shift.utils.tracking import Tracker
from dynamics_shift.utils.checkpoint import capture_rng, restore_rng
from dynamics_shift.utils.training_state import restore_training_state
from dynamics_shift.experiments.train_sac_source import _git_metadata


def train_mbpo_source(
    config: MBPORunConfig,
    output_root: str | Path = "outputs",
    *,
    resume: str | Path | None = None,
    show_progress: bool = True,
) -> Path:
    device = check_device(config.training.device)

    restored = load_mbpo_checkpoint(resume, device) if resume else None

    # ------------------------------------------------------------------
    # Resume compatibility check
    # ------------------------------------------------------------------
    if restored:
        old = MBPORunConfig.from_dict(restored[2]["config"]).to_dict()
        new = config.to_dict()

        for settings in (old, new):
            # Tracking settings do not alter the learning algorithm.
            # This allows, for example, offline -> online W&B on resume.
            settings.pop("tracking", None)

            for key in (
                "real_env_steps",
                "device",
                "torch_threads",
                "log_every",
                "checkpoint_every",
            ):
                settings["training"].pop(key)

        if (
            old != new
            or restored[2]["counters"]["real_env_steps"]
            > config.training.real_env_steps
        ):
            raise ValueError(
                "Resume config differs or total budget precedes checkpoint"
            )

    # ------------------------------------------------------------------
    # Run directory
    # ------------------------------------------------------------------
    run = (
        Path(output_root)
        / config.name
        / "seed_0"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "_"
            + uuid4().hex[:8]
        )
    )

    run.mkdir(parents=True, exist_ok=False)
    (run / "metrics").mkdir()

    (run / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False)
    )

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------
    counters = dict(
        real_env_steps=0,
        policy_gradient_steps=0,
        dynamics_model_train_steps=0,
        dynamics_model_refit_count=0,
        synthetic_transition_count=0,
        episodes=0,
        real_policy_samples=0,
        synthetic_policy_samples=0,
    )

    metadata = {
        **_git_metadata(),
        "algorithm": "mbpo",
        "seed": 0,
        "environment": config.env.id,
        "actuator_strength": 1.0,
        "device": str(device),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            name: version(name)
            for name in ("torch", "gymnasium", "mujoco", "numpy")
        },
        "resume_from": str(resume) if resume else None,
        "status": "running",
    }

    env = None
    tracker = None

    previous_threads = torch.get_num_threads()

    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    handlers = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGTERM, signal.SIGUSR1)
    }

    start = time.perf_counter()
    model_seconds = 0.0

    try:
        # --------------------------------------------------------------
        # Deterministic setup
        # --------------------------------------------------------------
        torch.set_num_threads(config.training.torch_threads)

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        rng = np.random.default_rng(config.seed)

        # --------------------------------------------------------------
        # Environment
        # --------------------------------------------------------------
        env = make_env(
            ExperimentConfig(
                config.env,
                config.dynamics,
                config.seed,
            )
        )

        if env.get_shift_parameter("actuator_strength") != 1.0:
            raise ValueError(
                "MBPO source training must remain nominal"
            )

        obs, _ = env.reset(seed=config.seed)

        od = env.observation_space.shape[0]
        ad = env.action_space.shape[0]

        # --------------------------------------------------------------
        # Replay buffers
        # --------------------------------------------------------------
        real = RealReplayBuffer(
            config.training.replay_capacity,
            od,
            ad,
            config.seed,
        )

        synthetic = ModelReplayBuffer(
            config.mbpo.model_replay_capacity,
            od,
            ad,
            config.seed + 1,
        )

        # --------------------------------------------------------------
        # Learner / dynamics model
        # --------------------------------------------------------------
        learner = (
            restored[0]
            if restored
            else SACLearner(
                od,
                env.action_space.low,
                env.action_space.high,
                config.algo,
                device,
            )
        )

        model = (
            restored[1]
            if restored
            else ProbabilisticEnsemble(
                od,
                ad,
                config.mbpo,
                device,
                config.seed,
            )
        )

        episode_return = 0.0
        episode_length = 0

        payload = restored[2] if restored else None

        # --------------------------------------------------------------
        # Restore continuation state
        # --------------------------------------------------------------
        if restored:
            counters = dict(payload["counters"])

            state = payload["training_state"]

            obs, episode_return, episode_length = restore_training_state(
                env,
                real,
                state,
            )

            synthetic.load_state_dict(
                state["mbpo"]["synthetic_replay"]
            )

            rng.bit_generator.state = state["mbpo"]["runner_rng"]

        # --------------------------------------------------------------
        # W&B Tracker
        #
        # wandb.init() may touch random state. As in SAC source training,
        # preserve/restore RNG so enabling W&B does not alter learning.
        # --------------------------------------------------------------
        rng_before_tracking = capture_rng(learner.device)

        tracker = Tracker(
            config,
            run,
            resume,
        )

        restore_rng(
            payload["rng"]
            if payload is not None
            else rng_before_tracking,
            learner.device,
        )

        # --------------------------------------------------------------
        # Checkpoint helper
        # --------------------------------------------------------------
        def save(path, overwrite=False):
            save_mbpo_checkpoint(
                path,
                learner,
                model,
                real,
                synthetic,
                env,
                obs,
                episode_return,
                episode_length,
                counters,
                config,
                rng,
                overwrite=overwrite,
            )

        latest = run / "checkpoints/latest.pt"

        save(latest)

        # --------------------------------------------------------------
        # CSV fields
        # --------------------------------------------------------------
        train_fields = [
            *counters,
            "real_replay_size",
            "model_replay_size",
            "episode_return",
            "episode_length",
            "actor_loss",
            "critic_loss",
            "alpha_loss",
            "alpha",
            "elapsed_seconds",
            "model_training_seconds",
        ]

        model_fields = [
            "real_env_steps",
            "dynamics_model_refit_count",
            "dynamics_model_train_steps",
            "member",
            "model_train_loss",
            "model_validation_loss",
            "model_validation_rmse",
            "elite",
            "train_samples",
            "holdout_samples",
        ]

        # --------------------------------------------------------------
        # Training
        # --------------------------------------------------------------
        with (
            (run / "metrics/train.csv").open(
                "x",
                newline="",
            ) as train_file,
            (run / "metrics/model.csv").open(
                "x",
                newline="",
            ) as model_file,
            tqdm(
                total=config.training.real_env_steps,
                initial=counters["real_env_steps"],
                desc="MBPO nominal",
                unit="env step",
                disable=not show_progress,
                mininterval=1.0,
            ) as progress,
        ):
            tw = csv.DictWriter(
                train_file,
                fieldnames=train_fields,
            )

            mw = csv.DictWriter(
                model_file,
                fieldnames=model_fields,
            )

            tw.writeheader()
            mw.writeheader()

            for _ in range(
                counters["real_env_steps"],
                config.training.real_env_steps,
            ):
                # ------------------------------------------------------
                # Environment interaction
                # ------------------------------------------------------
                if (
                    counters["real_env_steps"]
                    < config.training.learning_starts
                ):
                    action = env.action_space.sample()
                else:
                    action = learner.act(obs)

                no, reward, terminated, truncated, _ = env.step(action)

                real.add(
                    obs,
                    action,
                    reward,
                    no,
                    terminated,
                    truncated,
                )

                counters["real_env_steps"] += 1

                episode_return += float(reward)
                episode_length += 1

                current = counters["real_env_steps"]

                # ------------------------------------------------------
                # Dynamics model fitting + synthetic rollout
                # ------------------------------------------------------
                if (
                    current >= config.training.learning_starts
                    and (
                        current - config.training.learning_starts
                    )
                    % config.mbpo.model_train_frequency
                    == 0
                ):
                    fit_start = time.perf_counter()

                    dataset = ModelDataset.from_real_replay(
                        real,
                        rng,
                        config.mbpo.holdout_ratio,
                        config.mbpo.model_max_samples,
                    )

                    metrics = model.train(dataset)

                    model_seconds += (
                        time.perf_counter() - fit_start
                    )

                    counters[
                        "dynamics_model_train_steps"
                    ] = model.train_steps

                    counters[
                        "dynamics_model_refit_count"
                    ] = model.refit_count

                    # --------------------------------------------------
                    # Per-member CSV model metrics
                    # --------------------------------------------------
                    for member in range(
                        config.mbpo.ensemble_size
                    ):
                        mw.writerow(
                            {
                                "real_env_steps": current,
                                "dynamics_model_refit_count":
                                    model.refit_count,
                                "dynamics_model_train_steps":
                                    model.train_steps,
                                "member": member,
                                **{
                                    k: metrics[k][member]
                                    for k in (
                                        "model_train_loss",
                                        "model_validation_loss",
                                        "model_validation_rmse",
                                    )
                                },
                                "elite":
                                    member in model.elites,
                                "train_samples":
                                    metrics["train_samples"],
                                "holdout_samples":
                                    metrics["holdout_samples"],
                            }
                        )

                    model_file.flush()

                    # --------------------------------------------------
                    # Discard stale synthetic samples after refit
                    # --------------------------------------------------
                    synthetic.clear()

                    generated = generate_rollouts(
                        learner,
                        model,
                        real,
                        synthetic,
                        config.mbpo.rollout_batch_size,
                        config.mbpo.rollout_horizon,
                        rng,
                    )

                    counters[
                        "synthetic_transition_count"
                    ] += generated

                    # --------------------------------------------------
                    # W&B: dynamics model / rollout diagnostics
                    # --------------------------------------------------
                    elite_indices = list(model.elites)

                    model_train_loss_mean = float(
                        np.mean(
                            metrics["model_train_loss"]
                        )
                    )

                    model_validation_loss_mean = float(
                        np.mean(
                            metrics["model_validation_loss"]
                        )
                    )

                    model_validation_rmse_mean = float(
                        np.mean(
                            metrics["model_validation_rmse"]
                        )
                    )

                    if elite_indices:
                        elite_validation_loss_mean = float(
                            np.mean(
                                [
                                    metrics[
                                        "model_validation_loss"
                                    ][i]
                                    for i in elite_indices
                                ]
                            )
                        )

                        elite_validation_rmse_mean = float(
                            np.mean(
                                [
                                    metrics[
                                        "model_validation_rmse"
                                    ][i]
                                    for i in elite_indices
                                ]
                            )
                        )
                    else:
                        elite_validation_loss_mean = (
                            model_validation_loss_mean
                        )

                        elite_validation_rmse_mean = (
                            model_validation_rmse_mean
                        )

                    tracker.log(
                        {
                            "train/model_train_loss_mean":
                                model_train_loss_mean,

                            "train/model_validation_loss_mean":
                                model_validation_loss_mean,

                            "train/model_validation_rmse_mean":
                                model_validation_rmse_mean,

                            "train/model_elite_validation_loss_mean":
                                elite_validation_loss_mean,

                            "train/model_elite_validation_rmse_mean":
                                elite_validation_rmse_mean,

                            "train/dynamics_model_train_steps":
                                counters[
                                    "dynamics_model_train_steps"
                                ],

                            "train/dynamics_model_refit_count":
                                counters[
                                    "dynamics_model_refit_count"
                                ],

                            "train/model_train_samples":
                                int(
                                    metrics["train_samples"]
                                ),

                            "train/model_holdout_samples":
                                int(
                                    metrics["holdout_samples"]
                                ),

                            "train/model_training_seconds":
                                model_seconds,

                            "train/generated_transitions":
                                int(generated),

                            "train/synthetic_transition_count":
                                counters[
                                    "synthetic_transition_count"
                                ],

                            "train/model_replay_size":
                                len(synthetic),
                        },
                        current,
                    )

                # ------------------------------------------------------
                # SAC policy updates using real + model replay
                # ------------------------------------------------------
                losses = {}

                if (
                    len(synthetic)
                    and len(real)
                    >= config.training.batch_size
                ):
                    for _ in range(
                        config.training.updates_per_env_step
                    ):
                        batch, nr, ns = mixed_batch(
                            real,
                            synthetic,
                            config.training.batch_size,
                            config.mbpo.real_ratio,
                            rng,
                        )

                        losses = learner.update(batch)

                        counters[
                            "real_policy_samples"
                        ] += nr

                        counters[
                            "synthetic_policy_samples"
                        ] += ns

                counters[
                    "policy_gradient_steps"
                ] = learner.policy_gradient_steps

                # ------------------------------------------------------
                # Episode boundary
                # ------------------------------------------------------
                finished = terminated or truncated

                counters["episodes"] += int(finished)

                # ------------------------------------------------------
                # CSV + W&B training metrics
                # ------------------------------------------------------
                if (
                    finished
                    or current
                    % config.training.log_every
                    == 0
                    or current
                    == config.training.real_env_steps
                ):
                    elapsed_seconds = (
                        time.perf_counter() - start
                    )

                    tw.writerow(
                        {
                            **counters,
                            **losses,
                            "real_replay_size":
                                len(real),
                            "model_replay_size":
                                len(synthetic),

                            "episode_return":
                                episode_return
                                if finished
                                else "",

                            "episode_length":
                                episode_length
                                if finished
                                else "",

                            "elapsed_seconds":
                                elapsed_seconds,

                            "model_training_seconds":
                                model_seconds,
                        }
                    )

                    train_file.flush()

                    tracker.log(
                        {
                            "train/policy_gradient_steps":
                                counters[
                                    "policy_gradient_steps"
                                ],

                            "train/episodes":
                                counters["episodes"],

                            "train/real_replay_size":
                                len(real),

                            "train/model_replay_size":
                                len(synthetic),

                            "train/dynamics_model_train_steps":
                                counters[
                                    "dynamics_model_train_steps"
                                ],

                            "train/dynamics_model_refit_count":
                                counters[
                                    "dynamics_model_refit_count"
                                ],

                            "train/synthetic_transition_count":
                                counters[
                                    "synthetic_transition_count"
                                ],

                            "train/real_policy_samples":
                                counters[
                                    "real_policy_samples"
                                ],

                            "train/synthetic_policy_samples":
                                counters[
                                    "synthetic_policy_samples"
                                ],

                            "train/model_training_seconds":
                                model_seconds,

                            "train/elapsed_seconds":
                                elapsed_seconds,

                            **{
                                f"train/{k}": float(v)
                                for k, v in losses.items()
                            },

                            **(
                                {
                                    "train/episode_return":
                                        episode_return,

                                    "train/episode_length":
                                        episode_length,
                                }
                                if finished
                                else {}
                            ),
                        },
                        current,
                    )

                    progress.set_postfix(
                        updates=learner.policy_gradient_steps,
                        refits=model.refit_count,
                        synthetic=counters[
                            "synthetic_transition_count"
                        ],
                        refresh=False,
                    )

                # ------------------------------------------------------
                # Advance state / reset episode
                # ------------------------------------------------------
                obs = no

                if finished:
                    obs, _ = env.reset()

                    episode_return = 0.0
                    episode_length = 0

                progress.update(1)

                # ------------------------------------------------------
                # Checkpoint
                # ------------------------------------------------------
                if (
                    current
                    % config.training.checkpoint_every
                    == 0
                    or stopped
                ):
                    save(
                        latest,
                        overwrite=True,
                    )

                if stopped:
                    metadata["status"] = "interrupted"
                    return run

        # ------------------------------------------------------------------
        # Final checkpoint
        # ------------------------------------------------------------------
        save(
            run / "checkpoints/final.pt"
        )

        save(
            latest,
            overwrite=True,
        )

        metadata["status"] = "complete"

    except Exception as error:
        metadata.update(
            status="failed",
            error=(
                f"{type(error).__name__}: {error}"
            ),
        )
        raise

    finally:
        # --------------------------------------------------------------
        # Restore signal handlers
        # --------------------------------------------------------------
        for sig, handler in handlers.items():
            signal.signal(
                sig,
                handler,
            )

        # --------------------------------------------------------------
        # Close environment
        # --------------------------------------------------------------
        if env is not None:
            env.close()

        torch.set_num_threads(
            previous_threads
        )

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------
        metadata.update(counters)

        metadata.update(
            ended_at=datetime.now(
                timezone.utc
            ).isoformat(),
            elapsed_seconds=(
                time.perf_counter() - start
            ),
            model_training_seconds=model_seconds,
        )

        (run / "metadata.json").write_text(
            json.dumps(
                metadata,
                indent=2,
            )
            + "\n"
        )

        # --------------------------------------------------------------
        # Finish W&B cleanly
        # --------------------------------------------------------------
        if tracker is not None:
            tracker.finish(
                failed=(
                    metadata["status"]
                    == "failed"
                )
            )

    return run