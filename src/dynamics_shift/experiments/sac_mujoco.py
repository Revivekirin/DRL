"""Nominal real-environment SAC orchestration; losses remain batch-only."""
from .dispatch import save_resolved_config
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import signal
import subprocess
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
from dynamics_shift.experiments.config import RunConfig
from dynamics_shift.experiments.evaluate_sac_shift import evaluate_checkpoint
from dynamics_shift.utils.checkpoint import save_checkpoint, load_checkpoint, restore_rng, capture_rng
from dynamics_shift.utils.tracking import Tracker
from dynamics_shift.utils.device import check_device
from .interaction import collect_transition
from .records import record_update, record_transition


from .provenance import _git_metadata


def train_source(config: RunConfig, output_root: str | Path = "outputs", *,
                 show_progress: bool = True, resume: str | Path | None = None) -> Path:
    if config.env.backend == "maniskill":
        from .train_pushcube_sac import train_pushcube_sac
        return train_pushcube_sac(config, output_root, show_progress=show_progress, resume=resume)
    from dynamics_shift.utils.training_state import capture_training_state, restore_training_state
    from dynamics_shift.evaluation.contracts import halfcheetah_contract

    device = check_device(config.training.device)
    restored_learner, payload = (load_checkpoint(resume, device=device) if resume else (None, None))
    if payload is not None:
        if payload.get("training_state") is None:
            raise ValueError("This checkpoint lacks replay/simulator state and cannot resume training")
        old, new = RunConfig.from_dict(payload["config"]).to_dict(), config.to_dict()
        for settings in (old, new):
            settings.pop("tracking")
            for key in ("real_env_steps", "device", "log_every", "torch_threads", "checkpoint_every"):
                settings["training"].pop(key)
        if old != new:
            raise ValueError("Resume configuration changes the learning algorithm, replay or environment")
        if payload["counters"]["real_env_steps"] > config.training.real_env_steps:
            raise ValueError("Total real_env_steps must be at least the saved count")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    run_dir = Path(output_root) / config.name / f"seed_{config.seed}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics").mkdir()
    save_resolved_config(run_dir, config, 'sac')
    metadata = {**_git_metadata(), "algorithm": "sac", "seed": config.seed,
                "environment": config.env.id, "source_dynamics": {"actuator_scale": 1.0},
                "target_dynamics": {"actuator_scale": config.evaluation.target_actuator_scale},
                "python": platform.python_version(), "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cuda_build": torch.version.cuda,
                "versions": {name: version(name) for name in ("gymnasium", "mujoco", "numpy", "torch")},
                "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                "replay_persisted": True, "simulator_persisted": True,
                "resume_from": str(Path(resume).resolve()) if resume else None,
                "resume_note": "Full continuation state; cross-platform bitwise equality is not guaranteed"}
    counters = {"real_env_steps": 0, "policy_gradient_steps": 0, "episodes": 0}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    previous_threads = torch.get_num_threads()
    env = None
    tracker = None
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    previous_signals = {sig: signal.signal(sig, request_stop)
                        for sig in (signal.SIGTERM, signal.SIGUSR1)}
    start = time.perf_counter()
    try:
        torch.set_num_threads(config.training.torch_threads)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        env = make_env(ExperimentConfig(config.env, config.dynamics, config.seed))
        learner = restored_learner or SACLearner(env.observation_space.shape[0], env.action_space.low, env.action_space.high, config.algo, device=device)
        if any(parameter.device != device for module in (learner.actor, learner.critic, learner.target_critic)
               for parameter in module.parameters()) or learner.log_alpha.device != device:
            raise RuntimeError("Learner parameter placement does not match the requested device")
        print(f"SAC learner device verified: {device}", flush=True)
        replay = ReplayBuffer(config.training.replay_capacity, learner.obs_dim, len(learner.action_low), config.seed)
        obs, _ = env.reset(seed=config.seed)
        episode_return, episode_length = 0.0, 0
        if payload is not None:
            counters = dict(payload["counters"])
            obs, episode_return, episode_length = restore_training_state(env, replay, payload["training_state"])
        rng_before_tracking = capture_rng(learner.device)
        tracker = Tracker(config, run_dir, resume)
        restore_rng(payload["rng"] if payload is not None else rng_before_tracking, learner.device)
        counters.setdefault('vector_steps', counters['real_env_steps'])
        counters.setdefault('real_policy_samples', learner.policy_gradient_steps*config.training.batch_size)
        for key in ('synthetic_policy_samples','synthetic_transition_count','dynamics_model_refit_count','dynamics_model_train_steps'):
            counters.setdefault(key, 0)
        metadata['initial_counters'] = dict(counters)
        def persist(path, overwrite=False):
            save_checkpoint(path, learner, counters, config.to_dict(), overwrite=overwrite,
                            environment_contract=halfcheetah_contract(env),
                            training_state=capture_training_state(env, replay, obs, episode_return, episode_length))
        latest = run_dir / "checkpoints" / "latest.pt"
        persist(latest)
        fields = [*counters, "elapsed_seconds", "replay_size", "episode_return", "episode_length",
                  "actor_loss", "critic_loss", "alpha_loss", "alpha"]
        latest_episode_return = None
        with (run_dir / "metrics" / "train.csv").open("x", newline="") as stream, tqdm(
            total=config.training.real_env_steps, initial=counters["real_env_steps"], desc="SAC source", unit="env step",
            dynamic_ncols=True, mininterval=1.0, disable=not show_progress,
        ) as progress:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for _ in range(counters["real_env_steps"], config.training.real_env_steps):
                action = (env.action_space.sample() if counters["real_env_steps"] < config.training.learning_starts
                          else learner.act(obs))
                next_obs, reward, terminated, truncated, _ = collect_transition(env, replay, obs, action)
                counters["real_env_steps"] += 1
                counters["vector_steps"] += 1
                episode_return += float(reward)
                episode_length += 1
                losses = {}
                if counters["real_env_steps"] >= config.training.learning_starts and len(replay) >= config.training.batch_size:
                    for _ in range(config.training.updates_per_env_step):
                        losses = learner.update(replay.sample(config.training.batch_size))
                        counters['policy_gradient_steps'] = learner.policy_gradient_steps
                        counters['real_policy_samples'] += config.training.batch_size
                        record_update(tracker, counters, losses, real_count=config.training.batch_size,
                            synthetic_count=0, requested_real_ratio=1.0, real_replay_size=len(replay),
                            model_replay_size=0, wall_time=time.perf_counter()-start)
                counters["policy_gradient_steps"] = learner.policy_gradient_steps
                finished = terminated or truncated
                counters["episodes"] += int(finished)
                record_transition(tracker, run_dir/'metrics/train.jsonl', counters, losses,
                                  time.perf_counter()-start)
                if finished:
                    latest_episode_return = episode_return
                if (finished or counters["real_env_steps"] % config.training.log_every == 0
                        or counters["real_env_steps"] == config.training.real_env_steps):
                    writer.writerow({**counters, "elapsed_seconds": time.perf_counter() - start,
                                     "replay_size": len(replay), **losses,
                                     "episode_return": episode_return if finished else "",
                                     "episode_length": episode_length if finished else ""})
                    stream.flush()
                    tracker.log({"train/policy_gradient_steps": counters["policy_gradient_steps"],
                                 "train/episodes": counters["episodes"], "train/replay_size": len(replay),
                                 **{f"train/{k}": v for k, v in losses.items()},
                                 **({"train/episode_return": episode_return, "train/episode_length": episode_length}
                                    if finished else {})}, counters["real_env_steps"])
                    progress.set_postfix({
                        "updates": counters["policy_gradient_steps"],
                        "episodes": counters["episodes"],
                        "return": (f"{latest_episode_return:.2f}"
                                   if latest_episode_return is not None else "n/a"),
                        **{name: f"{value:.4g}" for name, value in losses.items()},
                    }, refresh=False)
                progress.update(1)
                obs = next_obs
                if finished:
                    obs, _ = env.reset()
                    episode_return, episode_length = 0.0, 0
                if counters["real_env_steps"] % config.training.checkpoint_every == 0 or stop_requested:
                    persist(latest, overwrite=True)
                if stop_requested:
                    metadata["status"] = "interrupted"
                    print(f"Stop signal received; resume from {latest}", flush=True)
                    return run_dir
                if config.tracking.video_every and counters["real_env_steps"] % config.tracking.video_every == 0:
                    tracker.videos(learner, config, run_dir, counters["real_env_steps"])
        metadata["training_elapsed_seconds"] = time.perf_counter() - start
        checkpoint = run_dir / "checkpoints" / "final.pt"
        persist(checkpoint)
        persist(latest, overwrite=True)
        if show_progress:
            tqdm.write(f"Checkpoint saved: {checkpoint}")
            tqdm.write("Running frozen source/target evaluation...")
        # Evaluate a fresh learner loaded from the artifact, never the live training object.
        summary = evaluate_checkpoint(checkpoint, run_dir / "metrics" / "frozen_shift", device=device)
        tracker.log({"eval/delta_return": summary["delta_return"],
                     **{f"eval/{condition}/{key}": value for condition in ("source", "target")
                        for key, value in summary[condition].items()}}, counters["real_env_steps"])
        if not config.tracking.video_every or counters["real_env_steps"] % config.tracking.video_every != 0:
            tracker.videos(learner, config, run_dir, counters["real_env_steps"])
        metadata["frozen_delta_return"] = summary["delta_return"]
        metadata["status"] = "complete"
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        if env is not None:
            env.close()
        torch.set_num_threads(previous_threads)
        metadata.update(counters)
        metadata.update(ended_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - start)
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if tracker is not None:
            tracker.finish(failed=metadata["status"] == "failed")
    return run_dir
