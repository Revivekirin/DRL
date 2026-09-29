"""Nominal real-environment SAC orchestration; losses remain batch-only."""
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
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
from dynamics_shift.utils.checkpoint import save_checkpoint


def _git_metadata() -> dict:
    root = Path(__file__).resolve().parents[3]
    def run(*args: str) -> str | None:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}


def train_source(config: RunConfig, output_root: str | Path = "outputs", *,
                 show_progress: bool = True) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    run_dir = Path(output_root) / config.name / f"seed_{config.seed}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics").mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    metadata = {**_git_metadata(), "algorithm": "sac", "seed": config.seed,
                "environment": config.env.id, "source_dynamics": {"actuator_scale": 1.0},
                "target_dynamics": {"actuator_scale": config.evaluation.target_actuator_scale},
                "python": platform.python_version(), "device": "cpu",
                "versions": {name: version(name) for name in ("gymnasium", "mujoco", "numpy", "torch")},
                "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                "replay_persisted": False, "exact_training_resume": False}
    counters = {"real_env_steps": 0, "policy_gradient_steps": 0, "episodes": 0}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    previous_threads = torch.get_num_threads()
    env = None
    start = time.perf_counter()
    try:
        torch.set_num_threads(config.training.torch_threads)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        env = make_env(ExperimentConfig(config.env, config.dynamics, config.seed))
        learner = SACLearner(env.observation_space.shape[0], env.action_space.low, env.action_space.high, config.algo)
        replay = ReplayBuffer(config.training.replay_capacity, learner.obs_dim, len(learner.action_low), config.seed)
        obs, _ = env.reset(seed=config.seed)
        episode_return, episode_length = 0.0, 0
        fields = [*counters, "elapsed_seconds", "replay_size", "episode_return", "episode_length",
                  "actor_loss", "critic_loss", "alpha_loss", "alpha"]
        latest_episode_return = None
        with (run_dir / "metrics" / "train.csv").open("x", newline="") as stream, tqdm(
            total=config.training.real_env_steps, desc="SAC source", unit="env step",
            dynamic_ncols=True, mininterval=1.0, disable=not show_progress,
        ) as progress:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for _ in range(config.training.real_env_steps):
                action = (env.action_space.sample() if counters["real_env_steps"] < config.training.learning_starts
                          else learner.act(obs))
                next_obs, reward, terminated, truncated, _ = env.step(action)
                # Store the actual final observation before resetting on either boundary.
                replay.add(obs, action, reward, next_obs, terminated, truncated)
                counters["real_env_steps"] += 1
                episode_return += float(reward)
                episode_length += 1
                losses = {}
                if counters["real_env_steps"] >= config.training.learning_starts and len(replay) >= config.training.batch_size:
                    for _ in range(config.training.updates_per_env_step):
                        losses = learner.update(replay.sample(config.training.batch_size))
                counters["policy_gradient_steps"] = learner.policy_gradient_steps
                finished = terminated or truncated
                counters["episodes"] += int(finished)
                if finished:
                    latest_episode_return = episode_return
                if (finished or counters["real_env_steps"] % config.training.log_every == 0
                        or counters["real_env_steps"] == config.training.real_env_steps):
                    writer.writerow({**counters, "elapsed_seconds": time.perf_counter() - start,
                                     "replay_size": len(replay), **losses,
                                     "episode_return": episode_return if finished else "",
                                     "episode_length": episode_length if finished else ""})
                    stream.flush()
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
        metadata["training_elapsed_seconds"] = time.perf_counter() - start
        checkpoint = run_dir / "checkpoints" / "final.pt"
        save_checkpoint(checkpoint, learner, counters, config.to_dict())
        if show_progress:
            tqdm.write(f"Checkpoint saved: {checkpoint}")
            tqdm.write("Running frozen source/target evaluation...")
        # Evaluate a fresh learner loaded from the artifact, never the live training object.
        summary = evaluate_checkpoint(checkpoint, run_dir / "metrics" / "frozen_shift")
        metadata["frozen_delta_return"] = summary["delta_return"]
        metadata["status"] = "complete"
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
        torch.set_num_threads(previous_threads)
        metadata.update(counters)
        metadata.update(ended_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - start)
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run_dir
