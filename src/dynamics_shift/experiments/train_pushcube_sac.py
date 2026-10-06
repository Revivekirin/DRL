"""Stage-3 single-CPU SAC wiring; learner restoration only, no simulator resume."""
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
from dynamics_shift.evaluation.contracts import pushcube_contract, success_flag
from dynamics_shift.utils.checkpoint import save_checkpoint
from .evaluate_sac_shift import evaluate_checkpoint


def train_pushcube_sac(config, output_root, *, show_progress=True, resume=None):
    if resume is not None:
        raise ValueError("Exact ManiSkill training resume is unsupported: replay, simulator, controller and environment RNG are not persisted. Use the evaluation CLI for learner loading.")
    if config.training.device != "cpu":
        raise ValueError("Stage-3 PushCube uses a CPU learner")
    run = Path(output_root) / config.name / "seed_0" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    (run / "metrics").mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    counters = dict(real_env_steps=0, policy_gradient_steps=0, episodes=0)
    metadata = dict(status="running", algorithm="sac", seed=0, backend="maniskill", device="cpu",
                    replay_persisted=False, simulator_persisted=False, training_resume_supported=False,
                    versions={name: version(name) for name in ("mani-skill", "sapien", "torch", "numpy", "gymnasium")})
    threads = torch.get_num_threads()
    env = None
    start = time.perf_counter()
    success_terminations = 0
    try:
        torch.set_num_threads(config.training.torch_threads)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        env = make_env(ExperimentConfig(config.env, None, 0))
        contract = pushcube_contract(config, env)
        metadata["environment_contract"] = contract
        learner = SACLearner(contract["observation_dim"], env.action_space.low, env.action_space.high, config.algo, device="cpu")
        replay = ReplayBuffer(config.training.replay_capacity, learner.obs_dim, contract["action_dim"], seed=0)
        obs, _ = env.reset(seed=0)

        def persist(name, overwrite=False):
            probe = dict(observation=obs.tolist(), action=learner.act(obs, deterministic=True).tolist())
            save_checkpoint(run / "checkpoints" / name, learner, counters, config.to_dict(),
                            environment_contract=contract, learner_probe=probe, overwrite=overwrite)

        persist("initial.pt")
        persist("latest.pt")
        episode_return, episode_length, success_once = 0.0, 0, False
        fields = [*counters, "replay_size", "episode_return", "episode_length", "success_once", "success_at_end",
                  "terminated", "truncated", "actor_loss", "critic_loss", "alpha_loss", "alpha"]
        with (run / "metrics/train.csv").open("x", newline="") as stream, tqdm(
            total=config.training.real_env_steps, desc="PushCube SAC smoke", disable=not show_progress
        ) as progress:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(1, config.training.real_env_steps + 1):
                action = env.action_space.sample() if step <= config.training.learning_starts else learner.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                success = success_flag(info, terminated)
                # CPUGymWrapper never auto-resets. Preserve the true final observation.
                replay.add(obs, action, reward, next_obs, terminated, truncated)
                counters["real_env_steps"] = step
                episode_length += 1
                episode_return += float(reward)
                success_once |= success
                if bool(truncated) != (episode_length == contract["horizon"]):
                    raise ValueError("PushCube training TimeLimit contract violated")
                losses = {}
                if step >= config.training.learning_starts and len(replay) >= config.training.batch_size:
                    for _ in range(config.training.updates_per_env_step):
                        losses = learner.update(replay.sample(config.training.batch_size))
                        if not all(np.isfinite(v) for v in losses.values()):
                            raise FloatingPointError(f"Nonfinite SAC losses at step {step}: {losses}")
                counters["policy_gradient_steps"] = learner.policy_gradient_steps
                finished = terminated or truncated
                if finished:
                    counters["episodes"] += 1
                    success_terminations += int(terminated and success)
                # Smoke diagnostics record every update, plus complete episode stats at boundaries.
                writer.writerow({**counters, "replay_size": len(replay), **losses,
                                 **(dict(episode_return=episode_return, episode_length=episode_length,
                                         success_once=success_once, success_at_end=success,
                                         terminated=bool(terminated), truncated=bool(truncated)) if finished else {})})
                stream.flush()
                obs = next_obs
                if finished:
                    obs, _ = env.reset()
                    episode_return, episode_length, success_once = 0.0, 0, False
                if step % config.training.checkpoint_every == 0:
                    persist("latest.pt", overwrite=True)
                progress.update(1)
            metadata["partial_episode"] = dict(length=episode_length, episode_return=episode_return, success_once=success_once)
        persist("final.pt")
        persist("latest.pt", overwrite=True)
        # The evaluation reloads the saved learner and verifies its deterministic action probe.
        summary = evaluate_checkpoint(run / "checkpoints/final.pt", run / "metrics/evaluation", device="cpu")
        metadata.update(status="complete", evaluation=summary,
                        success_termination_episodes=success_terminations,
                        success_termination="verified" if success_terminations else "UNVERIFIED: no successful training episode observed")
        print(json.dumps(dict(event="pushcube_sac_complete", run_dir=str(run.resolve()), **counters,
                              learner_restore_probe=summary["learner_restore_probe"], success_termination=metadata["success_termination"])), flush=True)
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
        torch.set_num_threads(threads)
        metadata.update(counters, elapsed_seconds=time.perf_counter() - start)
        (run / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run
