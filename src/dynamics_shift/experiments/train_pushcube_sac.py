"""Single-CPU simulation with CPU/CUDA SAC; no simulator resume."""
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
from dynamics_shift.utils.device import check_device
from .evaluate_sac_shift import evaluate_checkpoint


def train_pushcube_sac(config, output_root, *, show_progress=True, resume=None):
    if resume is not None:
        raise ValueError("Exact ManiSkill training resume is unsupported: replay, simulator, controller and environment RNG are not persisted. Use the evaluation CLI for learner loading.")
    device = check_device(config.training.device)
    run = Path(output_root) / config.name / "seed_0" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    (run / "metrics").mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    counters = dict(real_env_steps=0, policy_gradient_steps=0, episodes=0)
    metadata = dict(status="running", algorithm="sac", seed=0, backend="maniskill", device=str(device),
                    sim_backend=config.env.sim_backend, num_envs=config.env.num_envs,
                    gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    cuda_build=torch.version.cuda,
                    replay_persisted=False, simulator_persisted=False, training_resume_supported=False,
                    versions={name: version(name) for name in ("mani-skill", "sapien", "torch", "numpy", "gymnasium")})
    threads = torch.get_num_threads()
    env = None
    start = time.perf_counter()
    success_terminations = 0
    evaluation_seconds = 0.0
    try:
        torch.set_num_threads(config.training.torch_threads)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        env = make_env(ExperimentConfig(config.env, None, 0))
        contract = pushcube_contract(config, env)
        metadata["environment_contract"] = contract
        learner = SACLearner(contract["observation_dim"], env.action_space.low, env.action_space.high, config.algo, device=device)
        if any(p.device != device for module in (learner.actor, learner.critic, learner.target_critic)
               for p in module.parameters()) or learner.log_alpha.device != device:
            raise RuntimeError("Learner parameter device mismatch")
        replay = ReplayBuffer(config.training.replay_capacity, learner.obs_dim, contract["action_dim"], seed=0)
        obs, _ = env.reset(seed=0)

        def persist(name, overwrite=False):
            probe = dict(observation=obs.tolist(), action=learner.act(obs, deterministic=True).tolist())
            save_checkpoint(run / "checkpoints" / name, learner, counters, config.to_dict(),
                            environment_contract=contract, learner_probe=probe, overwrite=overwrite)

        persist("initial.pt")
        persist("latest.pt")

        def evaluate_saved(name, output_name):
            nonlocal evaluation_seconds
            began = time.perf_counter()
            summary = evaluate_checkpoint(run / "checkpoints" / name, run / "metrics" / output_name, device=device)
            evaluation_seconds += time.perf_counter() - began
            record = dict(real_env_steps=counters["real_env_steps"],
                          policy_gradient_steps=counters["policy_gradient_steps"],
                          wall_time_seconds=time.perf_counter() - start, **summary)
            with (run / "metrics/evaluation_history.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(dict(event="pushcube_evaluation", real_env_steps=counters["real_env_steps"],
                                  mean_return=summary["mean_return"], success_once=summary["success_once"],
                                  success_at_end=summary["success_at_end"],
                                  learner_restore_probe=summary["learner_restore_probe"])), flush=True)
            return summary

        if config.evaluation.interval:
            evaluate_saved("initial.pt", "evaluation_step_0")
        episode_return, episode_length, success_once = 0.0, 0, False
        fields = [*counters, "replay_size", "episode_return", "episode_length", "success_once", "success_at_end",
                  "terminated", "truncated", "actor_loss", "critic_loss", "alpha_loss", "alpha",
                  "wall_time_seconds", "evaluation_seconds", "transitions_per_second",
                  "training_transitions_per_second"]
        with (run / "metrics/train.csv").open("x", newline="") as stream, tqdm(
            total=config.training.real_env_steps, desc="PushCube SAC", disable=not show_progress
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
                elapsed = time.perf_counter() - start
                writer.writerow({**counters, "replay_size": len(replay), **losses,
                                 "wall_time_seconds": elapsed, "evaluation_seconds": evaluation_seconds,
                                 "transitions_per_second": step / max(elapsed, 1e-9),
                                 "training_transitions_per_second": step / max(elapsed - evaluation_seconds, 1e-9),
                                 **(dict(episode_return=episode_return, episode_length=episode_length,
                                         success_once=success_once, success_at_end=success,
                                         terminated=bool(terminated), truncated=bool(truncated)) if finished else {})})
                stream.flush()
                obs = next_obs
                if finished:
                    obs, _ = env.reset()
                    episode_return, episode_length, success_once = 0.0, 0, False
                checkpoint_due = step % config.training.checkpoint_every == 0
                eval_due = (config.evaluation.interval > 0 and step % config.evaluation.interval == 0
                            and step < config.training.real_env_steps)
                if checkpoint_due:
                    persist("latest.pt", overwrite=True)
                if checkpoint_due or eval_due:
                    persist(f"step_{step}.pt")
                if eval_due:
                    evaluate_saved(f"step_{step}.pt", f"evaluation_step_{step}")
                progress.update(1)
            metadata["partial_episode"] = dict(length=episode_length, episode_return=episode_return, success_once=success_once)
        persist("final.pt")
        persist("latest.pt", overwrite=True)
        # The evaluation reloads the saved learner and verifies its deterministic action probe.
        summary = evaluate_saved("final.pt", "evaluation")
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
        elapsed = time.perf_counter() - start
        metadata.update(counters, elapsed_seconds=elapsed, wall_time_seconds=elapsed,
                        evaluation_seconds=evaluation_seconds,
                        transitions_per_second=counters["real_env_steps"] / max(elapsed, 1e-9),
                        training_transitions_per_second=counters["real_env_steps"] / max(elapsed - evaluation_seconds, 1e-9))
        (run / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run
