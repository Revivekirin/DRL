"""Nominal deterministic PushCube evaluation with explicit termination coverage."""
import csv
import hashlib
import json
import numpy as np
import torch
from dynamics_shift.config import ExperimentConfig
from dynamics_shift.envs import make_env
from dynamics_shift.utils.checkpoint import capture_rng, restore_rng
from .contracts import assert_learner_contract, pushcube_contract, success_flag


def collect_episodes(learner, env, *, seed, episodes, horizon):
    rows = []
    was_training = learner.actor.training
    learner.actor.eval()
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed) if episode == 0 else env.reset()
            total, success_once = 0.0, False
            for length in range(1, horizon + 1):
                obs, reward, terminated, truncated, info = env.step(learner.act(obs, deterministic=True))
                success = success_flag(info, terminated)
                success_once |= success
                total += float(reward)
                if not np.isfinite(total) or not np.isfinite(obs).all():
                    raise FloatingPointError("Nonfinite PushCube evaluation data")
                if bool(truncated) != (length == horizon):
                    raise ValueError("PushCube evaluation TimeLimit differs from saved contract")
                if terminated or truncated:
                    rows.append(dict(episode=episode + 1, seed=seed, per_episode_return=total,
                                     episode_length=length, success_once=success_once, success_at_end=success,
                                     terminated=bool(terminated), truncated=bool(truncated)))
                    break
            else:
                raise RuntimeError("PushCube evaluation did not end at its horizon")
    finally:
        learner.actor.train(was_training)
    return rows


def evaluate_loaded_checkpoint(learner, payload, config, checkpoint, output_dir):
    state = capture_rng(learner.device)
    threads = torch.get_num_threads()
    env = None
    try:
        torch.set_num_threads(config.training.torch_threads)
        env = make_env(ExperimentConfig(config.env, None, config.seed))
        contract = pushcube_contract(config, env)
        if payload.get("environment_contract") != contract:
            raise ValueError("Saved PushCube environment contract differs from evaluation environment")
        assert_learner_contract(learner, env)
        probe = payload.get("learner_probe")
        if probe is None:
            raise ValueError("PushCube checkpoint lacks learner restoration probe")
        predicted = learner.act(np.asarray(probe["observation"], dtype=np.float32), deterministic=True)
        np.testing.assert_allclose(predicted, probe["action"], rtol=1e-6, atol=1e-6)
        before_updates = learner.policy_gradient_steps
        rows = collect_episodes(learner, env, seed=config.evaluation.seed,
                                episodes=config.evaluation.episodes, horizon=contract["horizon"])
        if learner.policy_gradient_steps != before_updates:
            raise RuntimeError("Evaluation changed learner update counter")
        success_terminations = sum(r["terminated"] and r["success_at_end"] for r in rows)
        returns = [r["per_episode_return"] for r in rows]
        summary = dict(environment_contract=contract, episodes=len(rows),
                       mean_return=float(np.mean(returns)), mean_episode_length=float(np.mean([r["episode_length"] for r in rows])),
                       success_once=float(np.mean([r["success_once"] for r in rows])),
                       success_at_end=float(np.mean([r["success_at_end"] for r in rows])),
                       success_termination_episodes=success_terminations,
                       success_termination="verified" if success_terminations else "UNVERIFIED: no successful episode observed",
                       learner_restore_probe="passed", learner_updates_during_evaluation=0,
                       training_resume_supported=False, deterministic=True, seed=config.evaluation.seed,
                       checkpoint=str(checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       counters=payload["counters"], device=str(learner.device))
        with (output_dir / "episodes.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        if env is not None:
            env.close()
        torch.set_num_threads(threads)
        restore_rng(state, learner.device)
