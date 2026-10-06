"""Nominal deterministic PushCube evaluation with explicit termination coverage."""
from copy import deepcopy
from dataclasses import replace
import csv
import hashlib
import json
import numpy as np
import torch
from dynamics_shift.config import ExperimentConfig
from dynamics_shift.envs import make_env
from dynamics_shift.utils.checkpoint import isolated_rng
from .contracts import assert_learner_contract, pushcube_contract, success_flag


def collect_episodes(learner, env, *, seed, episodes, horizon, episode_seeds=None):
    rows = []
    was_training = learner.actor.training
    learner.actor.eval()
    try:
        for episode in range(episodes):
            episode_seed = episode_seeds[episode] if episode_seeds is not None else seed
            obs, _ = env.reset(seed=episode_seed) if episode_seeds is not None or episode == 0 else env.reset()
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
                    rows.append(dict(episode=episode + 1, seed=episode_seed, per_episode_return=total,
                                     episode_length=length, success_once=success_once, success_at_end=success,
                                     terminated=bool(terminated), truncated=bool(truncated),
                                     successful_termination=bool(terminated and success)))
                    break
            else:
                raise RuntimeError("PushCube evaluation did not end at its horizon")
    finally:
        learner.actor.train(was_training)
    return rows


def assert_state_equal(before, after):
    """Include optimizer, temperature, targets and counters, not just actor weights."""
    if isinstance(before, torch.Tensor):
        equal = torch.equal(before, after)
    elif isinstance(before, np.ndarray):
        equal = np.array_equal(before, after)
    elif isinstance(before, dict):
        equal = before.keys() == after.keys()
        if equal:
            for key in before:
                assert_state_equal(before[key], after[key])
    elif isinstance(before, (tuple, list)):
        equal = len(before) == len(after)
        if equal:
            for a, b in zip(before, after):
                assert_state_equal(a, b)
    else:
        equal = before == after
    if not equal:
        raise RuntimeError("Evaluation changed learner state")


def evaluate_loaded_checkpoint(learner, payload, config, checkpoint, output_dir,
                               *, evaluation_overrides=None, episode_seeds=None):
    overrides = evaluation_overrides or {}
    if set(overrides) - {"sim_backend", "num_envs"}:
        raise ValueError("Only sim_backend and num_envs are evaluation overrides")
    eval_env_config = replace(config.env, **overrides)
    if eval_env_config.sim_backend != "cpu" or eval_env_config.num_envs != 1:
        raise ValueError("Evaluation requires explicit --eval-sim-backend cpu --eval-num-envs 1 for vector checkpoints")
    if episode_seeds is not None and (not episode_seeds or any(type(x) is not int or x < 0 for x in episode_seeds)):
        raise ValueError("episode_seeds must contain nonnegative integers")
    threads = torch.get_num_threads()
    env = None
    with isolated_rng(learner.device):
        try:
            torch.set_num_threads(config.training.torch_threads)
            env = make_env(ExperimentConfig(eval_env_config, None, config.seed))
            contract = pushcube_contract(replace(config, env=eval_env_config), env)
            saved = payload.get("training_environment_contract", payload.get("environment_contract"))
            unverified = []
            if "training_environment_contract" not in payload:
                unverified.append("Legacy training termination metadata is not verified; recorded policy may be incorrect")
            # Operational changes are explicit; all other recorded semantics must match.
            operational = {"sim_backend", "num_envs", "automatic_reset", "termination_policy"}
            if saved is None:
                unverified.append("Checkpoint has no environment contract; config and learner spaces checked only")
            else:
                for key, value in contract.items():
                    if key in operational:
                        continue
                    if key not in saved:
                        unverified.append("Missing saved contract field: " + key)
                    elif saved[key] != value:
                        raise ValueError("Saved PushCube contract differs: " + key)
            assert_learner_contract(learner, env)
            before = deepcopy(learner.state_dict())
            probe = payload.get("learner_probe")
            probe_status = "UNVERIFIED: checkpoint has no learner_probe"
            if probe is not None:
                predicted = learner.act(np.asarray(probe["observation"], dtype=np.float32), deterministic=True)
                np.testing.assert_allclose(predicted, probe["action"], rtol=1e-6, atol=1e-6)
                probe_status = "passed"
            else:
                unverified.append(probe_status)
            before_updates = learner.policy_gradient_steps
            rows = collect_episodes(learner, env, seed=config.evaluation.seed,
                episodes=len(episode_seeds) if episode_seeds is not None else config.evaluation.episodes,
                horizon=contract["horizon"], episode_seeds=episode_seeds)
            assert_state_equal(before, learner.state_dict())
            if learner.policy_gradient_steps != before_updates:
                raise RuntimeError("Evaluation changed learner update counter")
            success_terminations = sum(r["terminated"] and r["success_at_end"] for r in rows)
            summary = dict(environment_contract=contract, evaluation_environment_contract=contract,
                recorded_training_environment_contract=saved, evaluation_overrides=overrides,
                operational_differences={k: {"recorded_training": saved.get(k) if saved else None,
                    "evaluation": contract[k]} for k in sorted(operational)},
                unverified=unverified, episodes=len(rows),
                mean_return=float(np.mean([r["per_episode_return"] for r in rows])),
                mean_episode_length=float(np.mean([r["episode_length"] for r in rows])),
                success_once=float(np.mean([r["success_once"] for r in rows])),
                success_at_end=float(np.mean([r["success_at_end"] for r in rows])),
                success_termination_episodes=success_terminations,
                success_termination="verified" if success_terminations else "UNVERIFIED: no successful termination observed",
                learner_restore_probe=probe_status, learner_state_unchanged=True,
                learner_updates_during_evaluation=0, training_resume_supported=False,
                deterministic=True, seed=config.evaluation.seed, episode_seeds=episode_seeds,
                seed_protocol="explicit reset per episode" if episode_seeds is not None else "initial seed then continuing reset stream",
                checkpoint=str(checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                counters=payload["counters"], device=str(learner.device))
            with (output_dir / "episodes.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with (output_dir / "summary.json").open("x") as stream:
                json.dump(summary, stream, indent=2)
            return summary
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                torch.set_num_threads(threads)
