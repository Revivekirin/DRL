"""Evaluate one saved policy in paired source and target episodes."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4
import torch
import numpy as np
from dynamics_shift.evaluation.policy_eval import evaluate_shift
from dynamics_shift.experiments.config import RunConfig
from dynamics_shift.utils.checkpoint import load_checkpoint, isolated_rng
from dataclasses import replace
from copy import deepcopy


def evaluate_checkpoint(checkpoint: str | Path, output_dir: str | Path | None = None, *,
                        device: str | torch.device = "cpu", evaluation_overrides=None, episode_seeds=None, episodes=None) -> dict:
    checkpoint = Path(checkpoint)
    if output_dir is None:
        output_dir = checkpoint.parent.parent / "evaluations" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    learner, payload = load_checkpoint(checkpoint, device=device)
    raw = dict(payload['config'])
    if 'mbpo' in raw:
        from dynamics_shift.algorithms.mbpo.config import MBPORunConfig
        config = MBPORunConfig.from_dict(raw)
    else:
        config = RunConfig.from_dict(raw)
    if episodes is not None:
        if type(episodes) is not int or episodes < 1:
            raise ValueError('episodes must be positive')
        if not hasattr(config.evaluation, 'episodes'):
            raise ValueError('Paired MuJoCo SAC uses explicit saved seeds, not --episodes')
        config = replace(config, evaluation=replace(config.evaluation, episodes=episodes))
    if config.env.backend == "maniskill":
        from dynamics_shift.evaluation.maniskill import evaluate_loaded_checkpoint
        return evaluate_loaded_checkpoint(learner, payload, config, checkpoint, output_dir,
                                          evaluation_overrides=evaluation_overrides, episode_seeds=episode_seeds)
    if evaluation_overrides or episode_seeds is not None:
        raise ValueError("ManiSkill evaluation options cannot be used for HalfCheetah")
    previous_threads = torch.get_num_threads()
    before = deepcopy(learner.state_dict())
    updates = learner.policy_gradient_steps
    try:
        torch.set_num_threads(config.training.torch_threads)
        with isolated_rng(learner.device):
            if 'mbpo' in raw:
                from dataclasses import asdict
                from dynamics_shift.config import ExperimentConfig
                from dynamics_shift.envs import make_env
                from dynamics_shift.evaluation.policy_eval import evaluate_policy
                seeds = tuple(config.evaluation.seed+i for i in range(config.evaluation.episodes))
                env = make_env(ExperimentConfig(config.env, config.dynamics, config.seed))
                try:
                    from dynamics_shift.evaluation.contracts import assert_learner_contract, halfcheetah_contract
                    assert_learner_contract(learner, env)
                    contract = halfcheetah_contract(env)
                    recorded = payload.get('environment_contract')
                    if recorded is not None and any(contract.get(k) != v for k,v in recorded.items()):
                        raise ValueError('Saved MuJoCo environment contract differs')
                    rows = [asdict(row) for row in evaluate_policy(learner, env, seeds, 'source', 1.0)]
                finally:
                    env.close()
                summary = {'mean_return': sum(r['per_episode_return'] for r in rows)/len(rows),
                           'episodes': len(rows), 'protocol': 'nominal; saved MBPO evaluation seeds'}
            else:
                rows, summary = evaluate_shift(learner, config.env, config.evaluation.seeds,
                                               config.evaluation.target_actuator_scale)
    finally:
        torch.set_num_threads(previous_threads)
    from dynamics_shift.evaluation.state import assert_state_equal
    assert_state_equal(before, learner.state_dict())
    if learner.policy_gradient_steps != updates:
        raise RuntimeError('Evaluation changed learner state')
    probe = payload.get('learner_probe')
    probe_status = 'UNVERIFIED: legacy checkpoint has no learner_probe'
    if probe:
        np.testing.assert_allclose(learner.act(np.asarray(probe['observation'], dtype=np.float32), deterministic=True),
                                   probe['action'], rtol=1e-5, atol=1e-6)
        probe_status = 'passed'
    summary.update(learner_state_unchanged=True, learner_updates_during_evaluation=0,
                   learner_restore_probe=probe_status)
    with (output_dir / "frozen_shift_eval.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary.update({"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "checkpoint": str(checkpoint.resolve()), "seeds": list(seeds if 'mbpo' in raw else config.evaluation.seeds),
                    "source_actuator_scale": 1.0,
                    "target_actuator_scale": getattr(config.evaluation, "target_actuator_scale", None),
                    "device": str(learner.device), "deterministic": True, "counters": payload["counters"]})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
