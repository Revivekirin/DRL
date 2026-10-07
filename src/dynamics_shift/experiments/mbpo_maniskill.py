"""Bounded nominal GPU PushCube MBPO; fresh initialization, no exact resume."""
from .dispatch import save_resolved_config
from .records import record_update, record_transition
from .artifacts import save_learner_artifact
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
from dynamics_shift.algorithms.sac.learner import SACLearner
from dynamics_shift.algorithms.mbpo.rollouts import mixed_batch
from dynamics_shift.algorithms.mbpo.pushcube import (
    observation_layout, StateDiagnostics, model_errors, generate_pushcube_rollouts,
)
from dynamics_shift.config import ExperimentConfig
from dynamics_shift.data.model_data import RealReplayBuffer, ModelReplayBuffer, ModelDataset
from dynamics_shift.envs import make_env
from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble
from dynamics_shift.utils.checkpoint import save_checkpoint, isolated_rng
from dynamics_shift.utils.tracking import Tracker
from dynamics_shift.utils.device import check_device
from dynamics_shift.experiments.provenance import _git_metadata
from dynamics_shift.experiments.interaction import (
    _single_action_space, _training_contract, _random_action, _policy_action,
    _true_next_observation, _add_replay_batch, _extract_vector_success,
    _to_numpy, _evaluate_saved_checkpoint, collect_transition,
)


def train_mbpo(config, settings, output_root='outputs'):
    device = check_device(config.training.device)
    run = Path(output_root) / config.name / 'seed_0' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    (run / 'metrics').mkdir()
    (run / 'checkpoints').mkdir()
    from dataclasses import asdict
    save_resolved_config(run, config, 'mbpo', settings)
    counters = dict(real_env_steps=0, vector_steps=0, policy_gradient_steps=0, episodes=0,
                    dynamics_model_refit_count=0, dynamics_model_train_steps=0,
                    synthetic_transition_count=0, real_policy_samples=0, synthetic_policy_samples=0,
                    real_truncations=0, final_observation_checks=0)
    metadata = dict(algorithm='mbpo', initialization='from_scratch', training_seed=0,
                    training_resume_supported=False, model_training_data='real_replay_only',
                    synthetic_termination_policy='terminated=False; truncated=False; horizon is computation cutoff',
                    model_replay_retention='clear at each refit; generate fresh horizon-1 transitions',
                    success_termination='DISABLED in training; inspect success_once and success_at_end',
                    versions={name: version(name) for name in ('mani-skill', 'torch', 'numpy')},
                    status='running', **_git_metadata())
    env = None
    tracker = None
    threads = torch.get_num_threads()
    start = time.perf_counter()
    try:
        torch.set_num_threads(config.training.torch_threads)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        env = make_env(ExperimentConfig(config.env, None, 0))
        if not env.ignore_terminations:
            raise ValueError('MBPO requires actual wrapper ignore_terminations=True')
        obs, _ = env.reset(seed=0)
        n = config.env.num_envs
        contract = _training_contract(config, env, num_envs=n)
        if obs.shape != (n, contract['observation_dim']):
            raise ValueError('Unexpected vector observation shape')
        layout = observation_layout(env.unwrapped.get_obs(unflattened=True), obs)
        metadata.update(training_environment_contract=contract, observation_layout=layout,
            dynamics_geometry='aligned_quaternion_delta_v1', fit_start_is_candidate=True)
        eval_contract = {**contract, 'sim_backend': 'cpu', 'num_envs': 1,
                         'termination_policy': 'terminate_on_success', 'automatic_reset': False}
        metadata['evaluation_environment_contract'] = eval_contract
        with isolated_rng(device):
            tracker = Tracker(config, run)
            tracker.metadata({**metadata, "mbpo_config": asdict(settings)})
        action_space = _single_action_space(env)
        od, ad = contract['observation_dim'], contract['action_dim']
        real = RealReplayBuffer(config.training.replay_capacity, od, ad, 0)
        synthetic = ModelReplayBuffer(settings.model_replay_capacity, od, ad, 1)
        learner = SACLearner(od, action_space.low, action_space.high, config.algo, device)
        learner.record_update_diagnostics = True
        model = ProbabilisticEnsemble(od, ad, settings, device, 0, observation_layout=layout, preserve_start=True)
        returns, lengths, successes = np.zeros(n), np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
        budget = 0.0
        next_refit = config.training.learning_starts
        next_checkpoint = config.training.checkpoint_every
        next_eval = config.evaluation.interval if config.evaluation.interval else None

        def persist(name):
            save_learner_artifact(run/'checkpoints'/f'{name}.pt', learner, counters,
                                  config, contract, obs)
            with (run / 'checkpoints' / f'{name}_model.pt').open('xb') as stream:
                torch.save({'model': model.state_dict(), 'training_resume_supported': False,
                            'replay_persisted': False, 'simulator_persisted': False,
                            'initialization': 'from_scratch', 'counters': dict(counters)}, stream)

        persist('initial')
        with (run / 'metrics/model_refits.jsonl').open('x') as model_file, \
             (run / 'metrics/episodes.jsonl').open('x') as episode_file:
            while counters['real_env_steps'] < config.training.real_env_steps:
                action = (_random_action(env, obs, num_envs=n, device=obs.device)
                          if counters['real_env_steps'] < config.training.learning_starts
                          else _policy_action(learner, obs))
                next_obs, reward, terminated, truncated, info = collect_transition(
                    env, real, obs, action, num_envs=n, automatic_reset=True)
                term = _to_numpy(terminated).astype(bool).reshape(n)
                trunc = _to_numpy(truncated).astype(bool).reshape(n)
                if term.any():
                    raise ValueError('Training termination contradicts ignore_terminations contract')
                done = term | trunc
                counters['final_observation_checks'] += int(done.sum())
                counters['real_truncations'] += int(trunc.sum())
                counters['real_env_steps'] += n
                counters['vector_steps'] += 1
                current = counters['real_env_steps']
                success = _extract_vector_success(info, terminated, n)
                returns += _to_numpy(reward).reshape(n)
                lengths += 1
                successes |= success
                for index in np.flatnonzero(done):
                    with isolated_rng(device):
                        tracker.scalars("train", dict(episode_return=float(returns[index]),
                            episode_length=int(lengths[index]), success_once=int(successes[index]),
                            success_at_end=int(success[index]), env_index=int(index)), current)
                    episode_file.write(json.dumps(dict(real_env_steps=current, env_index=int(index),
                        episode_return=float(returns[index]), episode_length=int(lengths[index]),
                        success_once=int(successes[index]), success_at_end=int(success[index]),
                        terminated=bool(term[index]), truncated=bool(trunc[index]), successful_termination=False)) + '\n')
                counters['episodes'] += int(done.sum())
                returns[done], lengths[done], successes[done] = 0, 0, False
                episode_file.flush()
                obs = next_obs  # Wrapper already reset only the finished environments.

                if current >= next_refit and len(real) >= 3:
                    dataset = ModelDataset.from_real_replay(real, rng, settings.holdout_ratio, settings.model_max_samples)
                    fitted = model.train(dataset)
                    errors = model_errors(model, dataset, layout)
                    diagnostics = StateDiagnostics(layout)
                    generated = generate_pushcube_rollouts(learner, model, real, synthetic, settings, rng, diagnostics, contract)
                    if synthetic._arrays['terminated'][:len(synthetic)].any() or synthetic._arrays['truncated'][:len(synthetic)].any():
                        raise ValueError('Synthetic horizon incorrectly stored as episode boundary')
                    ids = dataset.holdout_indices
                    real_diagnostics = StateDiagnostics(layout)
                    original = dataset.inputs[ids, :od]
                    real_diagnostics.observe(original, original + dataset.targets[ids, :od], dataset.targets[ids, -1])
                    counters['dynamics_model_refit_count'] = model.refit_count
                    counters['dynamics_model_train_steps'] = model.train_steps
                    counters['synthetic_transition_count'] += generated
                    model_file.write(json.dumps(dict(real_env_steps=current, refit=model.refit_count,
                        model_train_steps=model.train_steps, generated=generated, dataset_source='real_only',
                        **fitted, **errors, synthetic_state_diagnostics=diagnostics.records,
                        real_state_diagnostics=real_diagnostics.records)) + '\n')
                    model_file.flush()
                    with isolated_rng(device):
                        tracker.scalars("model", {**fitted, **errors, "refit": model.refit_count,
                            "member_optimizer_steps": model.train_steps,
                            "generated": generated, "synthetic_state_diagnostics_raw": diagnostics.records,
                            "real_state_diagnostics": real_diagnostics.records}, current)
                    print(json.dumps(dict(event='pushcube_mbpo_refit', real_env_steps=current,
                        refit=model.refit_count, generated=generated, **errors)), flush=True)
                    while next_refit <= current:
                        next_refit += settings.model_train_frequency
                losses = {}
                nr = ns = 0
                updates = 0
                if current >= config.training.learning_starts and len(real) >= config.training.batch_size:
                    budget += n * config.training.utd
                    while budget >= 1:
                        if model.refit_count < 1 or not len(synthetic):
                            raise RuntimeError('No synthetic sampling allowed before model fit')
                        batch, nr, ns = mixed_batch(real, synthetic, config.training.batch_size, settings.real_ratio, rng)
                        losses = learner.update(batch)
                        if not all(np.isfinite(v) for v in losses.values()):
                            raise FloatingPointError('Nonfinite learner diagnostics')
                        counters['real_policy_samples'] += nr
                        counters['synthetic_policy_samples'] += ns
                        updates += 1
                        budget -= 1
                        counters['policy_gradient_steps'] = learner.policy_gradient_steps
                        record_update(tracker, counters, losses, real_count=nr, synthetic_count=ns,
                            requested_real_ratio=settings.real_ratio, real_replay_size=len(real),
                            model_replay_size=len(synthetic), wall_time=time.perf_counter()-start)
                counters['policy_gradient_steps'] = learner.policy_gradient_steps
                elapsed = time.perf_counter() - start
                record_transition(tracker, run/'metrics/train.jsonl', counters, losses, elapsed,
                    update_budget=budget, updates_this_vector_step=updates,
                    batch_real_count=nr, batch_synthetic_count=ns,
                    actual_batch_synthetic_ratio=ns/(nr+ns) if nr+ns else None,
                    requested_real_ratio=settings.real_ratio,
                    loss_aggregation='last update in this vector call',
                    real_replay_size=len(real), model_replay_size=len(synthetic))
                checkpoint_due = current >= next_checkpoint
                eval_due = next_eval is not None and current >= next_eval and current < config.training.real_env_steps
                if checkpoint_due or eval_due:
                    persist(f'step_{current}')
                    with isolated_rng(device):
                        tracker.checkpoint_video(run / f'checkpoints/step_{current}.pt', config, current)
                    if eval_due:
                        evaluation = _evaluate_saved_checkpoint(run / f'checkpoints/step_{current}.pt',
                            run / f'metrics/evaluation_step_{current}', config, device=device)
                        tracker.scalars('eval', evaluation, current)
                        while next_eval <= current:
                            next_eval += config.evaluation.interval
                    while next_checkpoint <= current:
                        next_checkpoint += config.training.checkpoint_every
        if not (model.refit_count and counters['policy_gradient_steps'] and counters['final_observation_checks']):
            raise RuntimeError('Smoke did not exercise fitting, updates and real episode boundaries')
        persist('final')
        metadata['evaluation'] = _evaluate_saved_checkpoint(run / 'checkpoints/final.pt',
            run / 'metrics/evaluation', config, device=device)
        with isolated_rng(device):
            tracker.scalars('eval', metadata['evaluation'], counters['real_env_steps'])
            tracker.checkpoint_video(run / 'checkpoints/final.pt', config, counters['real_env_steps'])
        metadata['status'] = 'complete'
        print(json.dumps(dict(event='pushcube_mbpo_complete', run_dir=str(run.resolve()), **counters)))
        return run
    except Exception as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            torch.set_num_threads(threads)
            elapsed = time.perf_counter() - start
            metadata.update(counters, wall_time_seconds=elapsed,
                            transitions_per_second=counters['real_env_steps'] / elapsed)
            if tracker is not None:
                tracker.metadata(metadata)
                tracker.finish(failed=metadata['status'] == 'failed')
            (run / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

# Historical Python API alias.
train_pushcube_mbpo = train_mbpo
