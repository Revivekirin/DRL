"""Real-only fixed-dataset fitting diagnostic; no SAC learner or synthetic replay."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from uuid import uuid4
import numpy as np
import torch
import yaml
from dynamics_shift.algorithms.mbpo.config import MBPOConfig
from dynamics_shift.algorithms.mbpo.pushcube import observation_layout, model_errors, StateDiagnostics, episode_split
from dynamics_shift.config import ExperimentConfig
from dynamics_shift.data.model_data import RealReplayBuffer, ModelDataset
from dynamics_shift.envs import make_env
from dynamics_shift.experiments.config import RunConfig
from dynamics_shift.experiments.train_pushcube_sac import (
    _random_action, _true_next_observation, _add_replay_batch, _to_numpy, _training_contract,
)
from dynamics_shift.experiments.train_sac_source import _git_metadata
from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble
from dynamics_shift.utils.checkpoint import isolated_rng
from dynamics_shift.utils.device import check_device
from dynamics_shift.utils.tracking import Tracker




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root', default='outputs/pushcube_dynamics_fit')
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    fitting, model_config = raw.pop('fitting'), MBPOConfig(**raw.pop('mbpo'))
    config = RunConfig.from_dict(raw)
    if (config.seed != 0 or config.env.backend != 'maniskill' or config.env.sim_backend != 'gpu'
        or config.env.num_envs < 2 or fitting['collection_policy'] != 'uniform_random'):
        raise ValueError('Requires seed-0 GPU PushCube random collection')
    if config.training.replay_capacity < config.training.real_env_steps:
        raise ValueError('Fixed dataset must fit in real replay without eviction')
    if (type(fitting['rounds']) is not int or not 1 <= fitting['rounds'] <= 5
        or type(fitting['holdout_episode_modulus']) is not int or fitting['holdout_episode_modulus'] < 2):
        raise ValueError('Invalid fitting rounds or holdout episode modulus')
    device = check_device(config.training.device)
    run = Path(args.output_root) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    (run/'config.yaml').write_text(yaml.safe_dump({**config.to_dict(), 'mbpo': asdict(model_config), 'fitting': fitting}))
    metadata = dict(status='running', experiment='dynamics_only', policy_gradient_steps=0,
        synthetic_policy_samples=0, training_resume_supported=False, collection_policy='uniform_random',
        checkpoint_initialization=None, seed=0, **_git_metadata())
    env = tracker = None
    start = time.perf_counter()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.training.torch_threads)
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        env = make_env(ExperimentConfig(config.env, None, 0))
        obs, _ = env.reset(seed=0)
        n = config.env.num_envs
        contract = _training_contract(config, env, num_envs=n)
        if not env.ignore_terminations:
            raise ValueError('Requires actual ignore_terminations=True')
        total = config.training.real_env_steps
        horizon = contract['horizon']
        if total % (n * horizon):
            raise ValueError('Collect complete episodes only')
        layout = observation_layout(env.unwrapped.get_obs(unflattened=True), obs)
        metadata.update(training_environment_contract=contract, observation_layout=layout)
        with isolated_rng(device):
            tracker = Tracker(config, run)
            tracker.metadata({**metadata, 'mbpo_config': asdict(model_config), 'fitting': fitting})
        real = RealReplayBuffer(total, contract['observation_dim'], contract['action_dim'], 0)
        episode_ids = []
        for step in range(total // n):
            action = _random_action(env, obs, num_envs=n, device=obs.device)
            no, reward, term, trunc, info = env.step(action)
            if bool(term.any()):
                raise ValueError('Unexpected physical termination')
            boundary = (step + 1) % horizon == 0
            if not np.all(_to_numpy(trunc) == boundary):
                raise ValueError('Unexpected episode boundary; fixed split invalid')
            true_next = _true_next_observation(no, term, trunc, info)
            _add_replay_batch(real, obs, action, reward, true_next, term, trunc, num_envs=n)
            episode_ids.extend((step // horizon) * n + i for i in range(n))
            obs = no
        arrays = {key: value[:total] for key, value in real._arrays.items()}
        ti, vi = episode_split(episode_ids, fitting['holdout_episode_modulus'])
        inputs = np.concatenate([arrays['obs'], arrays['action']], axis=1)
        targets = np.concatenate([arrays['next_obs']-arrays['obs'], arrays['reward']], axis=1)
        dataset = ModelDataset(inputs, targets, ti, vi)
        np.savez_compressed(run/'real_dataset.npz', **arrays, episode_ids=episode_ids, train_indices=ti, holdout_indices=vi)
        metadata.update(real_env_steps=total, vector_steps=total//n, train_samples=len(ti), holdout_samples=len(vi),
            train_episode_ids=sorted(set(np.asarray(episode_ids)[ti].tolist())),
            holdout_episode_ids=sorted(set(np.asarray(episode_ids)[vi].tolist())))
        model = ProbabilisticEnsemble(contract['observation_dim'], contract['action_dim'], model_config, device, 0)
        # Identical standard-normal draws at every round for diagnostic comparability.
        noise = np.random.default_rng(123).standard_normal((len(vi), model.obs_dim+1))
        with (run/'fitting.jsonl').open('x') as stream:
            for round_id in range(1, fitting['rounds']+1):
                metrics = model.train(dataset)
                errors = model_errors(model, dataset, layout)
                means, variances = model.predict(inputs[vi])
                diagnostics = []
                for member in range(model_config.ensemble_size):
                    predicted = means[member]+np.sqrt(variances[member])*noise
                    diagnostic = StateDiagnostics(layout)
                    diagnostic.observe(inputs[vi, :model.obs_dim], inputs[vi, :model.obs_dim]+predicted[:, :-1],
                                       predicted[:, -1], means, model.elites)
                    diagnostics.append({'member': member, **diagnostic.records[0]})
                row = dict(real_env_steps=total, fitting_round=round_id, member_optimizer_steps=model.train_steps,
                    wall_time_seconds=time.perf_counter()-start, **metrics, **errors,
                    sampled_holdout_validity=diagnostics)
                stream.write(json.dumps(row)+'\n'); stream.flush()
                with (run/f'model_round_{round_id}.pt').open('xb') as file:
                    torch.save({'model': model.state_dict(), 'training_resume_supported': False}, file)
                with isolated_rng(device):
                    tracker.scalars('model', row, total)
                print(json.dumps({'event':'dynamics_fit_round', 'round':round_id,
                                  'holdout_state_rmse':errors['state_prediction_rmse'],
                                  'holdout_reward_rmse':errors['reward_prediction_rmse']}), flush=True)
        metadata.update(status='complete', model_train_steps=model.train_steps)
    except Exception as error:
        metadata.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        try:
            if env is not None: env.close()
        finally:
            torch.set_num_threads(threads)
            metadata['wall_time_seconds'] = time.perf_counter()-start
            (run/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
            if tracker is not None:
                tracker.metadata(metadata)
                tracker.finish(failed=metadata['status']!='complete')
    print(run)


if __name__ == '__main__':
    main()
