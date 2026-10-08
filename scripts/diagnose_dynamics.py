"""Bounded real collection and paired dynamics-only diagnostic. Server execution only."""
import argparse
from dataclasses import replace, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from uuid import uuid4
import numpy as np
import torch
from dynamics_shift.experiments.dispatch import load_experiment
from dynamics_shift.experiments.provenance import write_run_manifest
from dynamics_shift.utils.checkpoint import isolated_rng
from dynamics_shift.data.model_data import ModelDataset
from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble
from dynamics_shift.algorithms.mbpo.maniskill import model_errors, StateDiagnostics, episode_split


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def coverage(arrays, ids, index):
    a, b = arrays['obs'][ids, index], arrays['next_obs'][ids, index]
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('Nonbinary real grasp labels')
    return {f'{x}->{y}': int(np.sum((a == x) & (b == y))) for x in (0, 1) for y in (0, 1)}


def collect(args, experiment, out):
    from dynamics_shift.envs import make_env
    from dynamics_shift.config import ExperimentConfig
    from dynamics_shift.experiments.interaction import (
        _training_contract, _random_action, _true_next_observation, _to_numpy)
    from dynamics_shift.utils.checkpoint import load_checkpoint
    from importlib.metadata import version
    if version('mani-skill') != '3.0.1':
        raise ValueError('Contact diagnostics audited for ManiSkill 3.0.1 only')
    config = experiment.run
    n = config.env.num_envs
    if config.env.backend != 'maniskill' or config.env.sim_backend != 'gpu' or n < 2:
        raise ValueError('Collection requires audited ManiSkill GPU vector config')
    if not 0 < args.transitions <= 20000:
        raise ValueError('Diagnostic collection bound is 1..20000 real transitions')
    learner = None
    if args.policy_checkpoint:
        learner, payload = load_checkpoint(args.policy_checkpoint, device=config.training.device)
        learner.actor.eval()
        write(out/'collection_policy.json', dict(path=str(Path(args.policy_checkpoint).resolve()),
            sha256=hashlib.sha256(Path(args.policy_checkpoint).read_bytes()).hexdigest(),
            role='frozen deterministic collection only, never learner update'))
    from copy import deepcopy
    from dynamics_shift.evaluation.state import assert_state_equal
    before = deepcopy(learner.state_dict()) if learner is not None else None
    env = None
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    try:
        env = make_env(ExperimentConfig(config.env, None, 0))
        obs, _ = env.reset(seed=0)
        contract = _training_contract(config, env, num_envs=n)
        codec = contract['observation_codec']
        if codec['boolean_fields'] != ['extra.is_grasped']:
            raise ValueError('This controlled binary diagnostic requires declared is_grasped')
        if learner is not None:
            saved = payload.get('environment_contract') or payload.get('metadata', {}).get('environment_contract')
            if saved is None:
                raise ValueError('Collection checkpoint lacks environment contract')
            for key in ('env_id', 'observation_dim', 'action_dim', 'obs_mode', 'robot_uids', 'control_mode', 'reward_mode', 'action_low', 'action_high'):
                if saved.get(key) != contract.get(key):
                    raise ValueError(f'Collection checkpoint contract mismatch: {key}')
        horizon = contract['horizon']
        if args.transitions % (n*horizon):
            raise ValueError('Collection must contain complete vector episodes')
        chunks = {k: [] for k in ('obs', 'action', 'next_obs', 'reward', 'terminated', 'truncated', 'episode_ids', 'policy_source', 'contact_force_norms', 'contact_valid')}
        for step in range(args.transitions//n):
            use_policy = learner is not None and step*n >= args.initial_transitions and (step//horizon) % 2 == 1
            action = (torch.as_tensor(learner.act(_to_numpy(obs), deterministic=True), device=obs.device)
                      if use_policy else _random_action(env, obs, num_envs=n, device=obs.device))
            base = env.unwrapped
            forces = np.stack([_to_numpy(torch.linalg.norm(base.scene.get_pairwise_contact_forces(link, base.cube), dim=-1))
                               for link in (base.agent.finger1_link, base.agent.finger2_link)], axis=1)
            if not np.isfinite(forces).all(): raise ValueError('Nonfinite contact force')
            chunks['contact_force_norms'].append(forces.copy())
            chunks['contact_valid'].append(np.full(n, step % horizon != 0))
            nxt, reward, term, trunc, info = env.step(action)
            if _to_numpy(term).any() or not np.all(_to_numpy(trunc) == ((step+1)%horizon == 0)):
                raise ValueError('Collection boundary contradicts success-suppressed contract')
            true_next = _true_next_observation(nxt, term, trunc, info)
            for key, value in dict(obs=obs, action=action, next_obs=true_next, reward=reward,
                                   terminated=term, truncated=trunc).items():
                a = _to_numpy(value).copy().reshape(n, -1)
                if not np.isfinite(a).all(): raise ValueError('Nonfinite real data')
                chunks[key].append(a)
            chunks['episode_ids'].append(np.arange(n)+(step//horizon)*n)
            chunks['policy_source'].append(np.full(n, int(use_policy)))
            obs = nxt
        arrays = {k:np.concatenate(v) for k,v in chunks.items()}
        ti, vi = episode_split(arrays['episode_ids'], 5)
        np.savez_compressed(out/'real_dataset.npz', **arrays, train_indices=ti, holdout_indices=vi)
        write(out/'dataset_metadata.json', dict(environment_contract=contract, observation_codec=codec,
            real_env_steps=args.transitions, vector_steps=args.transitions//n, policy_gradient_steps=0,
            collection='alternating whole vector episodes: uniform random / frozen SAC' if learner else 'uniform random',
            contact_labels='current-state Panda finger/cube force norms (N); reset rows excluded; contact >=0.5N on either finger, not grasp angle predicate',
            train_holdout='fixed disjoint whole episodes modulo 5'))
    finally:
        if env is not None: env.close()
        if learner is not None:
            assert_state_equal(before, learner.state_dict())
            write(out/'frozen_collection_check.json',dict(learner_unchanged=True, updates_executed=0))


def conditional_metrics(model, dataset, arrays, ids, index, seed):
    means, variances = model.predict(dataset.inputs[ids])
    samples = model.sample_predictions(means, variances, np.random.default_rng(seed))
    current = arrays['obs'][ids]
    prepared = model.prepare_dataset(dataset)
    truth = model.reconstruct(current, prepared.targets[ids, :model.obs_dim], normalize=False)
    masks = {'all': np.ones(len(ids), bool)}
    for a in (0, 1):
        masks[f'current_grasp_{a}'] = current[:, index] == a
        for b in (0, 1): masks[f'{a}->{b}'] = (current[:, index] == a)&(truth[:, index] == b)
    if 'contact_force_norms' in arrays:
        valid = arrays['contact_valid'][ids].astype(bool)
        contact = np.max(arrays['contact_force_norms'][ids], axis=1) >= .5
        masks['current_contact'] = valid & contact
        masks['current_no_contact'] = valid & ~contact
    canonical = dataset.inputs[ids].copy()
    canonical[:, :model.obs_dim] = model.geometry.canonical(canonical[:, :model.obs_dim])
    normalized, _ = model._normalized(canonical)
    normalized = normalized.detach().cpu().numpy()
    results = {}
    for label, predictions in [('conditional_mean', means), ('sampled', samples)]:
        rows = []
        for member, pred in enumerate(predictions):
            raw = model.reconstruct(current, pred[:, :-1], normalize=False)
            restored = model.reconstruct(current, pred[:, :-1])
            diagnostic = StateDiagnostics(model.geometry.layout, model.observation_codec)
            diagnostic.observe(current, raw, pred[:, -1], means, model.elites)
            diagnostic.observe_projected(restored)
            row = dict(member=member, validity=diagnostic.records[0], groups={})
            for name, mask in masks.items():
                count = int(mask.sum())
                row['groups'][name] = dict(count=count, status='observed' if count else 'UNVERIFIED: no labels')
                if not count: continue
                row['groups'][name].update(
                    normalized_input_abs_max=float(np.abs(normalized[mask]).max()),
                    normalized_input_abs_p99=float(np.quantile(np.abs(normalized[mask]),.99)),
                    reward_min=float(pred[mask,-1].min()), reward_max=float(pred[mask,-1].max()),
                    reward_noise_contribution_abs_max=float(np.abs(pred[mask,-1]-means[member,mask,-1]).max()),
                    physical_state_rmse=float(np.sqrt(np.mean((raw[mask]-truth[mask])**2))),
                    reward_rmse=float(np.sqrt(np.mean((pred[mask,-1]-arrays['reward'][ids][mask,0])**2))),
                    grasp_brier_or_mse=float(np.mean((raw[mask,index]-truth[mask,index])**2)),
                    grasp_threshold_accuracy=float(np.mean((raw[mask,index] >= .5) == truth[mask,index])))
                from dynamics_shift.models.quaternion import rotation_error
                row['groups'][name]['rotation_mae_rad'] = {key:float(rotation_error(restored[mask,sl],truth[mask,sl]).mean())
                    for key,sl in model.geometry.slices.items()}
            row['stored_finite'] = bool(np.isfinite(restored).all() and np.isfinite(pred[:,-1]).all())
            row['projected_quaternion_valid'] = all(np.allclose(np.linalg.norm(restored[:,sl],axis=1),1.,atol=1e-5) for sl in model.geometry.slices.values())
            row['stored_binary_valid'] = bool(np.isin(restored[:,index], [0,1]).all())
            rows.append(row)
        results[label] = rows
    return results


@torch.no_grad()
def coordinate_loss(model, dataset):
    prepared = model.prepare_dataset(dataset)
    result = {}
    for partition, ids in [('train',dataset.train_indices), ('holdout',dataset.holdout_indices)]:
        x,y = model._normalized(prepared.inputs[ids], prepared.targets[ids])
        rows = []
        for member in model.members:
            mean, logvar = member(x)
            terms = .5*((mean-y).square()*(-logvar).exp()+logvar)
            if model.binary:
                idx=model.binary.indices
                terms[:,idx]=torch.nn.functional.binary_cross_entropy_with_logits(mean[:,idx],y[:,idx],reduction='none')
            rows.append(terms.mean(0).cpu().tolist())
        result[partition]=rows
    return dict(values=result, axes='partition/member/target coordinate; last coordinate reward',
                binary_likelihood='Bernoulli' if model.binary else 'Gaussian',
                continuous_likelihood='normalized Gaussian NLL excluding constant')


def fit(args, experiment, out):
    source = Path(args.dataset_dir)
    meta = json.loads((source/'dataset_metadata.json').read_text())
    codec = meta['observation_codec']
    if meta['environment_contract']['env_id'] != experiment.run.env.id:
        raise ValueError('Dataset/config task mismatch')
    for key in ('obs_mode','robot_uids','control_mode','reward_mode'):
        if meta['environment_contract'][key] != getattr(experiment.run.env,key):
            raise ValueError(f'Dataset/config mismatch: {key}')
    with np.load(source/'real_dataset.npz', allow_pickle=False) as data:
        arrays = {k:data[k].copy() for k in data.files}
    if len(arrays['obs']) > experiment.model.model_max_samples:
        raise ValueError('Dataset exceeds unchanged model_max_samples; collect a smaller fixed dataset')
    ti, vi = arrays['train_indices'], arrays['holdout_indices']
    if (len(ti)<2 or not len(vi) or np.intersect1d(ti,vi).size or
        not np.array_equal(np.sort(np.concatenate([ti,vi])), np.arange(len(arrays['obs']))) or
        set(arrays['episode_ids'][ti]) & set(arrays['episode_ids'][vi])):
        raise ValueError('Invalid or leaking episode partition')
    index = codec['layout']['extra.is_grasped'][0]
    cov = {name:coverage(arrays, ids, index) for name,ids in [('train',ti),('holdout',vi)]}
    write(out/'coverage.json', cov)
    write(out/'dataset_reference.json', dict(path=str(source.resolve()), metadata=meta,
        sha256=hashlib.sha256((source/'real_dataset.npz').read_bytes()).hexdigest()))
    dataset = ModelDataset(np.concatenate((arrays['obs'],arrays['action']),1),
        np.concatenate((arrays['next_obs']-arrays['obs'],arrays['reward']),1), ti, vi)
    if not 3 <= args.initial_transitions < len(arrays['obs']):
        raise ValueError('Initial normalizer partition must be a strict real-data prefix')
    initial_ids = np.arange(args.initial_transitions)
    first_ti = ti[ti < args.initial_transitions]
    first_vi = vi[vi < args.initial_transitions]
    initial = ModelDataset(dataset.inputs[initial_ids], dataset.targets[initial_ids], first_ti, first_vi)
    write(out/'initial_coverage.json', {name:coverage(arrays, ids, index)
        for name,ids in [('train', first_ti),('holdout', first_vi)]})
    rows = []
    for representation in ('legacy_gaussian_delta','bernoulli_next_v1'):
        target = out/representation; target.mkdir()
        config = replace(experiment.model, binary_features=representation)
        write(target/'model_config.json', asdict(config))
        model = None
        try:
            with isolated_rng(experiment.run.training.device):
                random.seed(0); np.random.seed(0); torch.manual_seed(0)
                model = ProbabilisticEnsemble(arrays['obs'].shape[1], arrays['action'].shape[1], config,
                    experiment.run.training.device, 0, observation_layout=codec['layout'], observation_codec=codec, preserve_start=True)
                torch.save({'model':model.state_dict()}, target/'initial_model.pt')
                for round_id in range(1,args.rounds+1):
                    fitting_data = initial if round_id == 1 else dataset
                    metrics = model.train(fitting_data)
                    torch.save({'model':model.state_dict(), 'training_resume_supported':False}, target/f'model_round_{round_id}.pt')
                    write(target/f'normalizer_round_{round_id}.json', {k:v.cpu().tolist() for k,v in model.normalization.items()})
                    conditional = {name:conditional_metrics(model,dataset,arrays,ids,index,123)
                                   for name,ids in [('train',ti),('holdout',vi)]}
                    row = dict(representation=representation, fitting_round=round_id,
                        real_env_steps=len(arrays['obs']), member_optimizer_steps=model.train_steps,
                        policy_gradient_steps=0, **metrics, **model_errors(model,fitting_data,codec['layout']), conditional=conditional,
                        normalized_coordinate_loss=coordinate_loss(model, fitting_data),
                        fit_partition='initial_prefix' if round_id == 1 else 'full_fixed_dataset',
                        conditional_scope='full fixed dataset including later-coverage diagnostic data; never used for round-1 fit')
                    with (target/'metrics.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
                    rows.append(row)
            write(target/'branch_status.json', dict(status='complete'))
        except Exception as error:
            if model is not None:
                torch.save({'model':model.state_dict(), 'diagnostic_failure':True}, target/'failed_model.pt')
            write(target/'branch_status.json', dict(status='failed', error_type=type(error).__name__, error=str(error)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for representation in ('legacy_gaussian_delta','bernoulli_next_v1'):
        r = [row for row in rows if row['representation']==representation]
        x = [row['fitting_round'] for row in r]
        for ax,key in zip(axes.flat,('model_train_loss','model_validation_loss','state_prediction_rmse','reward_prediction_rmse')):
            ax.plot(x,[np.mean(row[key]) for row in r],label=representation); ax.set_title(key)
            ax.set_xlabel('fitting_round; fixed real dataset'); ax.set_yscale('symlog',linthresh=.01); ax.legend()
    fig.tight_layout();fig.savefig(out/'paired_diagnostics.png');plt.close(fig)
    write(out/'gate.json', dict(technical_fit_completed=len(rows)==2*args.rounds,
        grasp_transition_coverage_complete=all(v>0 for group in cov.values() for v in group.values()),
        contact_conditioning='recorded current contact/no-contact; inspect per-group counts' if 'contact_force_norms' in arrays else 'UNVERIFIED: labels absent',
        semantic_invariants='finite/quaternion/binary checks recorded; relational/goal drift requires review without extra projection',
        phase_d_authorized=False, decision='REVIEW_REQUIRED: inspect coverage, errors and invariants; no automatic RL launch',
        metric_caution='Gaussian and mixed Bernoulli/Gaussian total NLL are different objectives; compare physical errors separately'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['collect','fit'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--output-root',default='outputs/grasp_diagnostic')
    parser.add_argument('--dataset-dir')
    parser.add_argument('--policy-checkpoint')
    parser.add_argument('--transitions',type=int,default=9600)
    parser.add_argument('--initial-transitions',type=int,default=4000)
    parser.add_argument('--rounds',type=int,default=5)
    args=parser.parse_args()
    if not 1<=args.rounds<=5: parser.error('rounds must be 1..5')
    if args.mode=='fit' and not args.dataset_dir: parser.error('fit requires --dataset-dir')
    experiment=load_experiment(args.config,algorithm='mbpo')
    out=Path(args.output_root)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    write_run_manifest(out,experiment.to_dict(),seeds={'collection':0,'model':0,'diagnostic_sampling':123})
    write(out/'diagnostic_arguments.json',vars(args))
    threads = torch.get_num_threads()
    torch.set_num_threads(experiment.run.training.torch_threads)
    try:
        collect(args,experiment,out) if args.mode=='collect' else fit(args,experiment,out)
        write(out/'status.json',dict(status='complete',policy_updates=0,phase_d_started=False))
    except Exception as error:
        write(out/'status.json',dict(status='failed',error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        torch.set_num_threads(threads)
        print(out,flush=True)

if __name__=='__main__':main()
