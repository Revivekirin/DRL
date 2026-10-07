"""PushCube MBPO contracts and diagnostics; shared learner/model/replay stay intact."""
from pathlib import Path
import numpy as np
import yaml
from dynamics_shift.algorithms.mbpo.config import MBPOConfig
from dynamics_shift.algorithms.mbpo.rollouts import generate_rollouts
from dynamics_shift.experiments.config import RunConfig


def load_pushcube_mbpo_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    raw = dict(raw)
    model = MBPOConfig(**raw.pop('mbpo'))
    config = RunConfig.from_dict(raw)
    if config.seed != 0 or config.env.backend != 'maniskill' or config.env.id != 'PushCube-v1':
        raise ValueError('PushCube MBPO smoke requires nominal PushCube, seed 0')
    if config.env.sim_backend != 'gpu' or config.env.num_envs < 2:
        raise ValueError('PushCube MBPO requires GPU vector simulation with num_envs >= 2')
    if not config.training.device.startswith('cuda'):
        raise ValueError('GPU learner device must be explicit')
    if config.evaluation.interval != 0:
        raise ValueError('This bounded smoke evaluates at completion only; evaluation.interval must be 0')
    if config.training.replay_capacity < config.env.num_envs:
        raise ValueError('Replay must hold at least one full vector batch')
    if model.rollout_horizon != 1:
        raise ValueError('PushCube MBPO smoke supports rollout_horizon=1 only')
    if config.training.learning_starts < 3 or config.training.batch_size < 2:
        raise ValueError('Need learning_starts >= 3 and batch_size >= 2')
    if config.training.real_env_steps % config.env.num_envs:
        raise ValueError('real_env_steps must be divisible by num_envs')
    if config.training.real_env_steps < max(config.training.learning_starts, config.training.batch_size):
        raise ValueError('Smoke must reach model fitting and learner updates')
    return config, model


def observation_layout(structured, flattened):
    """Discover leaf slices and verify concatenation against the actual state tensor.

    ManiSkill 3.0.1 get_obs(unflattened=True) uses ordered nested dictionaries.
    Fail on an unknown layout instead of guessing quaternion indices.
    """
    def numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    flat = numpy(flattened)
    leaves, layout = [], {}
    offset = 0

    def visit(node, prefix=''):
        nonlocal offset
        for key, value in node.items():
            path = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                visit(value, path)
            else:
                array = numpy(value).reshape(flat.shape[0], -1)
                layout[path] = [offset, offset + array.shape[1]]
                offset += array.shape[1]
                leaves.append(array)
    visit(structured)
    np.testing.assert_allclose(np.concatenate(leaves, axis=1), flat, rtol=1e-6, atol=1e-6)
    for key, width in [('extra.tcp_pose', 7), ('extra.obj_pose', 7), ('extra.goal_pos', 3)]:
        if key not in layout or layout[key][1] - layout[key][0] != width:
            raise ValueError(f'Unsupported PushCube observation layout: {key}')
    return layout


class StateDiagnostics:
    """Report raw model validity, never silently project quaternions or clip rewards."""
    def __init__(self, layout):
        self.layout = layout
        self.records = []

    def observe(self, obs, next_obs, rewards, means=None, elites=None):
        if not all(np.isfinite(x).all() for x in (obs, next_obs, rewards)):
            raise FloatingPointError('Nonfinite model/real diagnostic data')
        record = {'count': len(next_obs), 'all_finite': True,
                  'reward_outside_0_1_fraction': float(np.mean((rewards < 0) | (rewards > 1))),
                  'reward_below_zero_fraction': float(np.mean(rewards < 0)),
                  'reward_above_one_fraction': float(np.mean(rewards > 1)),
                  'reward_min': float(np.min(rewards)), 'reward_max': float(np.max(rewards))}
        if means is not None and elites is not None:
            elite_rewards = means[elites, :, -1]
            if not np.isfinite(elite_rewards).all():
                raise FloatingPointError('Nonfinite elite mean rewards')
            record['elite_mean_reward_min'] = float(elite_rewards.min())
            record['elite_mean_reward_max'] = float(elite_rewards.max())
            record['elite_mean_reward_outside_0_1_fraction'] = float(np.mean(
                (elite_rewards < 0) | (elite_rewards > 1)))
        for key in ('extra.tcp_pose', 'extra.obj_pose'):
            start, stop = self.layout[key]
            deviation = np.abs(np.linalg.norm(next_obs[:, start + 3:stop], axis=1) - 1)
            record[key + '_quaternion_norm_mae'] = float(deviation.mean())
            record[key + '_quaternion_norm_max_error'] = float(deviation.max())
            record[key + '_quaternion_invalid_fraction_tol_0.01'] = float(np.mean(deviation > .01))
        start, stop = self.layout['extra.goal_pos']
        record['goal_position_drift_rmse'] = float(np.sqrt(np.mean((next_obs[:, start:stop] - obs[:, start:stop]) ** 2)))
        self.records.append(record)


def model_errors(model, dataset, layout=None):
    """Member-order mean-prediction RMSE, separate train and current holdout.

    Legacy top-level keys remain holdout-only. These calculations never sample
    stochastic predictions, change normalizers, or consume replay/runner RNG.
    """
    partitions = {}
    blocks = dict(layout or {})
    for key in ('extra.tcp_pose', 'extra.obj_pose'):
        if key in blocks:
            start, stop = blocks[key]
            blocks[key + '.position'] = [start, start + 3]
            blocks[key + '.quaternion_components'] = [start + 3, stop]
    for name, ids in (('train', dataset.train_indices), ('holdout', dataset.holdout_indices)):
        means, _ = model.predict(dataset.inputs[ids])
        targets = dataset.targets[ids]
        errors = means - targets[None]
        if not np.isfinite(errors).all():
            raise FloatingPointError('Nonfinite model validation predictions')
        partitions[name] = {
            'samples': len(ids),
            'state_prediction_rmse': np.sqrt(np.mean(errors[:, :, :model.obs_dim] ** 2, axis=(1, 2))).tolist(),
            'reward_prediction_rmse': np.sqrt(np.mean(errors[:, :, -1] ** 2, axis=1)).tolist(),
            'state_blocks': {},
        }
        for key, (start, stop) in blocks.items():
            partitions[name]['state_blocks'][key] = {
                'member_rmse': np.sqrt(np.mean(errors[:, :, start:stop] ** 2, axis=(1, 2))).tolist(),
                'zero_delta_baseline_rmse': float(np.sqrt(np.mean(targets[:, start:stop] ** 2))),
            }
    return {**{key: partitions['holdout'][key] for key in ('state_prediction_rmse', 'reward_prediction_rmse')},
            'prediction_errors': partitions,
            'error_semantics': {'array_axis': 'ensemble member index, not elite rank or state coordinate',
                'prediction': 'conditional mean in original coordinates; no Gaussian sampling',
                'state_target': 'next observation minus current observation',
                'holdout': 'current refit real snapshot partition; may have appeared in earlier training',
                'train': 'current real training partition after fitting; not bootstrap duplicates',
                'zero_delta_baseline': 'predict unchanged observation; diagnostic only'}}


def generate_pushcube_rollouts(learner, model, real, synthetic, settings, rng, diagnostics, contract):
    if contract['termination_policy'] != 'ignore_terminations':
        raise ValueError('Synthetic termination=False requires training ignore_terminations')
    if settings.rollout_horizon != 1:
        raise ValueError('Only horizon 1 is supported')
    if model.refit_count < 1 or model.normalization is None:
        raise RuntimeError('Cannot generate or sample synthetic data before a real-only model fit')
    # In this training protocol success does NOT terminate. A model-computation
    # cutoff is neither a physical terminal nor an environment TimeLimit.
    return generate_rollouts(learner, model, real, synthetic, settings.rollout_batch_size,
                             1, rng, diagnostics=diagnostics)


def episode_split(episode_ids, modulus):
    """Deterministic whole-episode split, fixed across every fit round."""
    holdout = np.asarray(episode_ids) % modulus == modulus - 1
    train_ids, holdout_ids = np.flatnonzero(~holdout), np.flatnonzero(holdout)
    if len(train_ids) < 2 or not len(holdout_ids):
        raise ValueError('Collection must contain both train and holdout episodes')
    return train_ids, holdout_ids
