"""ManiSkill state MBPO contracts and diagnostics; shared learner/model/replay stay intact."""
import numpy as np
from dynamics_shift.algorithms.mbpo.rollouts import generate_rollouts


def load_pushcube_mbpo_config(path):
    # Compatibility API; all public/legacy formats use one validation path.
    from dynamics_shift.experiments.dispatch import load_experiment
    experiment = load_experiment(path, algorithm='mbpo')
    if experiment.run.env.backend != 'maniskill':
        raise ValueError('Expected ManiSkill model contract')
    return experiment.run, experiment.model


from dynamics_shift.envs.state_codec import observation_layout


class StateDiagnostics:
    """Report raw model validity, never silently project quaternions or clip rewards."""
    def __init__(self, layout, codec=None):
        self.layout = layout
        self.codec = codec or {}
        self.poses = self.codec.get("quaternion_poses", [k for k in ("extra.tcp_pose", "extra.obj_pose") if k in layout])
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
        for key in self.poses:
            start, stop = self.layout[key]
            deviation = np.abs(np.linalg.norm(next_obs[:, start + 3:stop], axis=1) - 1)
            record[key + '_quaternion_norm_mae'] = float(deviation.mean())
            record[key + '_quaternion_norm_max_error'] = float(deviation.max())
            record[key + '_quaternion_invalid_fraction_tol_0.01'] = float(np.mean(deviation > .01))
        for key in self.codec.get('invariant_fields', ['extra.goal_pos'] if 'extra.goal_pos' in self.layout else []):
            start, stop = self.layout[key]
            name = 'goal_position_drift_rmse' if key == 'extra.goal_pos' else key + '_drift_rmse'
            record[name] = float(np.sqrt(np.mean((next_obs[:, start:stop] - obs[:, start:stop]) ** 2)))
        for key in self.codec.get('boolean_fields', []):
            start, stop = self.layout[key]
            values = next_obs[:, start:stop]
            record[key + '_outside_0_1_fraction'] = float(np.mean((values < 0) | (values > 1)))
            record[key + '_distance_to_boolean_mean'] = float(np.minimum(np.abs(values), np.abs(values-1)).mean())
            record[key + '_range_violation_max'] = float(np.maximum(np.maximum(-values, values-1), 0).max())
            record[key + '_distance_to_boolean_max'] = float(np.minimum(np.abs(values), np.abs(values-1)).max())
        for vector, destination, source in self.codec.get('relations', []):
            start, stop = self.layout[vector]
            d, _ = self.layout[destination]
            a, _ = self.layout[source]
            residual = next_obs[:, start:stop] - (next_obs[:, d:d+3] - next_obs[:, a:a+3])
            record[vector + '_consistency_rmse'] = float(np.sqrt(np.mean(residual**2)))
        self.records.append(record)

    def observe_projected(self, next_obs):
        self.records[-1]['projected_quaternion_max_norm_error'] = {
            key: float(np.abs(np.linalg.norm(next_obs[:, self.layout[key][0]+3:self.layout[key][1]], axis=1)-1).max())
            for key in self.poses}


def model_errors(model, dataset, layout=None):
    """Member-order mean-prediction RMSE, separate train and current holdout.

    Legacy top-level RMSE keys remain holdout-only conditional-mean errors.
    Separate sampled validity diagnostics use a private fixed RNG; neither path
    changes normalizers or consumes replay/runner RNG.
    """
    if hasattr(model, "prepare_dataset"):
        dataset = model.prepare_dataset(dataset)
    binary_ids = model.binary.indices if getattr(model, 'binary', None) else []
    partitions = {}
    blocks = dict(layout or {})
    for key in (getattr(model, 'observation_codec', None) or {}).get('quaternion_poses', ('extra.tcp_pose', 'extra.obj_pose')):
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
        if getattr(model, 'geometry', None):
            from dynamics_shift.models.quaternion import rotation_error
            current = dataset.inputs[ids, :model.obs_dim]
            truth = current + targets[:, :model.obs_dim]
            raw_next = current[None] + means[:, :, :model.obs_dim]
            partitions[name]['conditional_mean_rotation'] = {}
            for key, sl in model.geometry.slices.items():
                angles = rotation_error(raw_next[:, :, sl], truth[None, :, sl])
                partitions[name]['conditional_mean_rotation'][key] = {
                    'raw_norm_mae': np.abs(np.linalg.norm(raw_next[:, :, sl], axis=-1)-1).mean(axis=1).tolist(),
                    'normalized_angle_mae_rad': angles.mean(axis=1).tolist(),
                    'normalized_angle_rmse_rad': np.sqrt(np.mean(angles**2, axis=1)).tolist(),
                    'unchanged_angle_mae_rad': float(rotation_error(current[:, sl], truth[:, sl]).mean()),
                }
        for key, (start, stop) in blocks.items():
            partitions[name]['state_blocks'][key] = {
                'member_rmse': np.sqrt(np.mean(errors[:, :, start:stop] ** 2, axis=(1, 2))).tolist(),
                'zero_delta_baseline_rmse': float(np.sqrt(np.mean((targets[:, start:stop] - dataset.inputs[ids, start:stop] if start in binary_ids and stop == start+1 else targets[:, start:stop]) ** 2))),
            }
    validity = {}
    if getattr(model, 'geometry', None):
        from dynamics_shift.models.quaternion import rotation_error
        ids = dataset.holdout_indices
        means, variances = model.predict(dataset.inputs[ids])
        noise = np.random.default_rng(0).standard_normal(means.shape)  # diagnostic RNG only
        current = dataset.inputs[ids, :model.obs_dim]
        truth = current + dataset.targets[ids, :model.obs_dim]
        for label, predictions in [('conditional_mean', means),
                                   ('sampled_prediction', model.sample_predictions(means, variances, np.random.default_rng(0)) if hasattr(model, 'sample_predictions') else means + np.sqrt(variances) * noise)]:
            validity[label] = []
            for member in range(len(predictions)):
                raw = model.reconstruct(current, predictions[member, :, :model.obs_dim], normalize=False)
                restored = model.reconstruct(current, predictions[member, :, :model.obs_dim])
                diagnostic = StateDiagnostics(model.geometry.layout, getattr(model, "observation_codec", None))
                diagnostic.observe(current, raw, predictions[member, :, -1])
                validity[label].append({'member': member, 'raw': diagnostic.records[0],
                    'normalized_angle_mae_rad': {key: float(rotation_error(restored[:, sl], truth[:, sl]).mean())
                                                for key, sl in model.geometry.slices.items()}})
    return {'holdout_prediction_validity': validity, **{key: partitions['holdout'][key] for key in ('state_prediction_rmse', 'reward_prediction_rmse')},
            'prediction_errors': partitions,
            'error_semantics': {'array_axis': 'ensemble member index, not elite rank or state coordinate',
                'prediction': 'conditional mean in original coordinates; no Gaussian sampling',
                'binary_target': 'absolute next-state Bernoulli probability' if binary_ids else 'legacy continuous delta',
                'state_target': ('aligned quaternion delta chart; other nonbinary coordinates next minus current'
                                 if getattr(model, 'geometry', None) else
                                 'next observation minus current observation'),
                'holdout': 'current refit real snapshot partition; may have appeared in earlier training',
                'train': 'current real training partition after fitting; not bootstrap duplicates',
                'zero_delta_baseline': 'predict unchanged observation; diagnostic only'}}


def generate_maniskill_rollouts(learner, model, real, synthetic, settings, rng, diagnostics, contract):
    from dynamics_shift.envs.maniskill_tasks import state_task
    state_task(contract['env_id'])  # audited success-only tasks
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

# Legacy API for old PushCube-only callers that did not pass env_id.
def generate_pushcube_rollouts(learner, model, real, synthetic, settings, rng, diagnostics, contract):
    return generate_maniskill_rollouts(learner, model, real, synthetic, settings, rng, diagnostics,
                                      {'env_id': 'PushCube-v1', **contract})
