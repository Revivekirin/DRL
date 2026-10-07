"""Opt-in server environment contract check through the common CLI; no learner."""
import json
import numpy as np
from dynamics_shift.config import ExperimentConfig
from dynamics_shift.envs import make_env
from dynamics_shift.data.model_data import RealReplayBuffer
from .interaction import (_training_contract, _to_numpy, _random_action,
                          collect_transition, _extract_vector_success)


def check_environment(config):
    if config.env.backend != 'maniskill':
        raise ValueError('Use existing MuJoCo regression tests for that backend')
    env = make_env(ExperimentConfig(config.env, None, config.seed))
    try:
        n = config.env.num_envs
        vector = config.env.sim_backend == 'gpu'
        contract = _training_contract(config, env, num_envs=n)
        obs, _ = env.reset(seed=0)
        replay = RealReplayBuffer(n * 2, contract['observation_dim'], contract['action_dim'], 0)
        boundaries = successes = truncations = partial_resets = partial_boundaries = 0
        lengths = np.zeros(n, dtype=int)
        horizon = contract['horizon']
        # Deliberately stagger one environment to exercise final masks/partial resets.
        for step in range(horizon * 3):
            if vector and step == horizon // 2:
                obs, _ = env.reset(options={'env_idx': [0]})
                partial_resets += 1
                lengths[0] = 0
            action = (_random_action(env, obs, num_envs=n, device=obs.device) if vector
                      else env.action_space.sample())
            action_array = _to_numpy(action).reshape(n, contract['action_dim'])
            if not np.isfinite(action_array).all() or np.any(action_array < np.asarray(contract['action_low'])) or np.any(action_array > np.asarray(contract['action_high'])):
                raise ValueError('Invalid action bounds/finiteness')
            lengths += 1
            obs, reward, terminated, truncated, info = collect_transition(
                env, replay, obs, action, num_envs=n, automatic_reset=vector)
            term = _to_numpy(terminated).reshape(n)
            trunc = _to_numpy(truncated).reshape(n)
            if term.dtype != np.bool_ or trunc.dtype != np.bool_:
                raise ValueError('Nonboolean termination flags')
            if not np.array_equal(trunc, lengths == horizon):
                raise ValueError('Per-environment horizon/truncation mismatch')
            done = term | trunc
            partial_boundaries += int(done.any() and not done.all())
            lengths[done] = 0
            if vector and term.any():
                raise ValueError('Training wrapper did not suppress success termination')
            values = _to_numpy(obs)
            expected_shape = (n, contract['observation_dim']) if vector else (contract['observation_dim'],)
            if values.shape != expected_shape or str(values.dtype) != 'float32':
                raise ValueError('Observation shape/dtype contract changed after step/reset')
            if not np.isfinite(values).all() or not np.isfinite(_to_numpy(reward)).all():
                raise ValueError('Nonfinite observations/rewards')
            boundaries += int((term | trunc).sum())
            truncations += int(trunc.sum())
            if vector:
                successes += int(_extract_vector_success(info, term, num_envs=n).sum())
            else:
                from dynamics_shift.evaluation.contracts import success_flag
                successes += int(success_flag(info, bool(term[0])))
                if term[0] or trunc[0]:
                    saved = replay._arrays['next_obs'].copy()
                    obs, _ = env.reset()
                    np.testing.assert_array_equal(saved, replay._arrays['next_obs'])
        if vector and not partial_boundaries:
            raise ValueError('Staggered partial episode boundaries were not exercised')
        if not boundaries:
            raise ValueError('No episode boundary was exercised')
        print(json.dumps(dict(status='PASS', environment_contract=contract,
            real_env_steps=3*horizon*n, vector_steps=3*horizon, episode_boundaries=boundaries,
            truncations=truncations, explicit_partial_resets=partial_resets, partial_boundaries=partial_boundaries,
            success_observations=successes,
            success_termination=('DISABLED by GPU training protocol' if vector else
                                 'verified' if successes else 'UNVERIFIED: no success observed'),
            learner_updates=0)))
    finally:
        env.close()
