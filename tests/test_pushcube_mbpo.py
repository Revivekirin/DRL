"""Server-side contract tests. No simulator, optimizer steps or trained-policy claims."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from dynamics_shift.algorithms.mbpo.config import MBPOConfig
from dynamics_shift.algorithms.mbpo.pushcube import (
    load_pushcube_mbpo_config, observation_layout, StateDiagnostics,
    model_errors, generate_pushcube_rollouts,
)
from dynamics_shift.algorithms.mbpo.rollouts import mixed_batch
from dynamics_shift.data.model_data import RealReplayBuffer, ModelReplayBuffer, ModelDataset
from dynamics_shift.experiments.train_pushcube_sac import _true_next_observation, _add_replay_batch


def test_smoke_config():
    config, model = load_pushcube_mbpo_config(Path(__file__).parents[1] / 'configs/testing/mbpo_pushcube_smoke.yaml')
    assert config.env.num_envs == 4 and config.seed == 0
    assert config.training.real_env_steps == 800 and model.rollout_horizon == 1


def test_layout_verifies_order_and_quaternion_diagnostics():
    pose = np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
    data = {'agent': {'qpos': np.zeros((1, 2))}, 'extra': {
        'tcp_pose': pose, 'goal_pos': np.zeros((1, 3)), 'obj_pose': pose}}
    flat = np.concatenate([data['agent']['qpos'], pose, data['extra']['goal_pos'], pose], axis=1)
    layout = observation_layout(data, flat)
    altered = flat.copy()
    altered[:, layout['extra.obj_pose'][0] + 3] = 2
    diag = StateDiagnostics(layout)
    diag.observe(flat, altered, np.array([1.2]))
    assert diag.records[0]['extra.obj_pose_quaternion_invalid_fraction_tol_0.01'] == 1
    assert diag.records[0]['reward_outside_0_1_fraction'] == 1
    with pytest.raises(AssertionError):
        observation_layout(data, altered)


def test_state_reward_error_separation():
    dataset = ModelDataset(np.zeros((3, 3)), np.zeros((3, 3)), np.array([0, 1]), np.array([2]))
    model = SimpleNamespace(obs_dim=2, predict=lambda x: (np.array([[[3., 3., 4.]]]), None))
    result = model_errors(model, dataset)
    assert result['state_prediction_rmse'] == [3.]
    assert result['reward_prediction_rmse'] == [4.]


def test_fit_gate_real_only_and_synthetic_flags():
    real, synthetic = RealReplayBuffer(8, 2, 1, 0), ModelReplayBuffer(8, 2, 1, 1)
    for _ in range(4):
        real.add(np.zeros(2), np.zeros(1), 1., np.ones(2), False, True)
    model = SimpleNamespace(obs_dim=2, refit_count=0, normalization=None, elites=[0],
        predict=lambda x: (np.zeros((1, len(x), 3)), np.zeros((1, len(x), 3))))
    learner = SimpleNamespace(act=lambda obs, deterministic=False: np.zeros((len(obs), 1)))
    settings = MBPOConfig(rollout_batch_size=4)
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match='before'):
        generate_pushcube_rollouts(learner, model, real, synthetic, settings, rng, None,
                                  {'termination_policy': 'ignore_terminations'})
    assert len(synthetic) == 0
    model.refit_count, model.normalization = 1, {}
    with pytest.raises(ValueError, match='termination'):
        generate_pushcube_rollouts(learner, model, real, synthetic, settings, rng, None,
                                  {'termination_policy': 'terminate_on_success'})
    with pytest.raises(ValueError, match='horizon'):
        generate_pushcube_rollouts(learner, model, real, synthetic, replace(settings, rollout_horizon=2), rng, None,
                                  {'termination_policy': 'ignore_terminations'})
    assert generate_pushcube_rollouts(learner, model, real, synthetic, settings, rng, None,
                                    {'termination_policy': 'ignore_terminations'}) == 4
    batch = synthetic.sample(8)
    assert not batch.terminated.any() and not batch.truncated.any()
    batch, nr, ns = mixed_batch(real, synthetic, 5, .5, rng)
    assert (nr, ns) == (2, 3)  # Report the actual ratio (0.6), not the requested 0.5.
    assert len(real) == 4
    with pytest.raises(TypeError):
        ModelDataset.from_real_replay(synthetic, rng, .2, 4)


def test_vector_final_observation_and_individual_flags():
    replay = RealReplayBuffer(8, 2, 1, 0)
    obs = torch.zeros(2, 2)
    reset_obs = torch.tensor([[0., 0.], [5., 6.]])
    terminal = torch.tensor([[3., 4.], [5., 6.]])
    term, trunc = torch.tensor([False, False]), torch.tensor([True, False])
    actual_next = _true_next_observation(reset_obs, term, trunc,
                                        {'final_observation': terminal, '_final_observation': trunc})
    _add_replay_batch(replay, obs, torch.zeros(2, 1), torch.ones(2), actual_next, term, trunc, num_envs=2)
    np.testing.assert_array_equal(replay._arrays['next_obs'][:2], terminal.numpy())
    np.testing.assert_array_equal(replay._arrays['truncated'][:2, 0], [True, False])
    assert not replay._arrays['terminated'][:2].any()
