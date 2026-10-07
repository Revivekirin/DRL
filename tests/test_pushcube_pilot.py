"""Geometry/tracking regressions; optimizer tests require explicit server opt-in."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import MagicMock, patch
import json
import random
import numpy as np
import pytest
import torch
from dynamics_shift.models.quaternion import QuaternionDelta, rotation_error
from dynamics_shift.data.model_data import ModelDataset
from dynamics_shift.experiments.config import RunConfig, TrackingConfig
from dynamics_shift.utils.tracking import Tracker
from dynamics_shift.utils.checkpoint import capture_rng
from dynamics_shift.evaluation.pushcube import assert_state_equal

LAYOUT = {'extra.tcp_pose': [0, 7], 'extra.obj_pose': [7, 14], 'extra.goal_pos': [14, 17]}


def observations():
    obs = np.zeros((4, 17), dtype=np.float32)
    obs[:, [3, 10]] = 1
    return obs


def test_sign_equivalence_targets_and_reconstruction():
    codec = QuaternionDelta(LAYOUT)
    obs = observations()
    no = obs.copy()
    no[:, 3:7] *= -1
    no[:, 10:14] *= -1
    data = ModelDataset(np.c_[obs, np.zeros((4, 1))], np.c_[no-obs, np.zeros(4)], np.array([0,1]), np.array([2,3]))
    transformed = codec.dataset(data, 17)
    np.testing.assert_allclose(transformed.targets, 0, atol=1e-7)
    flipped = obs.copy(); flipped[:, 3:7] *= -1
    np.testing.assert_allclose(codec.canonical(flipped), codec.canonical(obs))
    delta = np.zeros_like(obs); delta[:, 3] = .1
    raw = codec.reconstruct(obs, delta, normalize=False)
    fixed = codec.reconstruct(obs, delta)
    assert np.linalg.norm(raw[0, 3:7]) > 1.01
    np.testing.assert_allclose(np.linalg.norm(fixed[:, 3:7], axis=1), 1)
    np.testing.assert_allclose(rotation_error(fixed[:, 3:7], -obs[:, 3:7]), 0, atol=1e-6)
    q90 = np.array([[np.sqrt(.5), 0, 0, np.sqrt(.5)]])
    np.testing.assert_allclose(rotation_error(q90, np.array([[1.,0,0,0]])), np.pi/2)


def test_tracking_ledger_equals_sdk_and_preserves_rng(tmp_path):
    sdk = MagicMock()
    def log(payload):
        random.random(); np.random.rand(); torch.rand(1)
    sdk.init.return_value.log.side_effect = log
    config = replace(RunConfig(), tracking=TrackingConfig(mode='offline'))
    state = capture_rng(torch.device('cpu'))
    with patch.dict('sys.modules', {'wandb': sdk}):
        tracker = Tracker(config, tmp_path)
        for update in (1,2):
            tracker.scalars('policy_update', {'loss': update+.25}, 128,
                            axis='policy_gradient_steps', axis_value=update)
        tracker.scalars('fit', {'nll': 2.}, 128, axis='fitting_round', axis_value=1)
        tracker.finish()
    assert_state_equal(state, capture_rng(torch.device('cpu')))
    rows = [json.loads(line) for line in (tmp_path/'tracking_metrics.jsonl').read_text().splitlines()]
    assert len(rows) == 3
    assert [row['metrics']['metric_event_id'] for row in rows] == [1,2,3]
    assert all(row['metrics']['real_env_steps']==128 for row in rows)
    assert [row['metrics'] for row in rows] == [call.args[0] for call in sdk.init.return_value.log.call_args_list]
    assert rows[-1]['x_axis'] == 'fitting_round'
    assert json.loads((tmp_path/'tracking_summary.json').read_text())['tracking_errors'] == 0


@pytest.mark.training
def test_fit_restores_start_model_and_optimizer(monkeypatch):
    from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble
    from dynamics_shift.algorithms.mbpo.config import MBPOConfig
    import dynamics_shift.models.probabilistic_ensemble as module
    config = MBPOConfig(ensemble_size=1, elite_size=1, model_hidden_dims=(4,),
                        model_max_epochs=1, model_batch_size=8)
    model = ProbabilisticEnsemble(2, 1, config, preserve_start=True)
    data = ModelDataset(np.ones((6,3), np.float32), np.ones((6,3), np.float32),
                        np.arange(4), np.array([4,5]))
    before_model = deepcopy(model.members.state_dict())
    before_opt = deepcopy(model.optimizers[0].state_dict())
    real_loss = module.gaussian_loss
    validation_calls = 0
    def controlled(mean, logvar, target):
        nonlocal validation_calls
        if not torch.is_grad_enabled():
            validation_calls += 1
            # Starting validation wins; later evaluation is deliberately worse.
            return torch.tensor(0. if validation_calls==1 else 10.)
        return real_loss(mean, logvar, target)
    monkeypatch.setattr(module, 'gaussian_loss', controlled)
    model.train(data)
    assert_state_equal(before_model, model.members.state_dict())
    assert_state_equal(before_opt, model.optimizers[0].state_dict())
    assert model.train_steps == 1
    assert model.last_metrics['restored_fit_start'] == [True]
    state = model.state_dict()
    loaded = ProbabilisticEnsemble.from_state_dict(state)
    assert loaded.preserve_start and loaded.geometry is None


def test_geometry_checkpoint_and_legacy_defaults():
    from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble
    from dynamics_shift.algorithms.mbpo.config import MBPOConfig
    config = MBPOConfig(ensemble_size=1, elite_size=1, model_hidden_dims=(4,))
    model = ProbabilisticEnsemble(17, 1, config, observation_layout=LAYOUT, preserve_start=True)
    restored = ProbabilisticEnsemble.from_state_dict(model.state_dict())
    assert restored.geometry.version == 'aligned_quaternion_delta_v1'
    legacy = model.state_dict(); legacy.pop('geometry'); legacy.pop('preserve_start')
    restored = ProbabilisticEnsemble.from_state_dict(legacy)
    assert restored.geometry is None and not restored.preserve_start


@pytest.mark.training
def test_update_diagnostics_do_not_change_learning_or_rng():
    from test_sac_update import make_learner, make_batch
    from dynamics_shift.algorithms.sac.learner import SACLearner
    from dynamics_shift.utils.checkpoint import restore_rng
    a = make_learner()
    b = SACLearner.from_state_dict(deepcopy(a.state_dict()))
    b.record_update_diagnostics = True
    batch = make_batch()
    state = capture_rng(torch.device('cpu'))
    ma = a.update(batch)
    after = capture_rng(torch.device('cpu'))
    restore_rng(state, torch.device('cpu'))
    mb = b.update(batch)
    assert_state_equal(a.state_dict(), b.state_dict())
    assert_state_equal(after, capture_rng(torch.device('cpu')))
    assert all(ma[k] == mb[k] for k in ma)
    assert np.isfinite(mb['entropy_estimate']) and np.isfinite(mb['policy_q_min_mean'])


def test_synthetic_rollout_projects_quaternion_but_keeps_horizon_nonterminal():
    from types import SimpleNamespace
    from dynamics_shift.algorithms.mbpo.rollouts import generate_rollouts
    from dynamics_shift.data.model_data import RealReplayBuffer, ModelReplayBuffer
    real, synthetic = RealReplayBuffer(8,17,1,0), ModelReplayBuffer(8,17,1,0)
    obs = observations()[0]
    real.add(obs, np.zeros(1), 0., obs, False, True)
    codec = QuaternionDelta(LAYOUT)
    def predict(x):
        mean = np.zeros((1,len(x),18)); mean[:,:,3] = .2
        return mean, np.zeros_like(mean)
    model = SimpleNamespace(obs_dim=17, elites=[0], predict=predict, reconstruct=codec.reconstruct)
    policy = SimpleNamespace(act=lambda x, deterministic=False: np.zeros((len(x),1)))
    generate_rollouts(policy, model, real, synthetic, 4, 1, np.random.default_rng(0))
    batch = synthetic.sample(4)
    np.testing.assert_allclose(np.linalg.norm(batch.next_obs[:,3:7],axis=1), 1)
    assert not batch.terminated.any() and not batch.truncated.any()


def test_ledger_audit_rejects_counter_mismatch(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('audit_mbpo', Path(__file__).parents[1]/'scripts/check_mbpo_tracking.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    (tmp_path/'metadata.json').write_text(json.dumps({'status':'complete','policy_gradient_steps':1}))
    (tmp_path/'tracking_summary.json').write_text(json.dumps({'tracking_errors':0}))
    (tmp_path/'tracking_metrics.jsonl').write_text('')
    with pytest.raises(AssertionError):
        module.audit(tmp_path)
