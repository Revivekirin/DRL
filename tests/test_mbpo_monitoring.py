"""Monitoring only: no optimizer steps, model fits, or real training."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch
from dynamics_shift.algorithms.mbpo.config import MBPORunConfig, load_mbpo_config
from dynamics_shift.algorithms.mbpo.monitoring import video_array, evaluate_nominal, RolloutDiagnostics, normalization_checks
from dynamics_shift.algorithms.mbpo.rollouts import generate_rollouts
from dynamics_shift.data.model_data import RealReplayBuffer, ModelReplayBuffer
from dynamics_shift.utils.tracking import Tracker


def test_full_diagnostic_parameters():
    full = load_mbpo_config('configs/experiment/mbpo_halfcheetah_source.yaml')
    diagnostic = load_mbpo_config('configs/experiment/mbpo_diagnostic_20k.yaml')
    assert full.mbpo == diagnostic.mbpo
    assert full.algo == diagnostic.algo
    assert replace(full.training, real_env_steps=20000) == diagnostic.training


def test_frame_conversion():
    frames = np.arange(2*4*6*3, dtype=np.uint8).reshape(2,4,6,3)
    np.testing.assert_array_equal(video_array(frames), frames.transpose(0,3,1,2))
    with pytest.raises(ValueError, match='dtype'):
        video_array(frames.astype(float))


@pytest.mark.parametrize('video', [False, True])
def test_isolated_evaluation(tmp_path, video):
    config = MBPORunConfig()
    env = MagicMock()
    env.spec.max_episode_steps = 2
    env.metadata = {'render_fps': 20}
    env.reset.return_value = (np.zeros(3), {})
    env.step.return_value = (np.zeros(3), 2., False, True, {})
    env.render.return_value = np.zeros((4,6,3), dtype=np.uint8)
    learner = SimpleNamespace(device=torch.device('cpu'), act=MagicMock(return_value=np.zeros(2)))
    tracker, sdk = MagicMock(), MagicMock()
    original = torch.get_rng_state().clone()
    def action(*args, **kwargs):
        torch.rand(1)
        assert kwargs == {'deterministic': True}
        return np.zeros(2)
    learner.act.side_effect = action
    with patch('dynamics_shift.algorithms.mbpo.monitoring.make_env', return_value=env) as factory, patch.dict('sys.modules', {'wandb': sdk}):
        result = evaluate_nominal(learner, config, tracker, tmp_path, 48, record_video=video)
    assert result['eval/source/mean_return'] == 2.
    assert torch.equal(original, torch.get_rng_state())
    assert learner.act.call_count == config.evaluation.episodes
    assert factory.call_args.kwargs == ({'render_mode': 'rgb_array'} if video else {})
    assert sdk.Video.call_count == int(video)
    if video:
        assert sdk.Video.call_args.args[0].shape == (2,3,4,6)
    else:
        env.render.assert_not_called()
    assert not list(tmp_path.iterdir())
    env.close.assert_called_once()
    # Function receives no replay, model, training environment or mutable counters.
    assert tracker.log.call_args.args[1] == 48


def test_disabled_has_no_sdk_dependency(tmp_path):
    with patch.dict('sys.modules', {'wandb': None}):
        tracker = Tracker(MBPORunConfig(), tmp_path)
        tracker.log({}, 0)
        tracker.finish()
    assert tracker.run is None


def test_diagnostics_do_not_change_rollouts():
    class Policy:
        def act(self, obs, deterministic=False):
            return np.zeros((len(obs), 2))
    class Model:
        obs_dim = 3
        elites = [0,1]
        def predict(self, inputs):
            return np.ones((2,len(inputs),4)), np.ones((2,len(inputs),4))
    outputs = []
    diagnostics = RolloutDiagnostics()
    for collector in (None, diagnostics):
        real = RealReplayBuffer(8,3,2,0)
        real.add(np.zeros(3), np.zeros(2), 0., np.ones(3), False, False)
        synthetic = ModelReplayBuffer(8,3,2,0)
        rng = np.random.default_rng(42)
        count = generate_rollouts(Policy(), Model(), real, synthetic, 4,2,rng, diagnostics=collector)
        outputs.append((synthetic.state_dict()['arrays'], rng.random()))
        assert count == 8
    assert outputs[0][1] == outputs[1][1]
    for key in outputs[0][0]:
        assert torch.equal(outputs[0][0][key], outputs[1][0][key])
    assert diagnostics.summary()['ensemble_disagreement_max'] == 0.
    assert diagnostics.summary()['synthetic_next_observation_finite_fraction'] == 1.


def test_diagnostics_values():
    collector = RolloutDiagnostics()
    collector.observe(np.zeros((2,2)), np.array([[3.,4.],[0.,0.]]), np.array([1.,3.]), np.array([[[0.,0.,0.],[0.,0.,0.]],[[2.,2.,0.],[2.,2.,0.]]]), [0,1])
    assert collector.summary()['synthetic_delta_norm_mean'] == 2.5
    assert collector.summary()['synthetic_reward_mean'] == 2.
    assert collector.summary()['ensemble_disagreement_mean'] == 1.
    model = SimpleNamespace(normalization={'input_mean': torch.tensor([float('nan')]), 'target_std': torch.ones(2)})
    assert normalization_checks(model) == {'input_mean_finite': False, 'target_std_finite': True}


def test_video_error_is_explicit_and_restores_rng(tmp_path):
    env = MagicMock()
    env.spec.max_episode_steps = 1
    env.reset.return_value = (np.zeros(3), {})
    env.step.return_value = (np.zeros(3), 0., False, True, {})
    env.render.return_value = None
    learner = SimpleNamespace(device=torch.device('cpu'), act=lambda *a, **k: np.zeros(2))
    original = torch.get_rng_state().clone()
    with patch('dynamics_shift.algorithms.mbpo.monitoring.make_env', return_value=env), patch.dict('sys.modules', {'wandb': MagicMock()}):
        with pytest.raises(RuntimeError, match='render_mode=rgb_array'):
            evaluate_nominal(learner, MBPORunConfig(), MagicMock(), tmp_path, 48, record_video=True)
    assert torch.equal(original, torch.get_rng_state())
    env.close.assert_called_once()
