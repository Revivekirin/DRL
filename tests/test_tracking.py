"""W&B calls are mocked; no external data is sent by these tests."""
from dataclasses import replace
from unittest.mock import MagicMock, patch
from dynamics_shift.experiments.config import RunConfig, TrackingConfig
from dynamics_shift.utils.tracking import Tracker


def test_tracking_logs_environment_counter(tmp_path):
    config = replace(RunConfig(), tracking=TrackingConfig(mode="offline"))
    sdk = MagicMock()
    with patch.dict("sys.modules", {"wandb": sdk}):
        tracker = Tracker(config, tmp_path)
        tracker.log({"train/actor_loss": 1.5}, 42)
        tracker.finish()
    sdk.init.assert_called_once()
    sdk.init.return_value.log.assert_called_with({"real_env_steps": 42, "train/actor_loss": 1.5})
    sdk.init.return_value.finish.assert_called_with(exit_code=0)


def test_tracking_disabled(tmp_path):
    tracker = Tracker(RunConfig(), tmp_path)
    assert tracker.run is None
    tracker.log({}, 0)
    tracker.finish()


def test_tracking_failure_is_visible_and_nonfatal(tmp_path):
    config = replace(RunConfig(), tracking=TrackingConfig(mode='offline'))
    sdk = MagicMock()
    sdk.init.side_effect = RuntimeError('SECRET_MUST_NOT_BE_SAVED')
    with patch.dict('sys.modules', {'wandb': sdk}):
        tracker = Tracker(config, tmp_path)
        tracker.log({'train/alpha': 1.}, 2)
    assert tracker.run is None and tracker.errors == 1
    text = (tmp_path / 'tracking_errors.jsonl').read_text()
    assert 'RuntimeError' in text and 'SECRET_MUST_NOT_BE_SAVED' not in text


def test_all_metrics_share_axis_and_array_entries_are_explicit(tmp_path):
    config = replace(RunConfig(), tracking=TrackingConfig(mode='offline'))
    sdk = MagicMock()
    with patch.dict('sys.modules', {'wandb': sdk}):
        tracker = Tracker(config, tmp_path)
        tracker.scalars('model', {'state_rmse': [1., 2.], 'missing': None}, 100)
    sdk.init.return_value.define_metric.assert_any_call('*', step_metric='real_env_steps')
    sdk.init.return_value.log.assert_called_with({'real_env_steps': 100,
        'model/state_rmse/0': 1., 'model/state_rmse/1': 2.})


def test_reserved_video_seeds_rejected():
    import pytest
    with pytest.raises(ValueError, match='Reserved'):
        TrackingConfig(video_seed=29999, video_episodes=2)


def test_video_failure_does_not_interrupt_training(tmp_path):
    config = replace(RunConfig(), tracking=TrackingConfig(mode='offline', video_every=100))
    sdk = MagicMock()
    with patch.dict('sys.modules', {'wandb': sdk}), patch(
        'dynamics_shift.evaluation.video.record_checkpoint_videos', side_effect=RuntimeError('render')) as video:
        tracker = Tracker(config, tmp_path)
        tracker.checkpoint_video('unused.pt', config, 96)
        video.assert_not_called()
        tracker.checkpoint_video('unused.pt', config, 112)
        assert tracker.errors == 1 and tracker.next_video == 200
