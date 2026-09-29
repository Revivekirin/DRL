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
