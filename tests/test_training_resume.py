"""Run on the remote server: these tests invoke short learner updates."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import pytest
from dynamics_shift.experiments.config import load_run_config
from dynamics_shift.experiments.train_sac_source import train_source
from dynamics_shift.utils.checkpoint import load_checkpoint, save_checkpoint
from test_sac_checkpoint import assert_nested_equal


@pytest.mark.training
def test_split_training_matches_uninterrupted(tmp_path):
    config = load_run_config(Path(__file__).parents[1] / "tests/fixtures/sac_smoke.yaml")
    config = replace(config, training=replace(config.training, device="cpu"))
    full = train_source(config, tmp_path, show_progress=False)
    partial = train_source(replace(config, training=replace(config.training, real_env_steps=32)),
                           tmp_path, show_progress=False)
    continued = train_source(config, tmp_path, show_progress=False,
                             resume=partial / "checkpoints/latest.pt")
    a, pa = load_checkpoint(full / "checkpoints/final.pt")
    b, pb = load_checkpoint(continued / "checkpoints/final.pt")
    assert_nested_equal(a.state_dict(), b.state_dict())
    assert_nested_equal(pa["training_state"], pb["training_state"])
    assert pa["counters"] == pb["counters"]
    latest = continued / "checkpoints/latest.pt"
    before = latest.read_bytes()
    with patch("torch.save", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            save_checkpoint(latest, b, pb["counters"], pb["config"], overwrite=True)
    assert latest.read_bytes() == before
