import csv
from dataclasses import replace
import json
from pathlib import Path
import pytest
from dynamics_shift.experiments.config import load_run_config, RunConfig
from dynamics_shift.experiments.train_sac_source import train_source
from dynamics_shift.utils.checkpoint import load_checkpoint


def test_cpu_smoke_training(tmp_path):
    config = load_run_config(Path(__file__).parents[1] / "configs/experiment/sac_smoke.yaml")
    config = replace(config, training=replace(config.training, device="cpu"))
    run = train_source(config, tmp_path)
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["real_env_steps"] == 64
    assert metadata["policy_gradient_steps"] == 49
    learner, payload = load_checkpoint(run / "checkpoints/final.pt")
    assert learner.policy_gradient_steps == 49
    assert RunConfig.from_dict(payload["config"]) == config
    with (run / "metrics/train.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[-1]["real_env_steps"] == "64"
    with (run / "metrics/frozen_shift/frozen_shift_eval.csv").open() as stream:
        episodes = list(csv.DictReader(stream))
    assert len(episodes) == 4
    # A second tiny run uses another directory rather than overwriting artifacts.
    shorter = replace(config, training=replace(config.training, real_env_steps=1),
                      evaluation=replace(config.evaluation, seeds=(100,)))
    assert train_source(shorter, tmp_path) != run


def test_config_rejects_non_nominal_source_and_unknown_keys():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"dynamics": {"actuator_scale": 0.7}})
    with pytest.raises(ValueError):
        RunConfig.from_dict({"training": {"steps": 100}})
    with pytest.raises(ValueError):
        RunConfig.from_dict({"training": {"batch_size": 10, "replay_capacity": 1}})
