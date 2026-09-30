"""REMOTE ONLY: model fitting, shared SAC updates and short nominal smoke run."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import torch
from dynamics_shift.algorithms.mbpo.config import MBPOConfig, load_mbpo_config
from dynamics_shift.algorithms.mbpo.rollouts import mixed_batch, generate_rollouts
from dynamics_shift.algorithms.mbpo.checkpoint import load_mbpo_checkpoint, load_frozen_model
from dynamics_shift.algorithms.sac import SACLearner, SACConfig
from dynamics_shift.data.model_data import ModelDataset, ModelReplayBuffer
from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble
from dynamics_shift.experiments.train_mbpo_source import train_mbpo_source
from test_mbpo_inference import real_data


def test_model_fit_normalization_checkpoint_and_shared_update(tmp_path):
    torch.set_num_threads(1)
    real = real_data()
    config = MBPOConfig(ensemble_size=2, elite_size=1, model_hidden_dims=(8,), model_max_epochs=2,
                        model_batch_size=8, model_patience=2)
    model = ProbabilisticEnsemble(3, 2, config)
    dataset = ModelDataset.from_real_replay(real, np.random.default_rng(1), .2, 12)
    before = [p.clone() for p in model.members.parameters()]
    metrics = model.train(dataset)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.members.parameters()))
    assert np.isfinite(metrics["model_train_loss"]).all()
    assert np.isfinite(metrics["model_validation_loss"]).all()
    assert model.elites == [int(np.argmin(metrics["model_validation_loss"]))]
    mean, var = model.predict(dataset.inputs)
    assert mean.shape == var.shape == (2, 12, 4)
    assert (var > 0).all()
    norm = {k: v.clone() for k, v in model.normalization.items()}
    model.validation_metrics(dataset.inputs, dataset.targets)
    model.disagreement(dataset.inputs)
    assert all(torch.equal(v, model.normalization[k]) for k, v in norm.items())
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    loaded = ProbabilisticEnsemble.from_state_dict(torch.load(path, weights_only=True))
    np.testing.assert_array_equal(loaded.predict(dataset.inputs)[0], mean)
    np.testing.assert_array_equal(loaded.predict(dataset.inputs)[1], var)
    assert all(torch.equal(v, loaded.normalization[k]) for k, v in norm.items())
    model.train(dataset)
    assert all(torch.equal(v, model.normalization[k]) for k, v in norm.items())
    learner = SACLearner(3, -np.ones(2), np.ones(2), SACConfig(hidden_dims=(8,)))
    synthetic = ModelReplayBuffer(32, 3, 2, 2)
    generate_rollouts(learner, loaded, real, synthetic, 4, 1, np.random.default_rng(2))
    batch, _, _ = mixed_batch(real, synthetic, 8, .5, np.random.default_rng(3))
    assert all(np.isfinite(v) for v in learner.update(batch).values())


def test_nominal_mbpo_smoke_and_resume(tmp_path):
    config = load_mbpo_config(Path(__file__).parents[1] / "configs/experiment/mbpo_smoke.yaml")
    config = replace(config, training=replace(config.training, device="cpu"))
    with patch("dynamics_shift.envs.DynamicsShiftController.__init__", side_effect=AssertionError("No shift events")):
        run = train_mbpo_source(config, tmp_path, show_progress=False)
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["real_env_steps"] == 48
    assert metadata["policy_gradient_steps"] == 33
    assert metadata["dynamics_model_refit_count"] == 3
    assert metadata["synthetic_transition_count"] == 24
    learner, model, payload = load_mbpo_checkpoint(run / "checkpoints/final.pt")
    assert model.train_steps == metadata["dynamics_model_train_steps"] > 0
    assert payload["training_state"]["mbpo"]["data_sources"]["real"] == "real_source"
    frozen = load_frozen_model(run / "checkpoints/final.pt")
    assert not any(p.requires_grad for p in frozen.members.parameters())
    inputs = np.zeros((2, 23), np.float32)
    np.testing.assert_array_equal(frozen.predict(inputs)[0], model.predict(inputs)[0])
    extended = replace(config, training=replace(config.training, real_env_steps=49))
    resumed = train_mbpo_source(extended, tmp_path, resume=run / "checkpoints/latest.pt", show_progress=False)
    result = json.loads((resumed / "metadata.json").read_text())
    assert result["policy_gradient_steps"] == 34 and result["real_env_steps"] == 49
