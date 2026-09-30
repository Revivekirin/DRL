"""No optimizer steps: network shapes, loss evaluation, replay and batch wiring."""
import numpy as np
import pytest
import torch
from dynamics_shift.algorithms.mbpo.config import MBPOConfig
from dynamics_shift.algorithms.mbpo.rollouts import mixed_batch, generate_rollouts
from dynamics_shift.data.model_data import RealReplayBuffer, ModelReplayBuffer, ModelDataset
from dynamics_shift.models.probabilistic_ensemble import GaussianDynamics, gaussian_loss


def real_data(n=12):
    real = RealReplayBuffer(32, 3, 2, 0)
    for i in range(n):
        real.add(np.full(3, i / 10), np.zeros(2), float(i), np.full(3, (i + 1) / 10), False, i == 11)
    return real


def test_gaussian_shapes_and_finite_loss():
    member = GaussianDynamics(5, 4, (8, 8))
    mean, logvar = member(torch.zeros(6, 5))
    assert mean.shape == logvar.shape == (6, 4)
    assert torch.isfinite(gaussian_loss(mean, logvar, torch.zeros(6, 4)))


def test_dataset_is_real_only_and_holdout_disjoint():
    real = real_data()
    dataset = ModelDataset.from_real_replay(real, np.random.default_rng(0), .2, 10)
    assert dataset.inputs.shape == (10, 5)
    assert dataset.targets.shape == (10, 4)
    np.testing.assert_allclose(dataset.targets[:, :3], .1, atol=1e-6)
    assert not np.intersect1d(dataset.train_indices, dataset.holdout_indices).size
    with pytest.raises(TypeError):
        ModelDataset.from_real_replay(ModelReplayBuffer(32, 3, 2, 0), np.random.default_rng(0), .2, 10)


def test_rollouts_mixing_and_buffer_separation():
    class Policy:
        def act(self, obs, deterministic=False):
            return np.zeros((len(obs), 2), np.float32)
    class Model:
        obs_dim = 3
        elites = [0]
        def predict(self, inputs):
            return np.full((1, len(inputs), 4), .1), np.zeros((1, len(inputs), 4))
    real = real_data()
    original = real.state_dict()["arrays"]["obs"].clone()
    synthetic = ModelReplayBuffer(32, 3, 2, 0)
    count = generate_rollouts(Policy(), Model(), real, synthetic, 4, 2, np.random.default_rng(4))
    assert count == len(synthetic) == 8
    assert len(real) == 12
    assert torch.equal(original, real.state_dict()["arrays"]["obs"])
    batch, nr, ns = mixed_batch(real, synthetic, 10, .3, np.random.default_rng(9))
    assert (nr, ns) == (3, 7)
    assert batch.obs.shape == batch.next_obs.shape == (10, 3)
    assert batch.action.shape == (10, 2)
    assert batch.reward.shape == batch.terminated.shape == batch.truncated.shape == (10, 1)
    assert not synthetic.sample(10).terminated.any()
    assert not synthetic.sample(10).truncated.any()
    synthetic.clear()
    assert len(synthetic) == 0 and len(real) == 12


def test_exact_shared_sac_class():
    from dynamics_shift.experiments.train_sac_source import SACLearner as PlainSAC
    from dynamics_shift.experiments.train_mbpo_source import SACLearner as MBPOSAC
    assert PlainSAC is MBPOSAC


@pytest.mark.parametrize("kwargs", [{"elite_size": 9}, {"real_ratio": 0}, {"holdout_ratio": 1}, {"rollout_horizon": 0}])
def test_invalid_mbpo_config(kwargs):
    with pytest.raises(ValueError):
        MBPOConfig(**kwargs)
