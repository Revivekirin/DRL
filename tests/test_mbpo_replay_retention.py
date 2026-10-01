"""FIFO rollout retention using fake predictions; no training or updates."""
import numpy as np
import torch
from dynamics_shift.algorithms.mbpo.config import MBPOConfig, load_mbpo_config
from dynamics_shift.algorithms.mbpo.rollouts import generate_rollouts, mixed_batch
from dynamics_shift.data.model_data import RealReplayBuffer, ModelReplayBuffer


class Policy:
    def act(self, obs, deterministic=False):
        return np.zeros((len(obs), 1), dtype=np.float32)


class Model:
    obs_dim = 2
    elites = [0]
    generation = 0

    def predict(self, inputs):
        mean = np.zeros((1, len(inputs), 3), dtype=np.float32)
        mean[:, :, -1] = self.generation
        return mean, np.zeros_like(mean)


def test_four_generations_retained_then_oldest_evicted_and_restored():
    real = RealReplayBuffer(8, 2, 1, 0)
    real.add(np.zeros(2), np.zeros(1), 0., np.ones(2), False, False)
    model = Model()
    synthetic = ModelReplayBuffer(12, 2, 1, 1)
    rng = np.random.default_rng(2)
    for generation in range(1, 6):
        model.generation = generation
        assert generate_rollouts(Policy(), model, real, synthetic, 3, 1, rng) == 3
        assert len(synthetic) == min(3 * generation, 12)
        rewards = synthetic.state_dict()['arrays']['reward'].numpy().ravel()
        assert set(rewards) == set(range(max(1, generation - 3), generation + 1))
    restored = ModelReplayBuffer(12, 2, 1, 99)
    restored.load_state_dict(synthetic.state_dict())
    assert restored.position == synthetic.position
    for buffer in (synthetic, restored):
        model.generation = 6
        generate_rollouts(Policy(), model, real, buffer, 3, 1, np.random.default_rng(5))
    for key, values in synthetic.state_dict()['arrays'].items():
        assert torch.equal(values, restored.state_dict()['arrays'][key])
    np.testing.assert_array_equal(synthetic.sample(20).reward, restored.sample(20).reward)
    _, nr, ns = mixed_batch(real, synthetic, 256, .05, np.random.default_rng(3))
    assert (nr, ns) == (12, 244)
    assert len(real) == 1


def test_fresh_diagnostic_uses_retained_pool_settings():
    config = load_mbpo_config('configs/experiment/mbpo_replay_20k.yaml')
    assert config.seed == 0 and config.training.real_env_steps == 20000
    assert config.mbpo == MBPOConfig()
    assert config.mbpo.rollout_batch_size == 100000
    assert config.mbpo.model_replay_capacity == 400000
    assert config.mbpo.rollout_horizon == 1
    assert config.mbpo.model_train_frequency == 250
    assert config.training.updates_per_env_step == 20
    assert config.algo.target_entropy is None
    assert config.tracking.video_every == 10000
