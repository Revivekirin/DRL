import gymnasium as gym
import numpy as np
from dynamics_shift.config import ExperimentConfig, DynamicsConfig
from dynamics_shift.envs import make_env


def test_nominal_equivalence():
    ordinary, wrapped = gym.make("HalfCheetah-v5"), make_env(ExperimentConfig())
    try:
        a, _ = ordinary.reset(seed=123)
        b, _ = wrapped.reset(seed=123)
        np.testing.assert_array_equal(a, b)
        actions = np.random.default_rng(19).uniform(-1, 1, (100, 6))
        for action in actions:
            left, right = ordinary.step(action), wrapped.step(action)
            np.testing.assert_allclose(left[0], right[0], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(left[1], right[1], rtol=1e-12, atol=1e-12)
            assert left[2:4] == right[2:4]
    finally:
        ordinary.close()
        wrapped.close()


def test_contract_and_full_horizon():
    source = make_env(ExperimentConfig())
    target = make_env(ExperimentConfig(dynamics=DynamicsConfig(0.7)))
    try:
        assert source.observation_space == target.observation_space
        assert source.action_space == target.action_space
        assert source.observation_space.shape == (17,)
        assert source.action_space.shape == (6,)
        assert source.spec.max_episode_steps == target.spec.max_episode_steps == 1000
        source.reset(seed=0)
        target.reset(seed=0)
        for step in range(1000):
            action = np.full(6, 0.1)
            for env in (source, target):
                obs, reward, terminated, truncated, info = env.step(action)
                assert env.observation_space.contains(obs)
                assert np.isfinite(reward)
                np.testing.assert_allclose(reward, info["reward_forward"] + info["reward_ctrl"])
                np.testing.assert_allclose(info["reward_ctrl"], -0.1 * np.sum(action**2))
                assert not terminated
                assert truncated == (step == 999)
    finally:
        source.close()
        target.close()
