import numpy as np
import pytest
from dynamics_shift.config import ExperimentConfig, DynamicsConfig
from dynamics_shift.envs import make_env
from dynamics_shift.envs.mujoco_state import capture_state, restore_state


@pytest.mark.parametrize("scale", [1.0, 0.7])
def test_roundtrip_nonzero_state_and_horizon(scale):
    env = make_env(ExperimentConfig(dynamics=DynamicsConfig(scale)))
    try:
        with pytest.raises(ValueError):
            capture_state(env)
        env.reset(seed=42)
        rng = np.random.default_rng(5)
        for action in rng.uniform(-0.3, 0.3, (990, 6)):
            env.step(action)
        snapshot = capture_state(env)
        actions = rng.uniform(-0.5, 0.5, (10, 6))
        first = [env.step(action) for action in actions]
        sample = env.action_space.sample()
        restore_state(env, snapshot)
        np.testing.assert_array_equal(capture_state(env).integration_state, snapshot.integration_state)
        second = [env.step(action) for action in actions]
        np.testing.assert_array_equal(env.action_space.sample(), sample)
        for a, b in zip(first, second):
            np.testing.assert_allclose(a[0], b[0], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(a[1], b[1], rtol=1e-12, atol=1e-12)
            assert a[2:4] == b[2:4]
        assert first[-1][3]
    finally:
        env.close()


def test_same_initial_state_shift_changes_trajectory():
    env, other = make_env(ExperimentConfig()), make_env(ExperimentConfig())
    try:
        env.reset(seed=0)
        snapshot = capture_state(env)
        action = np.full(6, 0.5)
        nominal = env.step(action)[0]
        restore_state(env, snapshot)
        env.set_actuator_scale(0.7)
        np.testing.assert_array_equal(capture_state(env).integration_state, snapshot.integration_state)
        assert np.linalg.norm(env.step(action)[0] - nominal) > 1e-6
        with pytest.raises(ValueError):
            restore_state(other, snapshot)
    finally:
        env.close()
        other.close()
