import gymnasium as gym
import mujoco
import numpy as np
import pytest
from dynamics_shift.config import ExperimentConfig
from dynamics_shift.envs import make_env
from dynamics_shift.envs.dynamics import DynamicsController


@pytest.fixture
def env():
    instance = make_env(ExperimentConfig())
    instance.reset(seed=0)
    yield instance
    instance.close()


def test_scaling_non_cumulative_reset_and_defensive_copy(env):
    nominal = env.nominal_actuator_gear
    exposed = env.nominal_actuator_gear
    exposed[:] = 0
    np.testing.assert_array_equal(env.nominal_actuator_gear, nominal)
    for scale in (0.7, 0.7, 0.5, 0.7):
        env.set_actuator_scale(scale)
        np.testing.assert_array_equal(env.unwrapped.model.actuator_gear, scale * nominal)
        assert env.get_parameters() == {"actuator_scale": scale}
    env.reset_to_nominal()
    np.testing.assert_array_equal(env.unwrapped.model.actuator_gear, nominal)
    assert env.get_parameters() == {"actuator_scale": 1.0}
    with pytest.raises(ValueError):
        DynamicsController(env)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True, "0.7"])
def test_invalid_scale_is_atomic(env, scale):
    before = env.unwrapped.model.actuator_gear.copy()
    with pytest.raises(ValueError):
        env.set_actuator_scale(scale)
    np.testing.assert_array_equal(env.unwrapped.model.actuator_gear, before)


def test_effective_joint_torque(env):
    base = env.unwrapped
    base.data.ctrl[:] = np.linspace(-0.8, 0.8, base.model.nu)
    mujoco.mj_forward(base.model, base.data)
    nominal_force = base.data.qfrc_actuator.copy()
    ctrlrange = base.model.actuator_ctrlrange.copy()
    mass = base.model.body_mass.copy()
    damping = base.model.dof_damping.copy()
    env.set_actuator_scale(0.7)
    np.testing.assert_allclose(base.data.qfrc_actuator, 0.7 * nominal_force, atol=1e-12)
    np.testing.assert_array_equal(base.model.actuator_ctrlrange, ctrlrange)
    np.testing.assert_array_equal(base.model.body_mass, mass)
    np.testing.assert_array_equal(base.model.dof_damping, damping)
