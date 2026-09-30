"""Fast environment/controller tests: no policies, updates or training runs."""
import io
import json
from unittest.mock import patch
import numpy as np
import pytest
import torch
from dynamics_shift.config import ExperimentConfig
from dynamics_shift.envs import make_env, DynamicsShiftController, AbruptShiftSpec, ShiftEvent
from dynamics_shift.data.replay_buffer import ReplayBuffer
from dynamics_shift.utils.training_state import capture_training_state, restore_training_state


@pytest.fixture
def env():
    instance = make_env(ExperimentConfig())
    instance.reset(seed=0)
    yield instance
    instance.close()


def test_before_trigger_state_roundtrip(env):
    controller = DynamicsShiftController(env, AbruptShiftSpec(10))
    assert not controller.maybe_shift(9)
    state = json.loads(json.dumps(controller.state_dict()))
    assert state == {"fired": False}
    # Loading a pre-shift state must restore source gear even after a later shift.
    assert controller.maybe_shift(10)
    controller.load_state_dict(state)
    assert env.get_shift_parameter("actuator_strength") == 1.0
    assert not controller.maybe_shift(9)
    assert controller.maybe_shift(10)
    assert not controller.maybe_shift(11)


def test_fired_restore_and_no_duplicate(env):
    controller = DynamicsShiftController(env, AbruptShiftSpec(0))
    assert controller.maybe_shift(0)
    state = controller.state_dict()
    env.reset_to_nominal()
    with pytest.raises(ValueError, match="disagree"):
        controller.state_dict()
    controller.load_state_dict(state)
    controller.load_state_dict(state)
    assert env.get_shift_parameter("actuator_strength") == .7
    with patch.object(env, "set_actuator_scale", wraps=env.set_actuator_scale) as setter:
        assert not controller.maybe_shift(0)
        assert not controller.maybe_shift(20)
        env.reset(seed=1)
        assert not controller.maybe_shift(30)
        setter.assert_not_called()


@pytest.mark.parametrize("state", [{}, {"fired": 1}, {"fired": "false"}, {"fired": True, "extra": 0}])
def test_invalid_state_does_not_mutate(env, state):
    controller = DynamicsShiftController(env, AbruptShiftSpec(1))
    with pytest.raises(ValueError):
        controller.load_state_dict(state)
    assert controller.state_dict() == {"fired": False}
    assert env.get_parameters()["actuator_scale"] == 1.0


@pytest.mark.parametrize("fired", [False, True])
def test_existing_checkpoint_hook_roundtrip(env, fired):
    event = AbruptShiftSpec(10)
    controller = DynamicsShiftController(env, event)
    if fired:
        controller.maybe_shift(10)
    replay = ReplayBuffer(4, 17, 6, 0)
    obs, _ = env.reset(seed=7)
    # Exercise the same weights-only serialization used by existing checkpoints.
    state = capture_training_state(env, replay, obs, 0., 0, shift_controller=controller)
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    fresh = make_env(ExperimentConfig())
    try:
        fresh.reset(seed=99)
        restored = DynamicsShiftController(fresh, event)
        new_replay = ReplayBuffer(4, 17, 6, 99)
        restored_obs, _, _ = restore_training_state(fresh, new_replay, state, shift_controller=restored)
        np.testing.assert_array_equal(obs, restored_obs)
        assert env.get_shift_parameter("actuator_strength") == fresh.get_shift_parameter("actuator_strength")
        action = np.full(6, .1)
        np.testing.assert_allclose(env.step(action)[0], fresh.step(action)[0], rtol=1e-12, atol=1e-12)
        assert not restored.maybe_shift(9)
        assert restored.maybe_shift(10) == (None if fired else ShiftEvent(10, "actuator_strength", 1.0, .7))
        assert not restored.maybe_shift(11)
        with pytest.raises(ValueError, match="both supply"):
            restore_training_state(fresh, new_replay, state)
    finally:
        fresh.close()


def test_mismatched_schedule_and_legacy_checkpoint(env):
    controller = DynamicsShiftController(env, AbruptShiftSpec(10))
    replay = ReplayBuffer(4, 17, 6, 0)
    obs, _ = env.reset(seed=0)
    state = capture_training_state(env, replay, obs, 0., 0, shift_controller=controller)
    other = DynamicsShiftController(env, AbruptShiftSpec(11))
    with pytest.raises(ValueError, match="same shift event"):
        restore_training_state(env, replay, state, shift_controller=other)
    legacy = capture_training_state(env, replay, obs, 0., 0)
    legacy.pop("shift_controller")
    restore_training_state(env, replay, legacy)
    with pytest.raises(ValueError, match="both supply"):
        restore_training_state(env, replay, legacy, shift_controller=controller)


def test_controller_uses_only_generic_interface():
    from dataclasses import FrozenInstanceError, asdict

    class FakeDynamics:
        __slots__ = ("value", "calls")

        def __init__(self):
            self.value, self.calls = 1.0, []

        def get_shift_parameter(self, parameter):
            assert parameter == "actuator_strength"
            return self.value

        def apply_dynamics_shift(self, parameter, value):
            assert parameter == "actuator_strength"
            self.calls.append(value)
            self.value = value

    env = FakeDynamics()  # No Gym wrapper, MuJoCo model or nominal gear array.
    controller = DynamicsShiftController(env, AbruptShiftSpec(10))
    assert controller.maybe_shift(9) is None
    event = controller.maybe_shift(12)
    assert event == ShiftEvent(12, "actuator_strength", 1.0, .7)
    assert asdict(event) == dict(env_step=12, parameter="actuator_strength", old_value=1., new_value=.7)
    with pytest.raises(FrozenInstanceError):
        event.env_step = 99
    assert controller.maybe_shift(12) is None
    assert controller.maybe_shift(20) is None
    assert env.calls == [.7]
    controller.load_state_dict({"fired": True})
    assert controller.maybe_shift(21) is None
    controller.load_state_dict({"fired": False})
    assert env.value == 1.0
    assert controller.maybe_shift(9) is None
    assert controller.maybe_shift(10) == ShiftEvent(10, "actuator_strength", 1., .7)


def test_unsupported_spec():
    with pytest.raises(ValueError, match="Unsupported"):
        AbruptShiftSpec(10, parameter="mass")
