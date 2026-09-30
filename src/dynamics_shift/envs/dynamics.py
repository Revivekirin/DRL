"""The only module allowed to change MuJoCo model parameters."""
import gymnasium as gym
import mujoco
import numpy as np
from numpy.typing import NDArray
from dynamics_shift.config import validate_scale


class DynamicsController(gym.Wrapper):
    """Wrap a fresh nominal HalfCheetah; all shifts use its original gear.

    This wrapper leaves observations, rewards, actions and episode limits alone.
    Use make_env() for configured construction. Never wrap a shifted model twice.
    """

    def __init__(self, env: gym.Env) -> None:
        current = env
        while isinstance(current, gym.Wrapper):
            if isinstance(current, DynamicsController):
                raise ValueError("Environment already has a dynamics controller")
            current = current.env
        super().__init__(env)
        if env.spec is None or env.spec.id != "HalfCheetah-v5":
            raise ValueError("Only stock HalfCheetah-v5 is supported")
        model = self.unwrapped.model
        if not (np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                and np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
                and np.all(model.actuator_gainprm[:, 0] == 1)
                and np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE)
                and np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
                and not np.any(model.actuator_forcelimited)):
            raise ValueError("Expected direct, unit-gain joint motors without force limits")
        # Bytes-backed array cannot be made writeable, including through a view.
        self._nominal_gear = np.frombuffer(model.actuator_gear.tobytes(), dtype=model.actuator_gear.dtype).reshape(model.actuator_gear.shape)
        self._actuator_scale = 1.0

    @property
    def nominal_actuator_gear(self) -> NDArray[np.float64]:
        """Return a defensive copy; modifying it cannot change the baseline."""
        return self._nominal_gear.copy()

    def get_parameters(self) -> dict[str, float]:
        return {"actuator_scale": self._actuator_scale}

    def get_shift_parameter(self, parameter: str) -> float:
        if parameter != "actuator_strength":
            raise ValueError(f"Unsupported dynamics parameter: {parameter}")
        if not np.array_equal(self.unwrapped.model.actuator_gear,
                              self._nominal_gear * self._actuator_scale):
            raise ValueError("MuJoCo actuator gear disagrees with its tracked scale")
        return self._actuator_scale

    def apply_dynamics_shift(self, parameter: str, value: float) -> None:
        if parameter != "actuator_strength":
            raise ValueError(f"Unsupported dynamics parameter: {parameter}")
        self.set_actuator_scale(value)

    def set_actuator_scale(self, scale: float) -> None:
        scale = validate_scale(scale)
        model, data = self.unwrapped.model, self.unwrapped.data
        gear = self._nominal_gear * scale
        if not np.all(np.isfinite(gear)):
            raise ValueError("Scaled actuator gear must be finite")
        model.actuator_gear[:] = gear
        self._actuator_scale = scale
        # Refresh derived fields without advancing time or changing warmstart.
        warmstart = data.qacc_warmstart.copy()
        mujoco.mj_forward(model, data)
        data.qacc_warmstart[:] = warmstart

    def reset_to_nominal(self) -> None:
        self.set_actuator_scale(1.0)
