"""One actuator-shift event driven by the caller's real-transition counter."""
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from dynamics_shift.config import validate_scale
from .interface import DynamicsInterface


@dataclass(frozen=True)
class AbruptShiftSpec:
    """Apply target dynamics before the transition following trigger_env_step."""

    trigger_env_step: int
    parameter: str = "actuator_strength"
    source: float = 1.0
    target: float = 0.7

    def __post_init__(self) -> None:
        if type(self.trigger_env_step) is not int or self.trigger_env_step < 0:
            raise ValueError("trigger_env_step must be a nonnegative integer")
        if self.parameter != "actuator_strength":
            raise ValueError(f"Unsupported dynamics parameter: {self.parameter}")
        validate_scale(self.source)
        validate_scale(self.target)


@dataclass(frozen=True)
class ShiftEvent:
    """An intervention that actually occurred, suitable for dataclasses.asdict."""

    env_step: int
    parameter: str
    old_value: float
    new_value: float


class DynamicsShiftController:
    """A single event; resets of the environment never reset its fired flag.

    Call maybe_shift(real_env_steps) before env.step, where real_env_steps is
    the number of completed transitions. Restore with the same event config.
    Applying dynamics while loading state restores physical parameters, but
    does not execute a new event: maybe_shift returns None after a fired load.
    """

    def __init__(self, env: DynamicsInterface, spec: AbruptShiftSpec) -> None:
        self.env = env
        self.spec = spec
        self._fired = False
        self.assert_consistent()

    def _expected_scale(self, fired: bool) -> float:
        return self.spec.target if fired else self.spec.source

    def assert_consistent(self) -> None:
        """Reject independent parameter mutations that invalidate event state."""
        scale = self._expected_scale(self._fired)
        if self.env.get_shift_parameter(self.spec.parameter) != scale:
            raise ValueError("Environment dynamics and shift-controller state disagree")

    def maybe_shift(self, real_env_steps: int) -> ShiftEvent | None:
        """Return an immutable event only when the intervention actually occurs.

        A due but unfired event also fires when the caller has passed the trigger;
        this supports restoring immediately before the event handling boundary.
        """
        if type(real_env_steps) is not int or real_env_steps < 0:
            raise ValueError("real_env_steps must be a nonnegative integer")
        self.assert_consistent()
        if self._fired or real_env_steps < self.spec.trigger_env_step:
            return None
        old_value = self.env.get_shift_parameter(self.spec.parameter)
        self.env.apply_dynamics_shift(self.spec.parameter, self.spec.target)
        self._fired = True
        self.assert_consistent()
        return ShiftEvent(real_env_steps, self.spec.parameter, old_value, self.spec.target)

    def state_dict(self) -> dict[str, bool]:
        self.assert_consistent()
        return {"fired": self._fired}

    def config_dict(self) -> dict:
        """Static event definition, separate from mutable state."""
        return asdict(self.spec)

    def validate_state_dict(self, state: Mapping[str, object]) -> bool:
        if not isinstance(state, Mapping) or set(state) != {"fired"} or type(state["fired"]) is not bool:
            raise ValueError("Controller state must contain exactly one boolean: fired")
        return state["fired"]

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore source/target parameter value, then restore the fired flag."""
        fired = self.validate_state_dict(state)
        scale = self._expected_scale(fired)
        # The environment owns how absolute parameter values are applied.
        self.env.apply_dynamics_shift(self.spec.parameter, scale)
        self._fired = fired
        self.assert_consistent()
