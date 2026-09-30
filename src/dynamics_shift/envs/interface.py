"""Simulator-independent interface for scalar dynamics interventions."""
from typing import Protocol


class DynamicsInterface(Protocol):
    def get_shift_parameter(self, parameter: str) -> float:
        """Read the current value; unsupported parameters must raise ValueError."""
        ...

    def apply_dynamics_shift(self, parameter: str, value: float) -> None:
        """Set an absolute parameter value, never a cumulative multiplier."""
        ...
