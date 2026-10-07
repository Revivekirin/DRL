"""Simulator-independent interface for scalar dynamics interventions."""
from typing import Protocol


class DynamicsInterface(Protocol):
    def get_shift_parameter(self, parameter: str) -> float:
        """Read the current value; unsupported parameters must raise ValueError."""
        ...

    def apply_dynamics_shift(self, parameter: str, value: float) -> None:
        """Set an absolute parameter value, never a cumulative multiplier."""
        ...


def supported_shift_parameters(backend: str, env_id: str) -> tuple[str, ...]:
    """Expose only implemented simulator interventions, separately from nominal tasks."""
    if backend == 'mujoco' and env_id == 'HalfCheetah-v5':
        return ('actuator_strength',)
    if backend == 'maniskill':
        from .maniskill_tasks import state_task
        return state_task(env_id).supported_shifts
    raise ValueError(f'Unsupported backend/task: {backend}/{env_id}')
