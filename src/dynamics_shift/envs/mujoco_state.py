"""In-memory snapshots for replay in the same initialized environment.

Captures MuJoCo mjSTATE_INTEGRATION, TimeLimit elapsed steps, and environment
and space RNG states. Model parameters are deliberately excluded, allowing
replay under a different actuator scale. This is not a portable checkpoint.
"""
from copy import deepcopy
from dataclasses import dataclass
import gymnasium as gym
import mujoco
import numpy as np
from numpy.typing import NDArray

_STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def _wrappers(env: gym.Env) -> list[gym.Wrapper]:
    result = []
    while isinstance(env, gym.Wrapper):
        result.append(env)
        env = env.env
    return result


@dataclass(frozen=True)
class SimulatorSnapshot:
    owner: gym.Env
    integration_state: NDArray[np.float64]
    elapsed_steps: tuple[int, ...]
    rng_states: tuple[dict, ...]


def capture_state(env: gym.Env) -> SimulatorSnapshot:
    """Capture after reset; snapshots are bound to this exact env instance."""
    wrappers = _wrappers(env)
    limits = [w for w in wrappers if isinstance(w, gym.wrappers.TimeLimit)]
    if any(w._elapsed_steps is None for w in limits):
        raise ValueError("Reset the environment before capturing state")
    base = env.unwrapped
    state = np.empty(mujoco.mj_stateSize(base.model, _STATE))
    mujoco.mj_getState(base.model, base.data, state, _STATE)
    state = np.frombuffer(state.tobytes(), dtype=state.dtype)
    rngs = (base.np_random, env.action_space.np_random, env.observation_space.np_random)
    return SimulatorSnapshot(env, state, tuple(w._elapsed_steps for w in limits),
                             tuple(deepcopy(r.bit_generator.state) for r in rngs))


def restore_state(env: gym.Env, snapshot: SimulatorSnapshot) -> None:
    """Restore integration inputs and recompute derived fields at current dynamics."""
    if snapshot.owner is not env:
        raise ValueError("Snapshot belongs to a different environment instance")
    base = env.unwrapped
    mujoco.mj_setState(base.model, base.data, snapshot.integration_state, _STATE)
    mujoco.mj_forward(base.model, base.data)
    # mj_forward may replace solver warmstart; retain original integration inputs.
    mujoco.mj_setState(base.model, base.data, snapshot.integration_state, _STATE)
    limits = [w for w in _wrappers(env) if isinstance(w, gym.wrappers.TimeLimit)]
    for wrapper, elapsed in zip(limits, snapshot.elapsed_steps, strict=True):
        wrapper._elapsed_steps = elapsed
    rngs = (base.np_random, env.action_space.np_random, env.observation_space.np_random)
    for rng, state in zip(rngs, snapshot.rng_states, strict=True):
        rng.bit_generator.state = deepcopy(state)
