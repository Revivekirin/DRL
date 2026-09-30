"""Serializable continuation state for the stock HalfCheetah factory."""
from importlib.metadata import version
import torch
from dynamics_shift.envs.shift_controller import DynamicsShiftController
from dynamics_shift.envs.mujoco_state import capture_state, restore_state, SimulatorSnapshot


def capture_training_state(env, replay, obs, episode_return, episode_length, *,
                           shift_controller: DynamicsShiftController | None = None) -> dict:
    controller_state = None
    if shift_controller is not None:
        if shift_controller.env is not env:
            raise ValueError("Shift controller belongs to a different environment")
        controller_state = {"config": shift_controller.config_dict(),
                            "state": shift_controller.state_dict()}
    snapshot = capture_state(env)
    return {"shift_controller": controller_state, "versions": {key: version(key) for key in ("gymnasium", "mujoco", "numpy", "torch")},
            "replay": replay.state_dict(), "obs": torch.as_tensor(obs.copy()),
            "episode_return": episode_return, "episode_length": episode_length,
            "integration": torch.from_numpy(snapshot.integration_state.copy()),
            "elapsed_steps": snapshot.elapsed_steps, "rng_states": snapshot.rng_states}


def restore_training_state(env, replay, state: dict, *,
                           shift_controller: DynamicsShiftController | None = None):
    saved_controller = state.get("shift_controller")
    if (saved_controller is None) != (shift_controller is None):
        raise ValueError("Checkpoint and resume call must both supply a shift controller, or neither")
    if shift_controller is not None:
        if shift_controller.env is not env:
            raise ValueError("Shift controller belongs to a different environment")
        if saved_controller["config"] != shift_controller.config_dict():
            raise ValueError("Resume requires the same shift event configuration")
        shift_controller.validate_state_dict(saved_controller["state"])
    for name, saved in state["versions"].items():
        if version(name) != saved:
            raise ValueError(f"Resume requires matching {name}: saved={saved}, installed={version(name)}")
    replay.load_state_dict(state["replay"])
    template = capture_state(env)
    integration = state["integration"].numpy()
    if integration.shape != template.integration_state.shape:
        raise ValueError("Simulator integration state shape mismatch")
    # Set physical parameters before recomputing derived simulator quantities.
    if shift_controller is not None:
        shift_controller.load_state_dict(saved_controller["state"])
    restore_state(env, SimulatorSnapshot(env, integration, tuple(state["elapsed_steps"]),
                                         tuple(state["rng_states"])))
    return state["obs"].numpy().copy(), state["episode_return"], state["episode_length"]
