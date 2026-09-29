"""Serializable continuation state for the stock HalfCheetah factory."""
from importlib.metadata import version
import torch
from dynamics_shift.envs.mujoco_state import capture_state, restore_state, SimulatorSnapshot


def capture_training_state(env, replay, obs, episode_return, episode_length) -> dict:
    snapshot = capture_state(env)
    return {"versions": {key: version(key) for key in ("gymnasium", "mujoco", "numpy", "torch")},
            "replay": replay.state_dict(), "obs": torch.as_tensor(obs.copy()),
            "episode_return": episode_return, "episode_length": episode_length,
            "integration": torch.from_numpy(snapshot.integration_state.copy()),
            "elapsed_steps": snapshot.elapsed_steps, "rng_states": snapshot.rng_states}


def restore_training_state(env, replay, state: dict):
    for name, saved in state["versions"].items():
        if version(name) != saved:
            raise ValueError(f"Resume requires matching {name}: saved={saved}, installed={version(name)}")
    replay.load_state_dict(state["replay"])
    template = capture_state(env)
    integration = state["integration"].numpy()
    if integration.shape != template.integration_state.shape:
        raise ValueError("Simulator integration state shape mismatch")
    restore_state(env, SimulatorSnapshot(env, integration, tuple(state["elapsed_steps"]),
                                         tuple(state["rng_states"])))
    return state["obs"].numpy().copy(), state["episode_return"], state["episode_length"]
