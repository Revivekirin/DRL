"""Learner checkpoints, not exact environment/replay resume snapshots."""
from pathlib import Path
import random
import numpy as np
import torch
from dynamics_shift.algorithms.sac.learner import SACLearner


def capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))


def save_checkpoint(path: str | Path, learner: SACLearner, counters: dict[str, int],
                    config: dict) -> None:
    if counters["policy_gradient_steps"] != learner.policy_gradient_steps:
        raise ValueError("Learner and experiment update counters disagree")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "learner": learner.state_dict(), "counters": dict(counters),
               "config": config, "rng": capture_rng(), "replay_persisted": False,
               "simulator_persisted": False}
    # Exclusive creation: never silently replace an existing experiment artifact.
    with path.open("xb") as stream:
        torch.save(payload, stream)


def load_checkpoint(path: str | Path, restore_random_state: bool = False) -> tuple[SACLearner, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint schema")
    # Constructing networks must not perturb caller RNG during frozen evaluation.
    with torch.random.fork_rng(devices=[]):
        learner = SACLearner.from_state_dict(payload["learner"])
    if learner.policy_gradient_steps != payload["counters"]["policy_gradient_steps"]:
        raise ValueError("Checkpoint counter mismatch")
    if restore_random_state:
        restore_rng(payload["rng"])
    return learner, payload
