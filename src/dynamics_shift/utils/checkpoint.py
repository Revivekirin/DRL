"""Learner checkpoints, not exact environment/replay resume snapshots."""
from pathlib import Path
import random
import numpy as np
import torch
from dynamics_shift.algorithms.sac.learner import SACLearner


def capture_rng(device: torch.device) -> dict:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state: dict, device: torch.device) -> None:
    if state.get("torch_cuda") is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)
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
               "config": config, "rng": capture_rng(learner.device), "replay_persisted": False,
               "simulator_persisted": False}
    # Exclusive creation: never silently replace an existing experiment artifact.
    with path.open("xb") as stream:
        torch.save(payload, stream)


def load_checkpoint(path: str | Path, restore_random_state: bool = False, *,
                    device: str | torch.device = "cpu") -> tuple[SACLearner, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint schema")
    # Constructing networks must not perturb caller RNG during frozen evaluation.
    with torch.random.fork_rng(devices=[]):
        learner = SACLearner.from_state_dict(payload["learner"], device=device)
    if learner.policy_gradient_steps != payload["counters"]["policy_gradient_steps"]:
        raise ValueError("Checkpoint counter mismatch")
    if restore_random_state:
        restore_rng(payload["rng"], learner.device)
    return learner, payload
