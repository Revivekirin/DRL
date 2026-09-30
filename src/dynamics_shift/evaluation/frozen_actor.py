"""Read only the actor into executable objects; never construct a SAC learner."""
from pathlib import Path
import numpy as np
import torch
from dynamics_shift.algorithms.sac.networks import GaussianActor


class FrozenActor:
    def __init__(self, state: dict, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.obs_dim = state["obs_dim"]
        self.action_low = np.asarray(state["action_low"], dtype=np.float32)
        self.action_high = np.asarray(state["action_high"], dtype=np.float32)
        with torch.random.fork_rng(devices=[]):
            self.actor = GaussianActor(self.obs_dim, self.action_low, self.action_high,
                                       tuple(state["config"]["hidden_dims"]))
        self.actor.load_state_dict(state["actor"], strict=True)
        self.actor.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if not deterministic:
            raise ValueError("Frozen calibration only supports deterministic actions")
        return self.actor.deterministic(torch.as_tensor(obs, dtype=torch.float32,
                                                       device=self.device)).cpu().numpy()


def load_frozen_actor(checkpoint: str | Path, device: str | torch.device) -> tuple[FrozenActor, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    config = payload["config"]
    if (config.get("seed") != 0 or config.get("env", {}).get("id") != "HalfCheetah-v5"
            or config.get("dynamics", {}).get("actuator_scale") != 1.0):
        raise ValueError("Require a nominal HalfCheetah-v5 source checkpoint trained with seed 0")
    actor = FrozenActor(payload["learner"], device)
    provenance = {"source_config": config, "source_counters": payload["counters"],
                  "source_versions": (payload.get("training_state") or {}).get("versions")}
    # Critics, optimizers and replay are deserialized data only, then released.
    return actor, provenance
