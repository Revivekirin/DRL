"""MBPO state inside the existing atomic SAC checkpoint container."""
from pathlib import Path
import torch
from dynamics_shift.utils.checkpoint import save_checkpoint, load_checkpoint
from dynamics_shift.utils.training_state import capture_training_state
from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble


def save_mbpo_checkpoint(path, learner, model, real, synthetic, env, obs,
                         episode_return, episode_length, counters, config, rng, *, overwrite=False) -> None:
    if counters["dynamics_model_train_steps"] != model.train_steps or counters["dynamics_model_refit_count"] != model.refit_count:
        raise ValueError("Dynamics model counters disagree")
    state = capture_training_state(env, real, obs, episode_return, episode_length)
    state["mbpo"] = {"model": model.state_dict(), "synthetic_replay": synthetic.state_dict(),
                     "runner_rng": rng.bit_generator.state,
                     "data_sources": {"real": real.source, "synthetic": synthetic.source}}
    save_checkpoint(path, learner, counters, config.to_dict(), training_state=state, overwrite=overwrite)


def load_mbpo_checkpoint(path, device="cpu"):
    learner, payload = load_checkpoint(path, device=device)
    state = (payload.get("training_state") or {}).get("mbpo")
    if state is None:
        raise ValueError("Checkpoint does not contain MBPO dynamics state")
    model = ProbabilisticEnsemble.from_state_dict(state["model"], device)
    return learner, model, payload


def load_frozen_model(path: str | Path, device="cpu") -> ProbabilisticEnsemble:
    """Load source normalization and model without creating a policy learner."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = (payload.get("training_state") or {}).get("mbpo")
    if state is None:
        raise ValueError("Not an MBPO checkpoint")
    model = ProbabilisticEnsemble.from_state_dict(state["model"], device)
    model.members.requires_grad_(False)
    return model
