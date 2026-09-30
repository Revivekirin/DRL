"""Validate continuation settings without loading models or running updates."""
from copy import deepcopy
from dynamics_shift.algorithms.mbpo.config import MBPORunConfig


def validate_resume_config(saved_config: dict, config: MBPORunConfig, saved_steps: int) -> None:
    if saved_steps > config.training.real_env_steps:
        raise ValueError(
            f"Resume checkpoint has {saved_steps} real_env_steps, but requested total budget is "
            f"{config.training.real_env_steps}. The budget is a total, not additional steps. "
            "Use an earlier checkpoint or increase the total budget."
        )
    old = MBPORunConfig.from_dict(deepcopy(saved_config)).to_dict()
    new = config.to_dict()
    for settings in (old, new):
        # Output labels and monitoring do not change the training algorithm.
        for key in ("name", "tracking", "evaluation"):
            settings.pop(key, None)
        for key in ("real_env_steps", "device", "torch_threads", "log_every", "checkpoint_every"):
            settings["training"].pop(key, None)

    def differences(left, right, prefix=""):
        result = []
        for key in sorted(left.keys() | right.keys()):
            path = f"{prefix}.{key}" if prefix else key
            a, b = left.get(key), right.get(key)
            if isinstance(a, dict) and isinstance(b, dict):
                result.extend(differences(a, b, path))
            elif a != b:
                result.append(f"{path}: checkpoint={a!r}, requested={b!r}")
        return result

    changed = differences(old, new)
    if changed:
        raise ValueError("Resume training config differs: " + "; ".join(changed))
