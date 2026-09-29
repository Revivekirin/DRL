"""Validated parameters actually used by the SAC learner."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SACConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    initial_alpha: float = 0.2
    target_entropy: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if not self.hidden_dims or any(type(n) is not int or n <= 0 for n in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        for name in ("actor_lr", "critic_lr", "alpha_lr", "initial_alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("Require gamma in [0, 1] and tau in (0, 1]")
        if self.target_entropy is not None and not isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite or null")
