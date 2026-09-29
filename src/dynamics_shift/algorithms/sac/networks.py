"""Squashed Gaussian actor and two independent Q networks."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(sizes: tuple[int, ...]) -> nn.Sequential:
    layers = []
    for index, (left, right) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        low, high = np.asarray(action_low, dtype=np.float32), np.asarray(action_high, dtype=np.float32)
        if low.ndim != 1 or low.shape != high.shape or not np.isfinite([low, high]).all() or np.any(high <= low):
            raise ValueError("Require finite one-dimensional action bounds with high > low")
        self.net = mlp((obs_dim, *hidden_dims, 2 * low.size))
        self.register_buffer("action_scale", torch.as_tensor((high - low) / 2))
        self.register_buffer("action_bias", torch.as_tensor((high + low) / 2))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        return mean, log_std.clamp(-20, 2)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return mean.tanh() * self.action_scale + self.action_bias

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        action = pre_tanh.tanh() * self.action_scale + self.action_bias
        # Stable log(1 - tanh(x)^2), including saturated tails, plus affine Jacobian.
        log_jacobian = 2 * (math.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))
        log_prob = (distribution.log_prob(pre_tanh) - log_jacobian
                    - self.action_scale.log()).sum(dim=-1, keepdim=True)
        return action, log_prob


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.q1 = mlp((obs_dim + action_dim, *hidden_dims, 1))
        self.q2 = mlp((obs_dim + action_dim, *hidden_dims, 1))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((obs, action), dim=-1)
        return self.q1(inputs), self.q2(inputs)
