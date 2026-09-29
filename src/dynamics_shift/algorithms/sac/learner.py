"""SAC loss/update core. Sampling and environment interaction live elsewhere."""
from copy import deepcopy
from dataclasses import asdict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dynamics_shift.data.replay_buffer import TransitionBatch
from .config import SACConfig
from .networks import GaussianActor, TwinQ


class SACLearner:
    def __init__(self, obs_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 config: SACConfig, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.config = config
        self.obs_dim = obs_dim
        self.action_low = np.asarray(action_low, dtype=np.float32).copy()
        self.action_high = np.asarray(action_high, dtype=np.float32).copy()
        self.actor = GaussianActor(obs_dim, self.action_low, self.action_high, config.hidden_dims).to(self.device)
        self.critic = TwinQ(obs_dim, len(self.action_low), config.hidden_dims).to(self.device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(config.initial_alpha), device=self.device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = (-float(len(self.action_low)) if config.target_entropy is None
                               else config.target_entropy)
        self.policy_gradient_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        tensor = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        if deterministic:
            action = self.actor.deterministic(tensor)
        else:
            action, _ = self.actor.sample(tensor)
        return action.cpu().numpy()

    @torch.no_grad()
    def bellman_target(self, reward: torch.Tensor, next_obs: torch.Tensor,
                       terminated: torch.Tensor) -> torch.Tensor:
        """TimeLimit truncation still bootstraps from the stored final observation."""
        action, log_prob = self.actor.sample(next_obs)
        q1, q2 = self.target_critic(next_obs, action)
        return reward + self.config.gamma * (1 - terminated) * (
            torch.minimum(q1, q2) - self.alpha * log_prob)

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        tensors = {name: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)
                   for name, value in vars(batch).items()}
        n = tensors["obs"].shape[0]
        shapes = dict(obs=(n, self.obs_dim), action=(n, len(self.action_low)), reward=(n, 1),
                      next_obs=(n, self.obs_dim), terminated=(n, 1), truncated=(n, 1))
        if n == 0 or any(tensors[k].shape != shape or not torch.isfinite(tensors[k]).all()
                         for k, shape in shapes.items()):
            raise ValueError("Batch arrays must be finite and have matching [batch, feature] shapes")
        if any(not torch.all((tensors[k] == 0) | (tensors[k] == 1)) for k in ("terminated", "truncated")):
            raise ValueError("Terminal flags must be binary")
        obs, action = tensors["obs"], tensors["action"]
        target = self.bellman_target(tensors["reward"], tensors["next_obs"], tensors["terminated"])
        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        try:
            sampled_action, log_prob = self.actor.sample(obs)
            q1, q2 = self.critic(obs, sampled_action)
            actor_loss = (self.alpha.detach() * log_prob - torch.minimum(q1, q2)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters(), strict=True):
                target_param.lerp_(param, self.config.tau)
        self.policy_gradient_steps += 1
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
                "alpha_loss": alpha_loss.item(), "alpha": self.alpha.item()}

    def state_dict(self) -> dict:
        return {"config": asdict(self.config), "obs_dim": self.obs_dim,
                "action_low": self.action_low.tolist(), "action_high": self.action_high.tolist(),
                "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().clone(),
                "policy_gradient_steps": self.policy_gradient_steps}

    @classmethod
    def from_state_dict(cls, state: dict, device: str | torch.device = "cpu") -> "SACLearner":
        learner = cls(state["obs_dim"], state["action_low"], state["action_high"], SACConfig(**state["config"]), device=device)
        for name in ("actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer", "alpha_optimizer"):
            getattr(learner, name).load_state_dict(state[name])
        with torch.no_grad():
            learner.log_alpha.copy_(state["log_alpha"])
        learner.policy_gradient_steps = state["policy_gradient_steps"]
        return learner
