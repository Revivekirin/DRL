"""Bootstrapped Gaussian ensemble predicting [observation delta, reward]."""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dynamics_shift.algorithms.mbpo.config import MBPOConfig
from dynamics_shift.data.model_data import ModelDataset


class GaussianDynamics(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers = []
        widths = (input_dim, *hidden_dims)
        for a, b in zip(widths, widths[1:]):
            layers.extend((nn.Linear(a, b), nn.SiLU()))
        layers.append(nn.Linear(widths[-1], 2 * output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw = self.net(inputs).chunk(2, -1)
        # Smooth fixed bounds avoid unbounded likelihood optimization.
        logvar = .5 - F.softplus(.5 - raw)
        logvar = -10. + F.softplus(logvar + 10.)
        return mean, logvar


def gaussian_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL excluding the constant log(2*pi)/2."""
    return .5 * ((mean - target).square() * (-logvar).exp() + logvar).mean()


class ProbabilisticEnsemble:
    """Model training is explicit; prediction never updates normalization.

    Normalizers are fitted once on the first real training partition (not its
    holdout) and remain fixed for later nominal refits and frozen diagnostics.
    Predictions are returned in physical observation/reward coordinates.
    """
    def __init__(self, obs_dim: int, action_dim: int, config: MBPOConfig,
                 device: str | torch.device = "cpu", seed: int = 0, *,
                 observation_layout=None, preserve_start: bool = False, observation_codec=None) -> None:
        from dynamics_shift.models.quaternion import QuaternionDelta
        self.observation_codec = observation_codec
        if observation_codec is not None:
            if (observation_codec.get('version') != 'maniskill_state_v1'
                    or observation_codec.get('target') != 'aligned_quaternion_delta_v1'
                    or observation_codec.get('invariant_policy') != 'learned_delta_with_drift_diagnostic'
                    or observation_codec.get('boolean_policy') != 'continuous_prediction_without_thresholding'
                    or observation_codec.get('reward_policy') != 'learned_gaussian_without_clipping'
                    or observation_codec['layout'] != observation_layout):
                raise ValueError('Unsupported or inconsistent observation/model codec')
        poses = observation_codec['quaternion_poses'] if observation_codec is not None else None
        self.geometry = QuaternionDelta(observation_layout, poses) if observation_layout else None
        self.preserve_start = preserve_start
        self.obs_dim, self.action_dim, self.config = obs_dim, action_dim, config
        self.device = torch.device(device)
        self.members = nn.ModuleList([GaussianDynamics(obs_dim + action_dim, obs_dim + 1,
                         config.model_hidden_dims) for _ in range(config.ensemble_size)]).to(self.device)
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=config.model_learning_rate) for m in self.members]
        self.rng = np.random.default_rng(seed)
        self.normalization = None
        self.elites = list(range(config.elite_size))
        self.train_steps = 0  # Counts individual member optimizer steps.
        self.refit_count = 0
        self.last_metrics = {}

    def _tensor(self, array) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _normalized(self, inputs, targets=None):
        if self.normalization is None:
            raise RuntimeError("Dynamics model has not been fitted")
        n = self.normalization
        x = (self._tensor(inputs) - n["input_mean"]) / n["input_std"]
        y = None if targets is None else (self._tensor(targets) - n["target_mean"]) / n["target_std"]
        return x, y

    def prepare_dataset(self, dataset):
        return self.geometry.dataset(dataset, self.obs_dim) if self.geometry else dataset

    def reconstruct(self, obs, delta, normalize=True):
        return self.geometry.reconstruct(obs, delta, normalize) if self.geometry else obs + delta

    def train(self, dataset: ModelDataset) -> dict:
        if not isinstance(dataset, ModelDataset):
            raise TypeError("Expected a real-replay ModelDataset")
        if dataset.inputs.shape[1] != self.obs_dim + self.action_dim or dataset.targets.shape[1] != self.obs_dim + 1:
            raise ValueError("Dynamics dataset dimensions differ")
        if not np.isfinite(dataset.inputs).all() or not np.isfinite(dataset.targets).all():
            raise ValueError("Nonfinite model data")
        dataset = self.prepare_dataset(dataset)
        ti, vi = dataset.train_indices, dataset.holdout_indices
        if len(ti) < 2 or len(vi) < 1 or np.intersect1d(ti, vi).size:
            raise ValueError("Require disjoint nonempty train/holdout partitions")
        if self.normalization is None:
            self.normalization = {}
            for prefix, data in (("input", dataset.inputs[ti]), ("target", dataset.targets[ti])):
                self.normalization[prefix + "_mean"] = self._tensor(data.mean(0))
                self.normalization[prefix + "_std"] = self._tensor(np.maximum(data.std(0), 1e-6))
        x, y = self._normalized(dataset.inputs, dataset.targets)
        starting_nll, kept_start = [], []
        for member, optimizer in zip(self.members, self.optimizers):
            member.eval()
            with torch.no_grad():
                start_loss = float(gaussian_loss(*member(x[vi]), y[vi]))
            if not np.isfinite(start_loss):
                raise FloatingPointError("Nonfinite initial validation NLL")
            starting_nll.append(start_loss)
            best_loss, stale, best = float("inf"), 0, None
            is_start = False
            if self.preserve_start:
                best_loss, best = start_loss, (deepcopy(member.state_dict()), deepcopy(optimizer.state_dict()))
                is_start = True
            bootstrap = self.rng.choice(ti, size=len(ti), replace=True)
            for _ in range(self.config.model_max_epochs):
                member.train()
                for indices in np.array_split(self.rng.permutation(bootstrap),
                                              max(1, int(np.ceil(len(ti) / self.config.model_batch_size)))):
                    ids = torch.as_tensor(indices, device=self.device)
                    loss = gaussian_loss(*member(x[ids]), y[ids])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite dynamics loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(member.parameters(), 100., error_if_nonfinite=True)
                    optimizer.step()
                    self.train_steps += 1
                member.eval()
                with torch.no_grad():
                    ids = torch.as_tensor(vi, device=self.device)
                    val = float(gaussian_loss(*member(x[ids]), y[ids]))
                if not np.isfinite(val):
                    raise FloatingPointError("Nonfinite dynamics validation loss")
                if val < best_loss:
                    best_loss, stale = val, 0
                    is_start = False
                    best = (deepcopy(member.state_dict()), deepcopy(optimizer.state_dict()))
                else:
                    stale += 1
                if stale >= self.config.model_patience:
                    break
            member.load_state_dict(best[0])
            optimizer.load_state_dict(best[1])
            kept_start.append(is_start)
        self.refit_count += 1
        metrics = self.validation_metrics(dataset.inputs[vi], dataset.targets[vi])
        self.elites = np.argsort(metrics["member_nll"])[:self.config.elite_size].tolist()
        train_metrics = self.validation_metrics(dataset.inputs[ti], dataset.targets[ti])
        self.last_metrics = {"fit_start_validation_nll": starting_nll, "restored_fit_start": kept_start,
                             "model_train_loss": train_metrics["member_nll"],
                             "model_validation_loss": metrics["member_nll"],
                             "model_validation_rmse": metrics["member_rmse"],
                             "train_samples": len(ti), "holdout_samples": len(vi), "elites": self.elites.copy()}
        return deepcopy(self.last_metrics)

    @torch.no_grad()
    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.geometry:
            inputs = inputs.copy()
            inputs[:, :self.obs_dim] = self.geometry.canonical(inputs[:, :self.obs_dim])
        x, _ = self._normalized(inputs)
        means, variances = [], []
        for member in self.members:
            member.eval()
            mean, logvar = member(x)
            means.append(mean * self.normalization["target_std"] + self.normalization["target_mean"])
            variances.append(logvar.exp() * self.normalization["target_std"].square())
        return torch.stack(means).cpu().numpy(), torch.stack(variances).cpu().numpy()

    @torch.no_grad()
    def validation_metrics(self, inputs: np.ndarray, targets: np.ndarray) -> dict:
        x, y = self._normalized(inputs, targets)
        losses, errors = [], []
        for member in self.members:
            nll_sum, mse_sum = 0., 0.
            for start in range(0, len(x), self.config.model_batch_size):
                stop = start + self.config.model_batch_size
                mean, logvar = member(x[start:stop])
                count = len(mean)
                nll_sum += float(gaussian_loss(mean, logvar, y[start:stop])) * count
                mse_sum += float(((mean - y[start:stop]) * self.normalization["target_std"]).square().mean()) * count
            losses.append(nll_sum / len(x))
            errors.append(float(np.sqrt(mse_sum / len(x))))
        return {"member_nll": losses, "member_rmse": errors}

    def disagreement(self, inputs: np.ndarray) -> np.ndarray:
        means, _ = self.predict(inputs)
        # Per-transition mean variance of elite predicted deltas; excludes reward.
        return means[self.elites, :, :self.obs_dim].var(axis=0).mean(axis=-1)

    def state_dict(self) -> dict:
        return {"obs_dim": self.obs_dim, "action_dim": self.action_dim, "config": asdict(self.config),
                "observation_codec": self.observation_codec,
                "geometry": {"version": self.geometry.version, "layout": self.geometry.layout} if self.geometry else None,
                "preserve_start": self.preserve_start,
                "members": self.members.state_dict(), "optimizers": [o.state_dict() for o in self.optimizers],
                "normalization": self.normalization, "elites": self.elites, "rng": self.rng.bit_generator.state,
                "train_steps": self.train_steps, "refit_count": self.refit_count, "last_metrics": self.last_metrics}

    @classmethod
    def from_state_dict(cls, state: dict, device: str | torch.device = "cpu", *, expected_codec=None) -> "ProbabilisticEnsemble":
        if expected_codec is not None and state.get("observation_codec") != expected_codec:
            raise ValueError("Checkpoint observation/model codec mismatch or missing legacy metadata")
        geometry = state.get("geometry")
        if geometry and geometry["version"] != "aligned_quaternion_delta_v1":
            raise ValueError("Unsupported dynamics geometry version")
        with torch.random.fork_rng(devices=[]):
            model = cls(state["obs_dim"], state["action_dim"], MBPOConfig(**state["config"]), device,
                        observation_layout=geometry["layout"] if geometry else None,
                        observation_codec=state.get("observation_codec"),
                        preserve_start=state.get("preserve_start", False))
        model.members.load_state_dict(state["members"])
        for optimizer, saved in zip(model.optimizers, state["optimizers"], strict=True):
            optimizer.load_state_dict(saved)
        model.normalization = None if state["normalization"] is None else {
            k: v.to(model.device) for k, v in state["normalization"].items()}
        model.elites = list(state["elites"])
        model.rng.bit_generator.state = state["rng"]
        model.train_steps, model.refit_count = state["train_steps"], state["refit_count"]
        model.last_metrics = state["last_metrics"]
        model.members.eval()
        return model
