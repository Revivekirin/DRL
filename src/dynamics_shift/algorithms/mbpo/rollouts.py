"""Short model rollouts and provenance-independent batches for shared SAC."""
import numpy as np
from dynamics_shift.data.replay_buffer import TransitionBatch
from dynamics_shift.data.model_data import RealReplayBuffer, ModelReplayBuffer
from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble
from dynamics_shift.algorithms.sac.learner import SACLearner


def generate_rollouts(learner: SACLearner, model: ProbabilisticEnsemble,
                      real: RealReplayBuffer, synthetic: ModelReplayBuffer,
                      batch_size: int, horizon: int, rng: np.random.Generator) -> int:
    """HalfCheetah-v5 has no physical terminal condition.

    Model horizon is a computation cutoff, not an environment TimeLimit; both
    flags are False. Bootstrapping is preserved. Do not reuse this termination
    rule unchanged for tasks with physical termination.
    """
    if not isinstance(real, RealReplayBuffer) or not isinstance(synthetic, ModelReplayBuffer):
        raise TypeError("Rollouts require separate typed real and model replay")
    if batch_size <= 0 or horizon <= 0:
        raise ValueError("Rollout batch and horizon must be positive")
    obs = real.sample(batch_size).obs
    generated = 0
    for _ in range(horizon):
        actions = learner.act(obs, deterministic=False)
        means, variances = model.predict(np.concatenate((obs, actions), axis=-1))
        # TS1: independently choose an elite for each transition at every depth.
        members = rng.choice(model.elites, size=batch_size)
        selected_mean = means[members, np.arange(batch_size)]
        selected_var = variances[members, np.arange(batch_size)]
        predictions = selected_mean + np.sqrt(selected_var) * rng.standard_normal(selected_mean.shape)
        next_obs = (obs + predictions[:, :model.obs_dim]).astype(np.float32)
        rewards = predictions[:, -1].astype(np.float32)
        if not np.isfinite(next_obs).all() or not np.isfinite(rewards).all():
            raise FloatingPointError("Nonfinite model rollout; no silent clipping or fallback")
        for o, a, r, no in zip(obs, actions, rewards, next_obs, strict=True):
            synthetic.add(o, a, float(r), no, False, False)
        generated += batch_size
        obs = next_obs
    return generated


def mixed_batch(real: RealReplayBuffer, synthetic: ModelReplayBuffer, batch_size: int,
                real_ratio: float, rng: np.random.Generator) -> tuple[TransitionBatch, int, int]:
    if not isinstance(real, RealReplayBuffer) or not isinstance(synthetic, ModelReplayBuffer):
        raise TypeError("Expected distinct real/model replay roles")
    if batch_size < 2 or not 0 < real_ratio < 1:
        raise ValueError("Require batch_size >= 2 and real_ratio in (0, 1)")
    n_real = max(1, min(batch_size - 1, int(batch_size * real_ratio)))
    n_model = batch_size - n_real
    real_batch, model_batch = real.sample(n_real), synthetic.sample(n_model)
    order = rng.permutation(batch_size)
    arrays = {key: np.concatenate((getattr(real_batch, key), getattr(model_batch, key)), axis=0)[order]
              for key in TransitionBatch.__dataclass_fields__}
    return TransitionBatch(**arrays), n_real, n_model
