"""Short model rollouts and provenance-independent batches for shared SAC."""
import numpy as np
from dynamics_shift.data.replay_buffer import TransitionBatch
from dynamics_shift.data.model_data import RealReplayBuffer, ModelReplayBuffer
from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble
from dynamics_shift.algorithms.sac.learner import SACLearner


def generate_rollouts(learner: SACLearner, model: ProbabilisticEnsemble,
                      real: RealReplayBuffer, synthetic: ModelReplayBuffer,
                      batch_size: int, horizon: int, rng: np.random.Generator, *, diagnostics=None) -> int:
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
        if not np.isfinite(means).all() or not np.isfinite(variances).all():
            raise FloatingPointError("Nonfinite ensemble prediction during model rollout")
        # TS1: independently choose an elite for each transition at every depth.
        members = rng.choice(model.elites, size=batch_size)
        selected_mean = means[members, np.arange(batch_size)]
        selected_var = variances[members, np.arange(batch_size)]
        predictions = (model.sample_predictions(selected_mean, selected_var, rng)
                       if hasattr(model, 'sample_predictions') else
                       selected_mean + np.sqrt(selected_var) * rng.standard_normal(selected_mean.shape))
        raw_next = (model.reconstruct(obs, predictions[:, :model.obs_dim], normalize=False)
                    if hasattr(model, "reconstruct") else obs + predictions[:, :model.obs_dim])
        next_obs = (model.reconstruct(obs, predictions[:, :model.obs_dim])
                    if hasattr(model, "reconstruct") else raw_next).astype(np.float32)
        rewards = predictions[:, -1].astype(np.float32)
        if diagnostics is not None and getattr(diagnostics, 'capture_extremes', False):
            ids = np.unique([int(np.argmin(rewards)), int(np.argmax(rewards))])
            diagnostics.extremes = dict(obs=obs[ids].copy(), action=actions[ids].copy(),
                next_obs=next_obs[ids].copy(), raw_next=raw_next[ids].copy(), reward=rewards[ids].copy(),
                selected_member=members[ids].copy(), mean=selected_mean[ids].copy(),
                variance=selected_var[ids].copy(), sampled_prediction=predictions[ids].copy(),
                sampling_contribution=(predictions-selected_mean)[ids].copy(), generation_offset=ids)
        if diagnostics is not None:
            diagnostics.observe(obs, raw_next, rewards, means, model.elites)
            if hasattr(diagnostics, "observe_projected"):
                diagnostics.observe_projected(next_obs)
        if not np.isfinite(next_obs).all() or not np.isfinite(rewards).all():
            raise FloatingPointError("Nonfinite model rollout; no silent clipping or fallback")
        if getattr(model, 'binary', None):
            model.binary.check(next_obs[:, model.binary.indices])
        for o, a, r, no in zip(obs, actions, rewards, next_obs, strict=True):
            synthetic.add(o, a, float(r), no, False, False)
        generated += batch_size
        obs = next_obs
    return generated


def mixed_batch(real: RealReplayBuffer, synthetic: ModelReplayBuffer, batch_size: int,
                real_ratio: float, rng: np.random.Generator, *, diagnostics=None, context=None) -> tuple[TransitionBatch, int, int]:
    if not isinstance(real, RealReplayBuffer) or not isinstance(synthetic, ModelReplayBuffer):
        raise TypeError("Expected distinct real/model replay roles")
    if batch_size < 2 or not 0 < real_ratio < 1:
        raise ValueError("Require batch_size >= 2 and real_ratio in (0, 1)")
    n_real = max(1, min(batch_size - 1, int(batch_size * real_ratio)))
    n_model = batch_size - n_real
    real_batch = real.sample(n_real)
    if context is None:
        model_batch = synthetic.sample(n_model)
    else:
        indices = synthetic.rng.integers(len(synthetic), size=n_model)
        model_batch = TransitionBatch(**{k: v[indices] for k, v in synthetic._arrays.items()})
    if diagnostics is not None:
        diagnostics.observe(model_batch.obs, model_batch.next_obs, model_batch.reward)
    order = rng.permutation(batch_size)
    if context is not None:
        context['synthetic_mask'] = np.concatenate((np.zeros(n_real, dtype=bool), np.ones(n_model, dtype=bool)))[order]
        context['synthetic_slot'] = np.concatenate((np.full(n_real, -1), indices))[order]
    arrays = {key: np.concatenate((getattr(real_batch, key), getattr(model_batch, key)), axis=0)[order]
              for key in TransitionBatch.__dataclass_fields__}
    return TransitionBatch(**arrays), n_real, n_model
