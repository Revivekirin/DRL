"""Read-only MBPO diagnostics and isolated nominal policy evaluation."""
import numpy as np
import torch
from dynamics_shift.config import ExperimentConfig
from dynamics_shift.envs import make_env
from dynamics_shift.utils.checkpoint import capture_rng, restore_rng
from dynamics_shift.evaluation.state import frozen_learner


class RolloutDiagnostics:
    def __init__(self):
        self.values = {}

    def observe(self, obs, next_obs, rewards, means, elites):
        values = {
            'synthetic_reward': rewards.astype(np.float64),
            'synthetic_delta_norm': np.linalg.norm(next_obs.astype(np.float64) - obs, axis=-1),
            'ensemble_disagreement': means[elites, :, :obs.shape[-1]].astype(np.float64).var(axis=0).mean(axis=-1),
            'synthetic_next_observation_finite': np.isfinite(next_obs).all(axis=-1).astype(float),
        }
        for key, value in values.items():
            count, total, low, high = self.values.get(key, (0, 0., np.inf, -np.inf))
            self.values[key] = (count + value.size, total + value.sum(),
                                np.minimum(low, value.min()), np.maximum(high, value.max()))

    def summary(self):
        result = {}
        for key, (count, total, low, high) in self.values.items():
            if key.endswith('_finite'):
                result[key + '_fraction'] = float(total / count)
            else:
                result.update({key + '_min': float(low), key + '_mean': float(total / count), key + '_max': float(high)})
        return result


def normalization_checks(model):
    return {key + '_finite': bool(torch.isfinite(value).all().item())
            for key, value in model.normalization.items()}


def video_array(frames):
    """W&B ndarray videos require uint8 TCHW, not renderer THWC."""
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8 or not len(array):
        raise ValueError(f"render_mode=rgb_array frames={len(frames)} shape={array.shape} dtype={array.dtype}; expected uint8 THWC RGB")
    return np.ascontiguousarray(array.transpose(0, 3, 1, 2))


@frozen_learner
def evaluate_nominal(learner, config, tracker, run_dir, step, *, record_video=False):
    """No access to training environment, replay, model, or training counters."""
    state = capture_rng(learner.device)
    env = None
    frames, returns = [], []
    try:
        env = make_env(ExperimentConfig(config.env, config.dynamics, config.evaluation.seed),
                       **({'render_mode': 'rgb_array'} if record_video else {}))
        for episode in range(config.evaluation.episodes):
            obs, _ = env.reset(seed=config.evaluation.seed + episode)
            total = 0.
            if record_video and episode == 0:
                frames.append(env.render())
            for index in range(env.spec.max_episode_steps):
                obs, reward, terminated, truncated, _ = env.step(learner.act(obs, deterministic=True))
                total += float(reward)
                if record_video and episode == 0 and index < config.tracking.video_steps:
                    frames.append(env.render())
                if terminated or truncated:
                    break
            returns.append(total)
        metrics = {'eval/source/mean_return': float(np.mean(returns)),
                   'eval/source/std_return': float(np.std(returns)),
                   'eval/source/median_return': float(np.median(returns)),
                   'eval/source/episodes': len(returns)}
        if record_video:
            import wandb
            metrics['video/eval'] = wandb.Video(video_array(frames), fps=env.metadata['render_fps'], format='mp4')
        tracker.log(metrics, step)
        return {k: v for k, v in metrics.items() if not k.startswith('video/')}
    except Exception as error:
        if record_video:
            tracker.error('nominal_video', step, error)
            return evaluate_nominal(learner, config, tracker, run_dir, step, record_video=False)
        raise RuntimeError(f"MBPO evaluation failed at real_env_steps={step}; render_mode={'rgb_array' if record_video else None}; frames={len(frames)}; output={run_dir}: {error}") from error
    finally:
        if env is not None:
            env.close()
        restore_rng(state, learner.device)
