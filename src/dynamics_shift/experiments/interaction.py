"""Shared scalar/vector interaction and contract helpers. No learner updates."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from dynamics_shift.evaluation.contracts import episode_horizon

def _to_numpy(value) -> np.ndarray:
    """Detach tensors and return CPU NumPy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _single_observation_space(env):
    if hasattr(env, "single_observation_space"):
        return env.single_observation_space
    return env.observation_space


def _single_action_space(env):
    if hasattr(env, "single_action_space"):
        return env.single_action_space
    return env.action_space


def _num_envs(env, config) -> int:
    if hasattr(env, "num_envs"):
        return int(env.num_envs)

    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "num_envs"):
        return int(unwrapped.num_envs)

    return int(config.env.num_envs or 1)


def _horizon(env) -> int:
    """Find horizon through either scalar or ManiSkill vector wrappers."""
    candidates = [
        env,
        getattr(env, "_env", None),
        getattr(env, "unwrapped", None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        try:
            return episode_horizon(candidate)
        except Exception:
            pass

    raise ValueError("Could not determine ManiSkill episode horizon")


def _batch_size_from_obs(obs, num_envs: int) -> int:
    array = obs if isinstance(obs, torch.Tensor) else np.asarray(obs)

    if num_envs == 1:
        return 1

    if array.ndim != 2 or array.shape[0] != num_envs:
        raise ValueError(
            f"Expected batched observations with shape "
            f"[{num_envs}, obs_dim], got {tuple(array.shape)}"
        )

    return num_envs


def _deterministic_action(
    learner: SACLearner,
    obs,
):
    """Run deterministic actor without forcing GPU observations through NumPy."""
    if isinstance(obs, torch.Tensor):
        tensor = obs.to(
            device=learner.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            return learner.actor.deterministic(tensor)

    return learner.act(
        np.asarray(obs),
        deterministic=True,
    )


def _policy_action(
    learner: SACLearner,
    obs,
):
    """Sample from SAC actor while preserving environment tensor type."""
    if isinstance(obs, torch.Tensor):
        tensor = obs.to(
            device=learner.device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            action, _ = learner.actor.sample(tensor)

        return action

    return learner.act(
        np.asarray(obs),
        deterministic=False,
    )


def _random_action(
    env,
    obs,
    *,
    num_envs: int,
    device: torch.device,
):
    """Generate uniformly distributed actions for warm-up collection."""
    single_space = _single_action_space(env)

    low = np.asarray(
        single_space.low,
        dtype=np.float32,
    )
    high = np.asarray(
        single_space.high,
        dtype=np.float32,
    )

    if num_envs == 1:
        return env.action_space.sample()

    # ManiSkill GPU env expects torch tensors.
    if isinstance(obs, torch.Tensor):
        low_t = torch.as_tensor(
            low,
            dtype=torch.float32,
            device=device,
        )
        high_t = torch.as_tensor(
            high,
            dtype=torch.float32,
            device=device,
        )

        random01 = torch.rand(
            (num_envs, low.shape[0]),
            dtype=torch.float32,
            device=device,
        )

        return low_t + random01 * (high_t - low_t)

    rng = np.random.random(
        (num_envs, low.shape[0])
    ).astype(np.float32)

    return low + rng * (high - low)


def _extract_vector_success(
    info: dict,
    terminated,
    num_envs: int,
) -> np.ndarray:
    """Return success flags for the transition that just occurred.

    ManiSkillVectorEnv may auto-reset completed environments. In that case,
    terminal-step metrics must be recovered from ``final_info`` using the
    ``_final_info`` mask rather than relying only on ``info["success"]``.
    """

    terminated_np = (
        _to_numpy(terminated)
        .astype(np.bool_, copy=False)
        .reshape(-1)
    )

    if terminated_np.shape != (num_envs,):
        raise ValueError(
            "terminated has wrong vector shape: "
            f"{terminated_np.shape}; expected ({num_envs},)"
        )

    # Current observations / non-final transitions.
    current_success = info.get("success")

    if current_success is None:
        success_np = np.zeros(
            num_envs,
            dtype=np.bool_,
        )
    else:
        success_np = (
            _to_numpy(current_success)
            .astype(np.bool_, copy=False)
            .reshape(-1)
            .copy()
        )

        if success_np.shape != (num_envs,):
            raise ValueError(
                "PushCube success has wrong vector shape: "
                f"{success_np.shape}; expected ({num_envs},)"
            )

    # ManiSkillVectorEnv auto-reset path.
    #
    # For environments that terminated/truncated on this step,
    # info["success"] can already correspond to the reset state.
    # The transition that actually ended the episode is recorded
    # in info["final_info"].
    final_info = info.get("final_info")
    final_mask = info.get("_final_info")

    if final_info is not None and final_mask is not None:
        final_mask_np = (
            _to_numpy(final_mask)
            .astype(np.bool_, copy=False)
            .reshape(-1)
        )

        if final_mask_np.shape != (num_envs,):
            raise ValueError(
                "_final_info has wrong vector shape: "
                f"{final_mask_np.shape}; expected ({num_envs},)"
            )

        final_success = final_info.get("success")

        if final_success is not None:
            final_success_np = (
                _to_numpy(final_success)
                .astype(np.bool_, copy=False)
                .reshape(-1)
            )

            if final_success_np.shape != (num_envs,):
                raise ValueError(
                    "final_info['success'] has wrong vector shape: "
                    f"{final_success_np.shape}; "
                    f"expected ({num_envs},)"
                )

            success_np[final_mask_np] = (
                final_success_np[final_mask_np]
            )

    # PushCube has success as its task termination condition.
    #
    # Validate only the implication needed by this project:
    #
    #     terminated -> success
    #
    # Do not require the raw current info["success"] tensor to be
    # elementwise identical to terminated because vector auto-reset
    # can replace current info with reset-state information.
    invalid = terminated_np & (~success_np)

    if invalid.any():
        raise ValueError(
            "PushCube reported terminated=True without terminal "
            "success for "
            f"{int(invalid.sum())} environment(s). "
            "Inspect final_info/_final_info semantics."
        )

    return success_np


def _true_next_observation(
    next_obs,
    terminated,
    truncated,
    info: dict,
):
    """Recover final observations before vector-env automatic resets."""
    if not isinstance(next_obs, torch.Tensor):
        return np.asarray(next_obs)

    real_next_obs = next_obs.clone()

    terminated_t = torch.as_tensor(
        terminated,
        dtype=torch.bool,
        device=next_obs.device,
    )
    truncated_t = torch.as_tensor(
        truncated,
        dtype=torch.bool,
        device=next_obs.device,
    )

    done = terminated_t | truncated_t

    if not bool(done.any()):
        return real_next_obs

    if "final_observation" not in info:
        raise ValueError(
            "Vector ManiSkill environment auto-reset a completed "
            "episode but did not expose info['final_observation']."
        )

    final_obs = info["final_observation"]

    if not isinstance(final_obs, torch.Tensor):
        final_obs = torch.as_tensor(
            final_obs,
            dtype=real_next_obs.dtype,
            device=real_next_obs.device,
        )

    if final_obs.shape != real_next_obs.shape:
        raise ValueError(
            "final_observation shape mismatch: "
            f"{tuple(final_obs.shape)} vs "
            f"{tuple(real_next_obs.shape)}"
        )

    real_next_obs[done] = final_obs[done]

    return real_next_obs


def _add_replay_batch(
    replay: ReplayBuffer,
    obs,
    action,
    reward,
    next_obs,
    terminated,
    truncated,
    *,
    num_envs: int,
) -> None:
    """Insert scalar or vector transitions into existing NumPy replay."""
    obs_np = _to_numpy(obs).astype(
        np.float32,
        copy=False,
    )
    action_np = _to_numpy(action).astype(
        np.float32,
        copy=False,
    )
    reward_np = _to_numpy(reward).astype(
        np.float32,
        copy=False,
    )
    next_obs_np = _to_numpy(next_obs).astype(
        np.float32,
        copy=False,
    )
    terminated_np = _to_numpy(terminated).astype(
        np.bool_,
        copy=False,
    )
    truncated_np = _to_numpy(truncated).astype(
        np.bool_,
        copy=False,
    )

    if num_envs == 1:
        replay.add(
            np.asarray(obs_np).reshape(-1),
            np.asarray(action_np).reshape(-1),
            float(np.asarray(reward_np).reshape(-1)[0]),
            np.asarray(next_obs_np).reshape(-1),
            bool(np.asarray(terminated_np).reshape(-1)[0]),
            bool(np.asarray(truncated_np).reshape(-1)[0]),
        )
        return

    expected_n = num_envs

    obs_np = obs_np.reshape(expected_n, -1)
    action_np = action_np.reshape(expected_n, -1)
    reward_np = reward_np.reshape(expected_n)
    next_obs_np = next_obs_np.reshape(expected_n, -1)
    terminated_np = terminated_np.reshape(expected_n)
    truncated_np = truncated_np.reshape(expected_n)

    # Keep ReplayBuffer unchanged for now.
    # This can later be replaced by ReplayBuffer.add_batch().
    for env_index in range(expected_n):
        replay.add(
            obs_np[env_index],
            action_np[env_index],
            float(reward_np[env_index]),
            next_obs_np[env_index],
            bool(terminated_np[env_index]),
            bool(truncated_np[env_index]),
        )


def _training_contract(
    config,
    env,
    *,
    num_envs: int,
) -> dict:
    obs_space = _single_observation_space(env)
    action_space = _single_action_space(env)

    return {
        "env_id": config.env.id,
        "backend": config.env.backend,
        "observation_dim": int(obs_space.shape[0]),
        "action_dim": int(action_space.shape[0]),
        "observation_dtype": str(obs_space.dtype),
        "action_dtype": str(action_space.dtype),
        "action_low": np.asarray(
            action_space.low
        ).tolist(),
        "action_high": np.asarray(
            action_space.high
        ).tolist(),
        "obs_mode": config.env.obs_mode,
        "robot_uids": config.env.robot_uids,
        "control_mode": config.env.control_mode,
        "reward_mode": config.env.reward_mode,
        "sim_backend": config.env.sim_backend,
        "num_envs": num_envs,
        "horizon": _horizon(env),
        "termination_policy":
            "ignore_terminations" if config.env.sim_backend == "gpu" else "terminate_on_success",
        "truncation_policy":
            "bootstrap_from_final_observation",
        "automatic_reset": config.env.sim_backend == "gpu",
    }


# ---------------------------------------------------------------------------
# Dedicated CPU evaluation for vector-trained checkpoints
# ---------------------------------------------------------------------------


def _evaluate_saved_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    config,
    *,
    device: torch.device,
) -> dict:
    """Use the same verified evaluator as the standalone CLI, isolating all RNG."""
    from dynamics_shift.experiments.evaluate_sac_shift import evaluate_checkpoint
    from dynamics_shift.utils.checkpoint import isolated_rng
    with isolated_rng(device):
        return evaluate_checkpoint(checkpoint, output_dir, device=device,
                                   evaluation_overrides={"sim_backend": "cpu", "num_envs": 1})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------



def collect_transition(env, replay, obs, action, *, num_envs=1, automatic_reset=False):
    """Store true final observations before a caller resets a scalar environment."""
    next_obs, reward, terminated, truncated, info = env.step(action)
    term = _to_numpy(terminated).astype(bool).reshape(num_envs)
    trunc = _to_numpy(truncated).astype(bool).reshape(num_envs)
    done = term | trunc
    if automatic_reset and done.any():
        mask = info.get('_final_observation')
        if mask is None or not np.array_equal(_to_numpy(mask).reshape(num_envs), done):
            raise ValueError('Missing or inconsistent final-observation mask')
    true_next = (_true_next_observation(next_obs, terminated, truncated, info)
                 if automatic_reset else next_obs)
    if num_envs == 1 and not automatic_reset:
        replay.add(obs, action, reward, true_next, terminated, truncated)
    else:
        _add_replay_batch(replay, obs, action, reward, true_next, terminated, truncated, num_envs=num_envs)
    if automatic_reset and done.any():
        slots = np.arange(replay.position-num_envs, replay.position) % replay.capacity
        np.testing.assert_array_equal(replay._arrays['next_obs'][slots[done]],
                                      _to_numpy(info['final_observation'])[done])
    return next_obs, reward, terminated, truncated, info
