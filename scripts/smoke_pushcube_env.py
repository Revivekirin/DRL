"""Remote-only environment/replay contract check. No learner or training runner."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys

import gymnasium as gym
import numpy as np

from dynamics_shift.config import load_config
from dynamics_shift.data.replay_buffer import ReplayBuffer
from dynamics_shift.envs import make_env


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), allow_nan=False), flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_vector(value, space, name):
    require(isinstance(value, np.ndarray), f"{name}: expected NumPy array, got {type(value)}")
    require(value.shape == space.shape and value.ndim == 1,
            f"{name}: expected unbatched {space.shape}, got {value.shape}")
    require(value.dtype == space.dtype, f"{name}: dtype {value.dtype} != {space.dtype}")
    require(np.isfinite(value).all(), f"{name}: nonfinite values")
    require(space.contains(value), f"{name}: outside declared space")


def check_replay(replay, obs, action, reward, next_obs, terminated, truncated):
    # Capacity one makes the sampled transition unambiguous, using only public APIs.
    batch = replay.sample(1)
    for name, value in dict(obs=obs, action=action, reward=[reward], next_obs=next_obs).items():
        actual = getattr(batch, name)
        require(actual.dtype == np.float32, f"replay {name}: expected float32")
        np.testing.assert_array_equal(actual[0], np.asarray(value, dtype=np.float32))
    for name, value in dict(terminated=terminated, truncated=truncated).items():
        actual = getattr(batch, name)
        require(actual.shape == (1, 1) and actual.dtype == np.bool_, f"replay {name}: invalid contract")
        require(bool(actual[0, 0]) == bool(value), f"replay {name}: changed flag")


def run(config_path, episodes):
    config = load_config(config_path)
    require(config.env.backend == "maniskill", "This smoke command requires ManiSkill PushCube")
    require(episodes >= 2, "At least two episodes are required")
    packages = {}
    for name in ("mani-skill", "sapien", "gymnasium", "mujoco", "numpy", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "missing"
    emit("preflight", python=sys.version, executable=sys.executable,
         platform=platform.platform(), packages=packages, config=asdict(config))
    env = None
    transitions = terminations = truncations = successes = 0
    try:
        env = make_env(config)
        require(not env.unwrapped.gpu_sim_enabled, "Expected CPU simulation")
        require(env.unwrapped.num_envs == 1, "Expected one environment")
        require(env.unwrapped.backend.render_device is None, "Rendering must be disabled")
        require(not env.ignore_terminations, "Terminations must be preserved")
        obs_space, action_space = env.observation_space, env.action_space
        require(isinstance(obs_space, gym.spaces.Box) and isinstance(action_space, gym.spaces.Box),
                "Expected Box observation and action spaces")
        require(len(obs_space.shape) == len(action_space.shape) == 1, "Spaces must be unbatched")
        require(np.isfinite(action_space.low).all() and np.isfinite(action_space.high).all()
                and (action_space.high > action_space.low).all(), "Invalid action bounds")
        # ManiSkill's TimeLimit can carry the horizon even when Gymnasium's
        # EnvSpec has max_episode_steps=None. Use the versioned official helper
        # to inspect the actual wrapper configuration, not a hardcoded fallback.
        from mani_skill.utils.gym_utils import find_max_episode_steps_value

        horizon = find_max_episode_steps_value(env)
        emit("horizon", spec_max_episode_steps=(env.spec.max_episode_steps if env.spec else None),
             resolved_max_episode_steps=horizon,
             resolver="mani_skill.utils.gym_utils.find_max_episode_steps_value")
        require(horizon == 50, f"PushCube-v1 3.0.1 horizon changed: {horizon}")
        replay = ReplayBuffer(1, obs_space.shape[0], action_space.shape[0], seed=0)
        obs, reset_info = env.reset(seed=0)
        check_vector(obs, obs_space, "reset observation")
        require(int(np.asarray(reset_info["elapsed_steps"]).item()) == 0, "Initial reset did not clear elapsed steps")
        emit("contract", observation_shape=list(obs.shape), observation_dtype=str(obs.dtype),
             action_shape=list(action_space.shape), action_dtype=str(action_space.dtype),
             observation_finite=True, action_low=action_space.low.tolist(),
             action_high=action_space.high.tolist(), configured_horizon=horizon,
             replay_capacity=1, replay_float_dtype="float32", automatic_reset=False,
             rendering=False, seed=0)
        for episode in range(episodes):
            episode_return = 0.0
            for length in range(1, horizon + 1):
                action = action_space.sample()
                check_vector(action, action_space, "action")
                previous_obs, saved_action = obs.copy(), action.copy()
                next_obs, reward, terminated, truncated, info = env.step(action)
                check_vector(next_obs, obs_space, "step observation")
                require(isinstance(reward, (float, int, np.floating, np.integer))
                        and not isinstance(reward, (bool, np.bool_))
                        and np.isfinite(reward), f"Expected finite scalar reward, got {type(reward)}")
                require(isinstance(terminated, (bool, np.bool_))
                        and isinstance(truncated, (bool, np.bool_)), "Expected boolean boundary flags")
                require(isinstance(info.get("success"), (bool, np.bool_)), "Expected scalar success flag")
                require(bool(terminated) == bool(info["success"]), "PushCube success termination was altered")
                require(bool(truncated) == (length == horizon), "Unexpected TimeLimit boundary")
                # Also detects a same-step automatic reset before replay insertion.
                require(int(np.asarray(info["elapsed_steps"]).item()) == length,
                        "Elapsed steps changed unexpectedly (possible automatic reset)")
                final_obs = next_obs.copy()
                replay.add(previous_obs, saved_action, reward, next_obs, terminated, truncated)
                check_replay(replay, previous_obs, saved_action, reward, final_obs, terminated, truncated)
                transitions += 1
                episode_return += float(reward)
                if transitions == 1:
                    emit("step_contract", action_shape=list(action.shape), action_dtype=str(action.dtype),
                         action_finite=True, action_in_bounds=True, reward_type=type(reward).__name__,
                         reward_scalar=True, terminated_type=type(terminated).__name__,
                         truncated_type=type(truncated).__name__, replay_roundtrip=True)
                if terminated or truncated:
                    terminations += int(terminated)
                    truncations += int(truncated)
                    successes += int(terminated and info["success"])
                    # Continue the seed-0 RNG stream; do not introduce new seeds.
                    obs, reset_info = env.reset()
                    check_vector(obs, obs_space, "manual reset observation")
                    require(int(np.asarray(reset_info["elapsed_steps"]).item()) == 0,
                            "Manual reset did not clear elapsed steps")
                    check_replay(replay, previous_obs, saved_action, reward, final_obs, terminated, truncated)
                    emit("episode", episode=episode + 1, length=length, episode_return=episode_return,
                         terminated=bool(terminated), truncated=bool(truncated), success=bool(info["success"]),
                         final_next_obs_sha256=hashlib.sha256(final_obs.tobytes()).hexdigest(),
                         final_next_obs_stored_before_reset=True, replay_unchanged_after_reset=True,
                         reset_contract_preserved=True)
                    break
                obs = next_obs
            else:
                raise RuntimeError("Episode did not end at the configured horizon")
    finally:
        if env is not None:
            env.close()
            emit("close", ok=True)
    emit("summary", status="PASS" if truncations else "INCOMPLETE", episodes=episodes,
         transitions=transitions, terminated_episodes=terminations, truncated_episodes=truncations,
         observed_time_limit=horizon if truncations else None,
         success_termination="verified" if successes else "UNVERIFIED: random policy produced no success termination",
         training_performed=False, performance_validated=False)
    return 0 if truncations else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pushcube_nominal.yaml"))
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    try:
        return run(args.config, args.episodes)
    except Exception as error:
        emit("failure", status="FAIL", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    sys.exit(main())
