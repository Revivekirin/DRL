"""Compare fixed-action trajectories from the same full simulator state."""
import argparse
import csv
from pathlib import Path
import numpy as np
from dynamics_shift.config import load_config
from dynamics_shift.envs import make_env
from dynamics_shift.envs.mujoco_state import capture_state, restore_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("configs/halfcheetah_nominal.yaml"))
    parser.add_argument("--target", type=Path, default=Path("configs/halfcheetah_actuator_070.yaml"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/actuator_shift_verification.csv"))
    args = parser.parse_args()
    source, target = load_config(args.source), load_config(args.target)
    if source.env != target.env or source.seed != target.seed:
        parser.error("Source and target must have identical env and seed")
    env = make_env(source)
    try:
        if not 1 <= args.steps <= env.spec.max_episode_steps:
            parser.error("steps must be within the episode horizon")
        env.reset(seed=source.seed)
        snapshot = capture_state(env)
        obs_space, action_space = env.observation_space, env.action_space
        actions = np.random.default_rng(source.seed).uniform(
            env.action_space.low, env.action_space.high,
            size=(args.steps, *env.action_space.shape)).astype(env.action_space.dtype)
        before = env.unwrapped.model.actuator_gear.copy()
        nominal = np.stack([env.step(action)[0] for action in actions])
        restore_state(env, snapshot)
        env.set_actuator_scale(target.dynamics.actuator_scale)
        after = env.unwrapped.model.actuator_gear.copy()
        initial_equal = np.array_equal(snapshot.integration_state, capture_state(env).integration_state)
        shifted = np.stack([env.step(action)[0] for action in actions])
        differences = np.linalg.norm(nominal - shifted, axis=1)
        print("Actuator gear before:\n", before)
        print("Actuator gear after:\n", after)
        print("Observation-space equality:", obs_space == env.observation_space)
        print("Action-space equality:", action_space == env.action_space)
        print("Initial-state equality:", initial_equal)
        for step, difference in enumerate(differences, 1):
            print(f"step {step:02d}: {difference:.10f}")
        print(f"Mean difference: {differences.mean():.10f}")
        print(f"Max difference: {differences.max():.10f}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "observation_l2_difference"])
            writer.writerows(enumerate(differences, 1))
        if not initial_equal:
            raise RuntimeError("Initial integration states differ")
    finally:
        env.close()


if __name__ == "__main__":
    main()
