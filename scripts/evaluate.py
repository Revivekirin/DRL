"""Load a learner and evaluate: paired HalfCheetah shift or nominal PushCube."""
import argparse
import json
from dynamics_shift.utils.device import check_device
from dynamics_shift.experiments.evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-sim-backend", choices=["cpu"])
    parser.add_argument("--eval-num-envs", type=int, choices=[1])
    parser.add_argument("--episode-seeds", type=int, nargs="+", help="Explicit reset seed for every PushCube evaluation episode")
    args = parser.parse_args()
    overrides = {k: v for k, v in {"sim_backend": args.eval_sim_backend,
                 "num_envs": args.eval_num_envs}.items() if v is not None}
    device = check_device(args.device)
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.output_dir, device=device,
        evaluation_overrides=overrides, episode_seeds=args.episode_seeds, episodes=args.episodes), indent=2))


if __name__ == "__main__":
    main()
