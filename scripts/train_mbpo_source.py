"""Nominal MBPO source runner. Use the smoke config before any approved full run."""
import argparse
from dataclasses import replace
from dynamics_shift.algorithms.mbpo.config import load_mbpo_config
from dynamics_shift.experiments.train_mbpo_source import train_mbpo_source
from dynamics_shift.utils.device import check_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device")
    parser.add_argument("--check-device-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"], help="Override tracking mode")
    parser.add_argument("--resume",  help="Full checkpoint to continue from (usually latest.pt)")
    args = parser.parse_args()
    config = load_mbpo_config(args.config)
    if args.wandb is not None:
        config = replace(config, tracking=replace(config.tracking, mode=args.wandb))
    if args.device:
        config = replace(config, training=replace(config.training, device=args.device))
    if args.check_device_only:
        check_device(config.training.device)
        return
    print(train_mbpo_source(config, args.output_root, resume=args.resume, show_progress=not args.no_progress))


if __name__ == "__main__":
    main()
