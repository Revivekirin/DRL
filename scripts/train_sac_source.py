"""Run source SAC with backend-specific checkpointing and final evaluation."""
import argparse
from dataclasses import replace
from dynamics_shift.utils.device import check_device
from dynamics_shift.experiments.config import load_run_config
from dynamics_shift.experiments.train_sac_source import train_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--no-progress", action="store_true", help="Disable the terminal progress bar")
    parser.add_argument("--device", help="Override training device, e.g. cuda:0")
    parser.add_argument("--check-device-only", action="store_true", help="Check device and exit without training")
    parser.add_argument("--resume", help="Full checkpoint to continue from (usually latest.pt)")
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"], help="Override tracking mode")
    args = parser.parse_args()
    config = load_run_config(args.config)
    if args.wandb is not None:
        config = replace(config, tracking=replace(config.tracking, mode=args.wandb))
    if args.device is not None:
        config = replace(config, training=replace(config.training, device=args.device))
    if args.check_device_only:
        check_device(config.training.device)
        return
    print(train_source(config, args.output_root,
                       show_progress=not args.no_progress, resume=args.resume))


if __name__ == "__main__":
    main()
