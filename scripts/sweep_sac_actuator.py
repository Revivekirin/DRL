"""Run constant-condition calibration with the existing seed-0 actor only."""
import argparse
from dataclasses import replace
from dynamics_shift.evaluation.severity_sweep import load_sweep_config
from dynamics_shift.experiments.frozen_severity_sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/frozen_actuator_severity.yaml")
    parser.add_argument("--checkpoint", help="Location of the same existing seed-0 source checkpoint")
    parser.add_argument("--device", help="Override evaluation device, e.g. cuda:0")
    args = parser.parse_args()
    config = load_sweep_config(args.config)
    if args.checkpoint:
        config = replace(config, checkpoint=args.checkpoint)
    if args.device:
        config = replace(config, device=args.device)
    print(run_sweep(config))


if __name__ == "__main__":
    main()
