"""Evaluate a checkpoint using its stored paired evaluation settings."""
import argparse
import json
from dynamics_shift.utils.device import check_device
from dynamics_shift.experiments.evaluate_sac_shift import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = check_device(args.device)
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.output_dir, device=device), indent=2))


if __name__ == "__main__":
    main()
