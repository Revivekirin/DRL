"""Run source SAC training and final frozen paired evaluation."""
import argparse
from dynamics_shift.experiments.config import load_run_config
from dynamics_shift.experiments.train_sac_source import train_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--no-progress", action="store_true", help="Disable the terminal progress bar")
    args = parser.parse_args()
    print(train_source(load_run_config(args.config), args.output_root,
                       show_progress=not args.no_progress))


if __name__ == "__main__":
    main()
