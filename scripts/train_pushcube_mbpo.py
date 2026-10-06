"""Fresh nominal PushCube MBPO smoke. No checkpoint initialization or exact resume."""
import argparse
from dynamics_shift.algorithms.mbpo.pushcube import load_pushcube_mbpo_config
from dynamics_shift.experiments.train_pushcube_mbpo import train_pushcube_mbpo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root', default='outputs/pushcube_mbpo_smoke')
    args = parser.parse_args()
    config, model = load_pushcube_mbpo_config(args.config)
    print(train_pushcube_mbpo(config, model, args.output_root))


if __name__ == '__main__':
    main()
