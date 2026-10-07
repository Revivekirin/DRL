"""Fresh nominal PushCube MBPO smoke. No checkpoint initialization or exact resume."""
import argparse
from dataclasses import replace
from dynamics_shift.algorithms.mbpo.pushcube import load_pushcube_mbpo_config
from dynamics_shift.experiments.train_pushcube_mbpo import train_pushcube_mbpo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root', default='outputs/pushcube_mbpo_smoke')
    parser.add_argument('--wandb', choices=['disabled', 'online', 'offline'])
    parser.add_argument('--video-every', type=int)
    args = parser.parse_args()
    config, model = load_pushcube_mbpo_config(args.config)
    overrides = {}
    if args.wandb is not None:
        overrides['mode'] = args.wandb
    if args.video_every is not None:
        overrides['video_every'] = args.video_every
    if overrides:
        config = replace(config, tracking=replace(config.tracking, **overrides))
    print(train_pushcube_mbpo(config, model, args.output_root))


if __name__ == '__main__':
    main()
