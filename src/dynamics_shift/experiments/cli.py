"""Public CLI shared by SAC and MBPO; no simulation in print-config-only mode."""
import argparse
import yaml
from .dispatch import load_experiment, execute, parse_experiment


def train_main(algorithm):
    parser = argparse.ArgumentParser(description=f'Train {algorithm.upper()} using an explicit backend contract')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root', default='outputs')
    parser.add_argument('--wandb', choices=['online','offline','disabled'])
    parser.add_argument('--device')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--print-config-only', action='store_true')
    parser.add_argument('--check-env-only', action='store_true', help='Server-only random interaction/contract check, no learner')
    parser.add_argument('--check-device-only', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--resume', help='Exact MuJoCo continuation only; rejects ManiSkill')
    args = parser.parse_args()
    overrides = list(args.set)
    if args.wandb is not None:
        overrides.append(f'tracking.mode={args.wandb}')
        if args.wandb == 'disabled':
            overrides.append('tracking.video_every=0')
    if args.device is not None:
        overrides.append(f'training.device={args.device}')
    experiment = load_experiment(args.config, algorithm=algorithm, overrides=overrides)
    print(yaml.safe_dump(experiment.to_dict(), sort_keys=False))
    if args.resume and experiment.run.env.backend == 'maniskill':
        parser.error('Exact ManiSkill training resume is unsupported')
    if args.print_config_only:
        return
    if args.check_env_only:
        from .environment_check import check_environment
        check_environment(experiment.run)
        return
    if args.check_device_only:
        from dynamics_shift.utils.device import check_device
        check_device(experiment.run.training.device)
        return
    print(execute(experiment, args.output_root, resume=args.resume, show_progress=not args.no_progress))
