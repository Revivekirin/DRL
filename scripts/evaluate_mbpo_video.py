"""Frozen nominal checkpoint evaluation/video; no training or model fitting."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from dynamics_shift.algorithms.mbpo.checkpoint import load_mbpo_checkpoint
from dynamics_shift.algorithms.mbpo.config import load_mbpo_config
from dynamics_shift.algorithms.mbpo.monitoring import evaluate_nominal
from dynamics_shift.utils.device import check_device
from dynamics_shift.utils.tracking import Tracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='configs/testing/mbpo_video_smoke.yaml')
    args = parser.parse_args()
    config = load_mbpo_config(args.config)
    config = replace(config, tracking=replace(config.tracking, mode='online'))
    device = check_device(config.training.device)
    learner, _, payload = load_mbpo_checkpoint(args.checkpoint, device)
    run = Path('outputs/mbpo_video_validation') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    run.mkdir(parents=True)
    tracker = Tracker(config, run, args.checkpoint)
    failed = True
    try:
        print(evaluate_nominal(learner, config, tracker, run, payload['counters']['real_env_steps'], record_video=True))
        print(f'W&B: {tracker.run.url}')
        failed = False
    finally:
        tracker.finish(failed=failed)


if __name__ == '__main__':
    main()
