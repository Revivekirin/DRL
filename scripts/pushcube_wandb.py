"""User-run posthoc W&B log import or checkpoint video reevaluation; never trains."""
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4
import yaml
from dynamics_shift.experiments.config import RunConfig, TrackingConfig
from dynamics_shift.experiments.provenance import _git_metadata
from dynamics_shift.utils.tracking import Tracker


def numeric_row(row):
    out = {}
    for key, value in row.items():
        if value in ('True', 'False'):
            out[key] = value == 'True'
        else:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                pass
    return out


def import_events(root):
    events = []
    if (root / 'metrics/train.jsonl').exists():
        for line in (root / 'metrics/train.jsonl').read_text().splitlines():
            row = json.loads(line)
            events.append((int(row['real_env_steps']), 'train', row))
    elif (root / 'metrics/train.csv').exists():
        with (root / 'metrics/train.csv').open() as stream:
            for row in csv.DictReader(stream):
                row = numeric_row(row)
                events.append((int(row['real_env_steps']), 'train', row))
    for name, namespace in [('model_refits', 'model'), ('episodes', 'train_episode')]:
        path = root / f'metrics/{name}.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                events.append((int(row['real_env_steps']), namespace, row))
    for path in sorted((root / 'metrics').glob('evaluation*/summary.json')):
        row = json.loads(path.read_text())
        events.append((int(row['counters']['real_env_steps']), 'eval', row))
    return sorted(events, key=lambda event: event[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['import-logs', 'videos'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['online', 'offline'], default='offline')
    parser.add_argument('--project', default='dynamics-shift')
    parser.add_argument('--entity')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoints', nargs='+', help='File names relative to original checkpoints directory')
    parser.add_argument('--video-seed', type=int, default=21000)
    parser.add_argument('--video-episodes', type=int, default=2)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.output_root.resolve().is_relative_to(root):
        parser.error('Output must be outside the original training run')
    raw = yaml.safe_load((root / 'config.yaml').read_text())
    mbpo = raw.pop('mbpo', None)
    config = RunConfig.from_dict(raw)
    config = replace(config, training=replace(config.training, device=args.device),
        tracking=TrackingConfig(mode=args.mode, project=args.project, entity=args.entity,
            video_every=1, video_seed=args.video_seed, video_episodes=args.video_episodes))
    checkpoints = []
    if args.operation == 'videos':
        if not args.checkpoints:
            parser.error('--checkpoints required for videos')
        for name in args.checkpoints:
            if Path(name).name != name:
                parser.error('Checkpoint names must be file basenames')
            path = root / 'checkpoints' / name
            if not path.is_file():
                parser.error(f'Missing checkpoint: {path}')
            checkpoints.append(path)
    output = args.output_root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(original_training_run=str(root), posthoc=True, operation=args.operation,
        provenance='posthoc_log_import' if args.operation=='import-logs' else 'posthoc_checkpoint_reevaluation',
        original_metadata=json.loads((root / 'metadata.json').read_text()),
        original_config=raw, mbpo_config=mbpo, upload_code_version=_git_metadata(),
        missing_q_entropy='Not reconstructed when absent from original logs')
    (output / 'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    tracker = Tracker(config, output)
    tracker.metadata(metadata)
    try:
        if args.operation == 'import-logs':
            events = import_events(root)
            if not events:
                raise ValueError('No supported log records found')
            for step, namespace, row in events:
                tracker.scalars(namespace, row, step)
            print(json.dumps(dict(output=str(output), imported_records=len(events), posthoc=True)))
        else:
            # Resolve steps from actual payloads, not file-name rounding.
            from dynamics_shift.utils.checkpoint import load_checkpoint, isolated_rng
            import torch
            with isolated_rng(torch.device(args.device)):
                ordered = []
                for path in checkpoints:
                    learner, payload = load_checkpoint(path, device='cpu')
                    ordered.append((payload['counters']['real_env_steps'], path))
                    del learner
                for step, path in sorted(ordered):
                    tracker.checkpoint_video(path, config, step, force=True, posthoc=True)
            print(json.dumps(dict(output=str(output), requested_checkpoints=len(checkpoints), posthoc=True)))
    finally:
        tracker.finish()
    if tracker.errors:
        raise SystemExit('Tracking incomplete; inspect tracking_errors.jsonl (original run unchanged)')


if __name__ == '__main__':
    main()
