"""Shared loss/counter records: one SDK event equals one local ledger payload."""
import json


def record_update(tracker, counters, losses, *, real_count, synthetic_count,
                  requested_real_ratio, real_replay_size, model_replay_size, wall_time):
    tracker.scalars('policy_update', {**counters, **losses,
        'batch_real_count': real_count, 'batch_synthetic_count': synthetic_count,
        'requested_real_ratio': requested_real_ratio,
        'actual_synthetic_ratio': synthetic_count/(real_count+synthetic_count),
        'real_replay_size': real_replay_size, 'model_replay_size': model_replay_size,
        'wall_time_seconds': wall_time}, counters['real_env_steps'],
        axis='policy_gradient_steps', axis_value=counters['policy_gradient_steps'])


def record_transition(tracker, path, counters, losses, elapsed, **extra):
    record = dict(**counters, **losses, wall_time_seconds=elapsed,
                  transitions_per_second=counters['real_env_steps']/max(elapsed, 1e-9), **extra)
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False)+'\n')
    tracker.scalars('train', record, counters['real_env_steps'])
