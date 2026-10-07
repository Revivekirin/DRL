"""Read-only audit of completed MBPO local counters/metric ledger; no ML runtime."""
import argparse
import json
from pathlib import Path


def audit(run):
    run = Path(run)
    metadata = json.loads((run/'metadata.json').read_text())
    tracking = json.loads((run/'tracking_summary.json').read_text())
    assert metadata['status'] == 'complete', metadata['status']
    assert tracking['tracking_errors'] == 0, tracking
    rows = [json.loads(line) for line in (run/'tracking_metrics.jsonl').read_text().splitlines()]
    assert [r['metrics']['metric_event_id'] for r in rows] == list(range(1,len(rows)+1))
    updates = [r['metrics'] for r in rows if 'policy_update/actor_loss' in r['metrics']]
    assert len(updates) == metadata['policy_gradient_steps']
    assert [r['policy_gradient_steps'] for r in updates] == list(range(1,len(updates)+1))
    assert sum(r['policy_update/batch_real_count'] for r in updates) == metadata['real_policy_samples']
    assert sum(r['policy_update/batch_synthetic_count'] for r in updates) == metadata['synthetic_policy_samples']
    for r in updates:
        assert r['policy_gradient_steps'] == r['policy_update/policy_gradient_steps']
        assert r['real_env_steps'] == r['policy_update/real_env_steps']
        count = r['policy_update/batch_real_count'] + r['policy_update/batch_synthetic_count']
        assert abs(r['policy_update/actual_synthetic_ratio'] - r['policy_update/batch_synthetic_count']/count) < 1e-12
    refits = [r['metrics'] for r in rows if 'model/generated' in r['metrics']]
    assert len(refits) == metadata['dynamics_model_refit_count']
    assert sum(r['model/generated'] for r in refits) == metadata['synthetic_transition_count']
    local = [json.loads(line) for line in (run/'metrics/train.jsonl').read_text().splitlines()]
    emitted = [r['metrics'] for r in rows if 'train/vector_steps' in r['metrics']]
    assert len(local) == len(emitted) == metadata['vector_steps']
    for a,b in zip(local,emitted):
        for key in ('real_env_steps','vector_steps','policy_gradient_steps','wall_time_seconds',
                    'synthetic_transition_count','real_policy_samples','synthetic_policy_samples'):
            assert a[key] == b['train/'+key], (key,a[key],b['train/'+key])
    assert all(r['delivery'] in ('offline_local','sdk_accepted_not_cloud_verified') for r in rows)
    return {'status':'PASS', 'events':len(rows), 'updates':len(updates),
            'real_env_steps':metadata['real_env_steps'], 'cloud_sync':'not_verified'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    print(json.dumps(audit(parser.parse_args().run_dir), indent=2))
