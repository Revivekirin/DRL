"""Read-only audit of completed SAC/MBPO local counters/metric ledger; no ML runtime."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path



def audit_episodes(run, rows, metadata):
    path = run/'metrics/episodes.jsonl'
    if not path.exists():
        return 'UNVERIFIED: legacy run has no per-environment episode file'
    groups = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        groups[row['real_env_steps']].append(row)
    initial = metadata.get('initial_counters', {}).get('episodes', 0)
    assert sum(map(len,groups.values())) == metadata['episodes']-initial
    events = defaultdict(list)
    for row in rows:
        m = row['metrics']
        for prefix in ('train/', 'train_episode/'):
            if prefix+'episode_return' in m:
                events[m['real_env_steps']].append((prefix,m))
    assert set(events) == set(groups), 'Episode steps missing from raw log or ledger'
    index_missing = False
    for step, episodes in groups.items():
        if all('env_index' in r for r in episodes):
            assert len({r['env_index'] for r in episodes}) == len(episodes), 'Duplicate environment episode'
        else:
            index_missing = True
        emitted = events[step]
        if len(emitted) == 1:
            prefix,m = emitted[0]
            assert m.get(prefix+'finished_episodes',len(episodes)) == len(episodes)
            for key in ('episode_return','episode_length','success_once','success_at_end'):
                expected = sum(float(r[key]) for r in episodes)/len(episodes)
                assert math.isclose(m[prefix+key],expected,rel_tol=1e-9,abs_tol=1e-9), (step,key)
        else:
            # Older runs logged one SDK event per environment, not a vector mean.
            assert len(emitted) == len(episodes)
            for (_,m),r in zip(emitted,episodes):
                for key in ('episode_return','episode_length','success_once','success_at_end'):
                    assert math.isclose(m['train/'+key],r[key],rel_tol=1e-9,abs_tol=1e-9)
    return 'PASS; legacy env_index unavailable' if index_missing else 'PASS'


def audit_generations(run, refits, local, ledger=None):
    path = run/'metrics/synthetic_generations.jsonl'
    if not path.exists():
        return 'UNVERIFIED: legacy run has no generation provenance' if refits else 'NOT_APPLICABLE'
    from dynamics_shift.utils.replay_provenance import ReplayProvenance
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == len(refits)
    replay = ReplayProvenance(records[0]['capacity']) if records else None
    position = size = 0
    size_by_step = {}
    for record,event in zip(records,refits):
        assert record['before_size'] == size and record['before_position'] == position
        assert record['real_env_steps'] == event['real_env_steps']
        assert record['refit'] == event['model/refit'] and record['generated'] == event['model/generated']
        expected = replay.append(record['refit'],record['real_env_steps'],record['generated'],
            before_size=size,before_position=position,after_size=record['after_size'],
            after_position=record['after_position'])
        assert record == expected, 'FIFO generation history differs'
        size,position=record['after_size'],record['after_position']
        size_by_step[record['real_env_steps']] = size
    if ledger is not None:
        emitted = [r['metrics'] for r in ledger if 'model_replay/before_position' in r['metrics']]
        assert len(emitted) == len(records)
        for record,event in zip(records,emitted):
            for key,value in record.items():
                if isinstance(value,(int,float)):
                    assert event['model_replay/'+key] == value, ('generation ledger mismatch',key)
    retained = 0
    for row in local:
        retained = size_by_step.get(row['real_env_steps'],retained)
        assert row['model_replay_size'] == retained, 'Replay changed between refits'
    return 'PASS'


def audit(run):
    run = Path(run)
    metadata = json.loads((run/'metadata.json').read_text())
    tracking = json.loads((run/'tracking_summary.json').read_text())
    assert metadata['status'] == 'complete', metadata['status']
    assert tracking['tracking_errors'] == 0, tracking
    initial = metadata.get('initial_counters', {})
    rows = [json.loads(line) for line in (run/'tracking_metrics.jsonl').read_text().splitlines()]
    assert [r['metrics']['metric_event_id'] for r in rows] == list(range(1,len(rows)+1))
    updates = [r['metrics'] for r in rows if 'policy_update/actor_loss' in r['metrics']]
    assert len(updates) == (metadata['policy_gradient_steps']-initial.get('policy_gradient_steps',0))
    assert [r['policy_gradient_steps'] for r in updates] == list(range(initial.get('policy_gradient_steps',0)+1,metadata['policy_gradient_steps']+1))
    assert sum(r['policy_update/batch_real_count'] for r in updates) == (metadata['real_policy_samples']-initial.get('real_policy_samples',0))
    assert sum(r['policy_update/batch_synthetic_count'] for r in updates) == (metadata['synthetic_policy_samples']-initial.get('synthetic_policy_samples',0))
    for r in updates:
        assert r['policy_gradient_steps'] == r['policy_update/policy_gradient_steps']
        assert r['real_env_steps'] == r['policy_update/real_env_steps']
        count = r['policy_update/batch_real_count'] + r['policy_update/batch_synthetic_count']
        assert abs(r['policy_update/actual_synthetic_ratio'] - r['policy_update/batch_synthetic_count']/count) < 1e-12
    refits = [r['metrics'] for r in rows if 'model/generated' in r['metrics']]
    assert len(refits) == (metadata['dynamics_model_refit_count']-initial.get('dynamics_model_refit_count',0))
    assert sum(r['model/generated'] for r in refits) == (metadata['synthetic_transition_count']-initial.get('synthetic_transition_count',0))
    local = [json.loads(line) for line in (run/'metrics/train.jsonl').read_text().splitlines()]
    emitted = [r['metrics'] for r in rows if 'train/vector_steps' in r['metrics']]
    assert len(local) == len(emitted) == (metadata['vector_steps']-initial.get('vector_steps',0))
    for a,b in zip(local,emitted):
        for key in ('real_env_steps','vector_steps','policy_gradient_steps','wall_time_seconds',
                    'synthetic_transition_count','real_policy_samples','synthetic_policy_samples'):
            assert a[key] == b['train/'+key], (key,a[key],b['train/'+key])
    assert all(r['delivery'] in ('offline_local','sdk_accepted_not_cloud_verified') for r in rows)
    episodes_status = audit_episodes(run, rows, metadata)
    generation_status = audit_generations(run, refits, local, rows)
    return {'episode_aggregation':episodes_status, 'rolling_provenance':generation_status, 'status':'PASS', 'events':len(rows), 'updates':len(updates),
            'real_env_steps':metadata['real_env_steps'], 'cloud_sync':'not_verified'}


