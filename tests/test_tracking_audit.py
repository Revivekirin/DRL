"""Recording-only regressions; never fit a model or update a learner."""
import json
import pytest
from dynamics_shift.utils.replay_provenance import ReplayProvenance
from dynamics_shift.utils.tracking_audit import audit_episodes, audit_generations


def test_fifo_generation_accounting_and_eviction(tmp_path):
    p=ReplayProvenance(5)
    a=p.append(1,10,3,before_size=0,before_position=0,after_size=3,after_position=3)
    b=p.append(2,20,3,before_size=3,before_position=3,after_size=5,after_position=1)
    assert b['evicted']==1
    assert [r['count'] for r in b['retained_generations']]==[2,3]
    (tmp_path/'metrics').mkdir()
    (tmp_path/'metrics/synthetic_generations.jsonl').write_text('\n'.join(map(json.dumps,[a,b])))
    refits=[{'real_env_steps':s,'model/refit':r,'model/generated':3} for s,r in [(10,1),(20,2)]]
    local=[{'real_env_steps':s,'model_replay_size':n} for s,n in [(1,0),(10,3),(11,3),(20,5)]]
    assert audit_generations(tmp_path,refits,local)=='PASS'
    local[2]['model_replay_size']=0
    with pytest.raises(AssertionError):audit_generations(tmp_path,refits,local)


def test_episode_mean_not_last_environment(tmp_path):
    (tmp_path/'metrics').mkdir()
    episodes=[dict(real_env_steps=100,env_index=i,episode_return=v,episode_length=50,
                   success_once=i,success_at_end=i) for i,v in enumerate([2.,8.])]
    (tmp_path/'metrics/episodes.jsonl').write_text('\n'.join(map(json.dumps,episodes)))
    m={'real_env_steps':100,'train/episode_return':5.,'train/episode_length':50.,
       'train/success_once':.5,'train/success_at_end':.5,'train/finished_episodes':2}
    assert audit_episodes(tmp_path,[{'metrics':m}],{'episodes':2})=='PASS'
    m['train/episode_return']=8.
    with pytest.raises(AssertionError):audit_episodes(tmp_path,[{'metrics':m}],{'episodes':2})


def test_missing_provenance_is_not_retroactively_verified(tmp_path):
    assert audit_episodes(tmp_path,[],{}) .startswith('UNVERIFIED')
    assert audit_generations(tmp_path,[{}],[]).startswith('UNVERIFIED')
