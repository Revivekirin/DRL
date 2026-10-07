"""Public config/dispatch regressions; simulator/update checks remain server-only."""
from dataclasses import replace
from pathlib import Path
import json
import pytest
from dynamics_shift.experiments.dispatch import load_experiment, parse_experiment, execute

ROOT = Path(__file__).parents[1]

@pytest.mark.parametrize('algorithm', ['sac','mbpo'])
@pytest.mark.parametrize('suffix,backend', [('', 'maniskill'),('_mujoco','mujoco')])
def test_common_cli_resolves_and_dispatches_without_simulation(monkeypatch, algorithm, suffix, backend):
    from types import SimpleNamespace
    import dynamics_shift.experiments.dispatch as module
    experiment=load_experiment(ROOT/f'configs/testing/common_{algorithm}{suffix}.yaml', algorithm=algorithm)
    assert experiment.dispatch_key == (algorithm, backend)
    called=[]
    name=module.RUNNERS[experiment.dispatch_key][1]
    monkeypatch.setattr(module, 'import_module', lambda path: SimpleNamespace(**{
        name: lambda *a, **kw: called.append((a,kw)) or 'dispatch-only'}))
    assert execute(experiment,'unused') == 'dispatch-only'
    assert called
    assert parse_experiment(experiment.to_dict()).to_dict() == experiment.to_dict()

@pytest.mark.parametrize('override', ['training.unknown=1','training.real_env_steps=801',
    'env.num_envs=1','env.id=PickCube-v1','training.utd=0.00000001',
    'model.rollout_horizon=2'])
def test_reject_unsupported_before_environment(override):
    with pytest.raises((ValueError,TypeError)):
        load_experiment(ROOT/'configs/testing/common_mbpo.yaml', overrides=[override])


def test_algorithm_mismatch_and_resume_rejected():
    with pytest.raises(ValueError):
        load_experiment(ROOT/'configs/testing/common_mbpo.yaml', algorithm='sac')
    experiment=load_experiment(ROOT/'configs/testing/common_sac.yaml')
    with pytest.raises(ValueError, match='resume'):
        execute(experiment,'unused',resume='not-read.pt')

@pytest.mark.parametrize('algorithm', ['sac','mbpo'])
def test_full_run_budget_and_preserved_hyperparameters(algorithm):
    e=load_experiment(ROOT/f'configs/runs/{algorithm}_pushcube_500k.yaml')
    t=e.run.training;n=e.run.env.num_envs
    assert t.real_env_steps == 500000 and n == 32
    eligible=t.real_env_steps//n - (max(t.learning_starts,t.batch_size)+n-1)//n + 1
    assert eligible*n*t.utd == 248016
    assert e.run.algo.gamma == .8 and e.run.algo.tau == .01
    if e.model:
        assert e.model.real_ratio == .8 and e.model.rollout_horizon == 1
        assert 1+(t.real_env_steps-t.learning_starts)//e.model.model_train_frequency == 249


def test_old_and_public_pilot_configs_preserve_algorithm():
    old=load_experiment(ROOT/'configs/archive/mbpo_pushcube_pilot_20k.yaml')
    new=load_experiment(ROOT/'configs/runs/mbpo_pushcube_500k.yaml')
    assert old.run.algo == new.run.algo
    assert old.model == new.model
    for key in ('batch_size','learning_starts','utd','replay_capacity','device'):
        assert getattr(old.run.training,key) == getattr(new.run.training,key)

@pytest.mark.training
@pytest.mark.parametrize('algorithm', ['sac','mbpo'])
def test_common_scalar_runner_and_checkpoint_evaluation(tmp_path,algorithm):
    from dynamics_shift.experiments.evaluate_sac_shift import evaluate_checkpoint
    e=load_experiment(ROOT/f'configs/testing/common_{algorithm}_mujoco.yaml',
                      overrides=['training.device=cpu','tracking.mode=disabled','tracking.video_every=0'])
    run=execute(e,tmp_path,show_progress=False)
    assert (run/'resolved_config.yaml').exists()
    summary=evaluate_checkpoint(run/'checkpoints/final.pt',run/'independent_eval',device='cpu')
    assert summary['learner_state_unchanged']
    assert summary['learner_updates_during_evaluation'] == 0


def test_success_metric_is_numeric_and_ledger_matches(tmp_path):
    from types import SimpleNamespace
    from dynamics_shift.utils.tracking import Tracker
    e=load_experiment(ROOT/'configs/testing/common_sac.yaml', overrides=['tracking.mode=disabled','tracking.video_every=0'])
    tracker=Tracker(e.run,tmp_path)
    tracker.scalars('train', {'success_once':True,'success_at_end':False}, 4)
    row=json.loads((tmp_path/'tracking_metrics.jsonl').read_text())['metrics']
    assert type(row['train/success_once']) is int and row['train/success_once']==1
    assert type(row['train/success_at_end']) is int and row['train/success_at_end']==0


def test_common_collection_restores_individual_final_observations():
    from types import SimpleNamespace
    import numpy as np
    import torch
    from dynamics_shift.experiments.interaction import collect_transition
    from dynamics_shift.data.replay_buffer import ReplayBuffer
    replay=ReplayBuffer(8,2,1,0)
    reset_obs=torch.zeros((3,2))
    final=torch.tensor([[7.,8.],[9.,10.],[11.,12.]])
    term=torch.tensor([True,False,False]); trunc=torch.tensor([False,True,False])
    info={'final_observation':final,'_final_observation':term|trunc}
    env=SimpleNamespace(step=lambda action:(reset_obs,torch.ones(3),term,trunc,info))
    out=collect_transition(env,replay,torch.ones((3,2)),torch.zeros((3,1)),num_envs=3,automatic_reset=True)
    np.testing.assert_array_equal(replay._arrays['next_obs'][:3], [[7,8],[9,10],[0,0]])
    assert out[0] is reset_obs
    assert replay._arrays['terminated'][:3,0].tolist()==[True,False,False]
    assert replay._arrays['truncated'][:3,0].tolist()==[False,True,False]
    info['_final_observation']=torch.zeros(3,dtype=torch.bool)
    with pytest.raises(ValueError,match='mask'):
        collect_transition(env,replay,torch.ones((3,2)),torch.zeros((3,1)),num_envs=3,automatic_reset=True)
