"""Declarative task/layout/codec regressions; run on the server, no simulator needed."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from dynamics_shift.envs.maniskill_tasks import TASKS, state_task
from dynamics_shift.envs.state_codec import observation_layout
from dynamics_shift.models.quaternion import QuaternionDelta, rotation_error
from dynamics_shift.data.model_data import ModelDataset
from dynamics_shift.algorithms.mbpo.maniskill import StateDiagnostics
from dynamics_shift.experiments.dispatch import load_experiment

ROOT = Path(__file__).parents[1]


def state(env_id):
    spec = state_task(env_id)
    structured = {'agent': {'qpos': np.zeros((4,9), np.float32), 'qvel': np.zeros((4,9), np.float32)}, 'extra': {}}
    for key, width in spec.extras:
        value = np.zeros((4,width), np.float32)
        if key in spec.poses:
            value[:,3] = 1
        structured['extra'][key.split('.')[1]] = value
    flat = np.concatenate([*structured['agent'].values(), *structured['extra'].values()], axis=1)
    layout = observation_layout(structured, flat, spec)
    return structured, flat, layout, spec.descriptor(layout)


@pytest.mark.parametrize('env_id,dim,poses', [('PushCube-v1',35,2),('PickCube-v1',42,2),('StackCube-v1',48,3)])
def test_actual_leaf_layout_and_all_quaternions(env_id, dim, poses):
    structured, flat, layout, descriptor = state(env_id)
    assert flat.shape == (4,dim)
    codec = QuaternionDelta(layout, descriptor['quaternion_poses'])
    assert len(codec.slices) == poses
    no = flat.copy()
    for sl in codec.slices.values():
        no[:,sl] *= -1
    data = ModelDataset(np.c_[flat,np.zeros((4,4))],np.c_[no-flat,np.zeros(4)],np.array([0,1]),np.array([2,3]))
    transformed = codec.dataset(data,dim)
    np.testing.assert_allclose(transformed.targets,0)
    normalized = codec.reconstruct(flat, transformed.targets[:,:dim])
    for sl in codec.slices.values():
        np.testing.assert_allclose(rotation_error(normalized[:,sl],no[:,sl]),0,atol=1e-6)
    diag = StateDiagnostics(layout,descriptor)
    diag.observe(flat, normalized, np.zeros(4))
    diag.observe_projected(normalized)
    assert ('goal_position_drift_rmse' in diag.records[0]) == (env_id != 'StackCube-v1')
    bad = deepcopy(structured); bad['extra']['unknown'] = np.zeros((4,1))
    with pytest.raises(ValueError,match='Unsupported'):
        observation_layout(bad,np.c_[flat,np.zeros(4)],state_task(env_id))
    with pytest.raises(AssertionError):
        observation_layout(structured,flat+1,state_task(env_id))


@pytest.mark.parametrize('task', ['pushcube','pickcube','stackcube'])
@pytest.mark.parametrize('algorithm', ['sac','mbpo'])
def test_nominal_configs_and_dispatch(task, algorithm):
    experiment = load_experiment(ROOT/f'configs/runs/{algorithm}_{task}_500k.yaml')
    assert experiment.dispatch_key == (algorithm,'maniskill')
    assert experiment.run.training.real_env_steps % experiment.run.env.num_envs == 0
    assert experiment.run.seed == 0
    if experiment.model:
        assert experiment.model.rollout_horizon == 1
        assert experiment.model.model_replay_capacity == 50000
        assert experiment.model.real_ratio == .8


def test_model_checkpoint_codec_roundtrip_and_legacy():
    from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble
    from dynamics_shift.algorithms.mbpo.config import MBPOConfig
    _,flat,layout,codec=state('StackCube-v1')
    config=MBPOConfig(ensemble_size=1,elite_size=1,model_hidden_dims=(4,))
    model=ProbabilisticEnsemble(48,4,config,observation_layout=layout,observation_codec=codec,preserve_start=True)
    saved=model.state_dict()
    restored=ProbabilisticEnsemble.from_state_dict(saved,expected_codec=codec)
    assert restored.observation_codec==codec and len(restored.geometry.slices)==3
    with pytest.raises(ValueError,match='codec'):
        ProbabilisticEnsemble.from_state_dict(saved,expected_codec=state('PushCube-v1')[3])
    _,_,oldlayout,_=state('PushCube-v1')
    legacy=ProbabilisticEnsemble(35,4,config,observation_layout=oldlayout).state_dict()
    legacy.pop('observation_codec')
    assert len(ProbabilisticEnsemble.from_state_dict(legacy).geometry.slices)==2


def test_no_quaternion_and_unsupported_physical_contract():
    from dataclasses import replace
    codec=QuaternionDelta({'position':[0,3]},poses=[])
    values=np.ones((2,3),np.float32)
    np.testing.assert_array_equal(codec.reconstruct(values,values),values*2)
    TASKS['TestFailure-v1']=replace(TASKS['PushCube-v1'],physical_termination=True)
    try:
        with pytest.raises(ValueError,match='Physical termination'):
            state_task('TestFailure-v1')
    finally:
        del TASKS['TestFailure-v1']


def test_sampled_diagnostics_preserve_batch_and_sampling_rng():
    from dynamics_shift.data.model_data import RealReplayBuffer, ModelReplayBuffer
    from dynamics_shift.algorithms.mbpo.rollouts import mixed_batch
    _,obs,layout,codec=state('StackCube-v1')
    real=RealReplayBuffer(8,48,4,0); synthetic=ModelReplayBuffer(8,48,4,1)
    for row in obs:
        for replay in (real,synthetic):
            replay.add(row,np.zeros(4),.2,row,False,False)
    real2,synthetic2=deepcopy(real),deepcopy(synthetic)
    rng=np.random.default_rng(7); rng2=deepcopy(rng)
    first,nr,ns=mixed_batch(real,synthetic,8,.8,rng)
    second,nr2,ns2=mixed_batch(real2,synthetic2,8,.8,rng2,diagnostics=StateDiagnostics(layout,codec))
    for field in first.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(first,field),getattr(second,field))
    assert (nr,ns)==(nr2,ns2)
    assert real.rng.bit_generator.state==real2.rng.bit_generator.state
    assert synthetic.rng.bit_generator.state==synthetic2.rng.bit_generator.state
    assert rng.bit_generator.state==rng2.bit_generator.state


def test_shift_capabilities_are_not_nominal_support():
    from dynamics_shift.envs.interface import supported_shift_parameters
    assert supported_shift_parameters('mujoco','HalfCheetah-v5')==('actuator_strength',)
    for env_id in TASKS:
        assert supported_shift_parameters('maniskill',env_id)==()
