"""Focused binary representation and diagnostic noninterference tests (server only)."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from dynamics_shift.algorithms.mbpo.config import MBPOConfig
from dynamics_shift.data.model_data import ModelDataset, RealReplayBuffer, ModelReplayBuffer
from dynamics_shift.models.binary import BinaryNext
from dynamics_shift.models.probabilistic_ensemble import ProbabilisticEnsemble


CODEC = dict(version='maniskill_state_v1', layout={'extra.is_grasped':[0,1], 'position':[1,2]},
             quaternion_poses=[], boolean_fields=['extra.is_grasped'], invariant_fields=[], relations=[],
             target='aligned_quaternion_delta_v1', invariant_policy='learned_delta_with_drift_diagnostic',
             boolean_policy='continuous_prediction_without_thresholding', reward_policy='learned_gaussian_without_clipping')


def model(binary=True):
    settings=MBPOConfig(ensemble_size=2,elite_size=1,model_hidden_dims=(8,),model_max_epochs=1,
        model_batch_size=4,binary_features='bernoulli_next_v1' if binary else 'legacy_gaussian_delta')
    return ProbabilisticEnsemble(2,1,settings,observation_layout=CODEC['layout'],observation_codec=CODEC,preserve_start=True)


def dataset():
    a=np.array([0,0,1,1]*4,dtype=np.float32)
    b=np.array([0,1,0,1]*4,dtype=np.float32)
    x=np.column_stack((a,np.zeros_like(a),np.zeros_like(a)))
    y=np.column_stack((b-a,np.zeros_like(a),np.ones_like(a)*.2))
    return ModelDataset(x,y,np.arange(12),np.arange(12,16))


@pytest.mark.parametrize('a,b',[(0,0),(0,1),(1,0),(1,1)])
def test_binary_targets_and_reconstruction(a,b):
    m=model();data=ModelDataset(np.array([[a,2.,0.]],dtype=np.float32),
        np.array([[b-a,.25,.3]],dtype=np.float32),np.array([0]),np.array([],dtype=int))
    prepared=m.prepare_dataset(data)
    assert prepared.targets[0,0]==b
    result=m.reconstruct(data.inputs[:,:2],prepared.targets[:,:2])
    np.testing.assert_array_equal(result,[[b,2.25]])
    np.testing.assert_array_equal(prepared.inputs,data.inputs)


def test_binary_sampling_not_gaussian_and_diagnostic_rng_private():
    m=model();means=np.array([[.25,1.,.3]]*1000);variance=np.ones_like(means)*.01
    before=np.random.get_state()
    rng=np.random.default_rng(1); reference=np.random.default_rng(1)
    draws=m.sample_predictions(means,variance,rng)
    reference.standard_normal(means.shape)
    assert rng.bit_generator.state==reference.bit_generator.state
    assert set(np.unique(draws[:,0]))=={0.,1.}
    assert .2 < draws[:,0].mean() < .3
    np.testing.assert_array_equal(np.random.get_state()[1],before[1])
    assert BinaryNext(CODEC).indices==[0]
    with pytest.raises(ValueError):m.binary.check(np.array([.2]))


@pytest.mark.training
def test_fixed_binary_scale_fit_restore_and_checkpoint():
    m=model();d=dataset()
    # Initial all-zero boolean, then all four transitions. Normalizer remains fixed.
    x,y=d.inputs.copy(),d.targets.copy(); x[:,0]=0; y[:,0]=0
    zero=ModelDataset(x,y,d.train_indices,d.holdout_indices)
    m.train(zero)
    start=deepcopy(m.state_dict())
    metrics=m.train(d)
    assert np.isfinite(metrics['model_validation_loss']).all()
    for prefix in ('input','target'):
        assert m.normalization[prefix+'_mean'][0]==0
        assert m.normalization[prefix+'_std'][0]==1
        torch.testing.assert_close(m.normalization[prefix+'_std'],start['normalization'][prefix+'_std'])
    assert np.all(np.array(metrics['model_validation_loss']) <= np.array(metrics['fit_start_validation_nll'])+1e-6)
    loaded=ProbabilisticEnsemble.from_state_dict(m.state_dict(),expected_codec=m.observation_codec)
    for a,b in zip(m.predict(d.inputs),loaded.predict(d.inputs)):np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError):ProbabilisticEnsemble.from_state_dict(m.state_dict(),expected_codec=CODEC)


def test_legacy_sampling_exact_sequence_and_codec():
    m=model(False);means=np.zeros((8,3));var=np.ones_like(means)
    np.testing.assert_array_equal(m.sample_predictions(means,var,np.random.default_rng(3)),
                                  np.random.default_rng(3).standard_normal(means.shape))
    assert m.observation_codec==CODEC


def test_mixed_batch_context_does_not_change_sampling():
    from dynamics_shift.algorithms.mbpo.rollouts import mixed_batch
    a,b=RealReplayBuffer(8,2,1,0),ModelReplayBuffer(8,2,1,1)
    for i in range(8):
        for replay in (a,b):replay.add(np.array([i,0]),np.array([0]),float(i),np.array([i+1,0]),False,False)
    aa,bb=deepcopy(a),deepcopy(b)
    plain,nr,ns=mixed_batch(a,b,10,.8,np.random.default_rng(2))
    context={};diag,_,_=mixed_batch(aa,bb,10,.8,np.random.default_rng(2),context=context)
    for key in vars(plain):np.testing.assert_array_equal(getattr(plain,key),getattr(diag,key))
    assert context['synthetic_mask'].sum()==ns
    assert np.all(context['synthetic_slot'][~context['synthetic_mask']]==-1)


@pytest.mark.training
def test_actual_td_diagnostics_do_not_change_update_or_rng():
    from dynamics_shift.algorithms.sac.learner import SACLearner
    from dynamics_shift.algorithms.sac.config import SACConfig
    from dynamics_shift.data.replay_buffer import TransitionBatch
    from dynamics_shift.evaluation.state import assert_state_equal
    from dynamics_shift.utils.checkpoint import isolated_rng
    torch.manual_seed(0)
    learner=SACLearner(2,np.array([-1.]),np.array([1.]),SACConfig(hidden_dims=(8,8)))
    other=deepcopy(learner)
    batch=TransitionBatch(np.zeros((4,2)),np.zeros((4,1)),np.ones((4,1)),np.zeros((4,2)),np.zeros((4,1)),np.ones((4,1)))
    with isolated_rng('cpu'):learner.update(batch)
    td={}
    with isolated_rng('cpu'):other.update(batch,diagnostics=td)
    assert_state_equal(learner.state_dict(),other.state_dict())
    torch.testing.assert_close(td['td_target'],td['reward']+td['bootstrap_multiplier']*(td['target_q_min']+td['entropy_term']))


def test_binary_rollout_preserves_quaternion_flags_and_unclipped_reward():
    from types import SimpleNamespace
    from dynamics_shift.algorithms.mbpo.rollouts import generate_rollouts
    codec=deepcopy(CODEC)
    codec.update(layout={'extra.is_grasped':[0,1],'pose':[1,8]},quaternion_poses=['pose'])
    m=ProbabilisticEnsemble(8,1,MBPOConfig(ensemble_size=1,elite_size=1,model_hidden_dims=(4,),
        binary_features='bernoulli_next_v1'),observation_layout=codec['layout'],observation_codec=codec)
    obs=np.zeros(8,dtype=np.float32);obs[4]=1
    real=RealReplayBuffer(4,8,1,0);real.add(obs,np.zeros(1),0.,obs,False,True)
    synthetic=ModelReplayBuffer(8,8,1,1)
    def predict(inputs):
        mean=np.zeros((1,len(inputs),9));mean[:,:,0]=.7;mean[:,:,4]=.1;mean[:,:,-1]=2.
        variance=np.zeros_like(mean);variance[:,:,0]=.21
        return mean,variance
    m.predict=predict
    learner=SimpleNamespace(act=lambda inputs,deterministic=False:np.zeros((len(inputs),1)))
    generate_rollouts(learner,m,real,synthetic,8,1,np.random.default_rng(2))
    values=synthetic._arrays
    assert np.isfinite(values['next_obs']).all()
    assert np.isin(values['next_obs'][:,0],[0,1]).all()
    np.testing.assert_allclose(np.linalg.norm(values['next_obs'][:,4:8],axis=1),1,atol=1e-6)
    assert (values['reward']==2).all()
    assert not values['terminated'].any() and not values['truncated'].any()
