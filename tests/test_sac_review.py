"""Server-only review regressions; fixtures do not establish physical policy success."""
from copy import deepcopy
from dataclasses import replace
import random
import numpy as np
import pytest
import torch
from dynamics_shift.experiments.config import TrainingConfig, load_run_config
from dynamics_shift.utils.checkpoint import isolated_rng, capture_rng
from dynamics_shift.evaluation.pushcube import evaluate_loaded_checkpoint, assert_state_equal
from dynamics_shift.evaluation.contracts import pushcube_contract
from test_pushcube_sac import BoundaryFixture, learner, CONFIG


def test_legacy_update_alias():
    assert TrainingConfig(updates_per_env_step=2).utd == 2
    assert TrainingConfig(utd=2).updates_per_env_step == 2
    assert TrainingConfig(utd=.5).updates_per_env_step is None
    assert TrainingConfig(utd=2, updates_per_env_step=2).utd == 2
    with pytest.raises(ValueError, match="must agree"):
        TrainingConfig(utd=.5, updates_per_env_step=1)
    for value in (True, 0, -1, float('nan')):
        with pytest.raises(ValueError):
            TrainingConfig(utd=value)


def test_rng_restored_on_exception():
    device = torch.device('cpu')
    before = capture_rng(device)
    with pytest.raises(RuntimeError):
        with isolated_rng(device):
            random.random()
            np.random.rand()
            torch.rand(3)
            raise RuntimeError('controlled failure')
    assert_state_equal(before, capture_rng(device))


def test_vector_legacy_evaluation(tmp_path, monkeypatch):
    config = load_run_config(CONFIG)
    env = BoundaryFixture()
    contract = pushcube_contract(config, env)
    config = replace(config, env=replace(config.env, sim_backend='gpu', num_envs=32))
    saved = {**contract, 'sim_backend': 'gpu', 'num_envs': 32, 'automatic_reset': True}
    agent = learner()
    payload = dict(environment_contract=saved, counters={'policy_gradient_steps': 0})
    checkpoint = tmp_path / 'original.pt'
    checkpoint.write_bytes(b'unchanged fixture')
    monkeypatch.setattr('dynamics_shift.evaluation.pushcube.make_env', lambda *a, **k: env)
    with pytest.raises(ValueError, match='explicit'):
        evaluate_loaded_checkpoint(agent, payload, config, checkpoint, tmp_path)
    out = tmp_path / 'evaluation'
    out.mkdir()
    summary = evaluate_loaded_checkpoint(agent, payload, config, checkpoint, out,
        evaluation_overrides={'sim_backend': 'cpu', 'num_envs': 1}, episode_seeds=[0, 1, 2])
    assert summary['episodes'] == 3
    assert summary['learner_state_unchanged']
    assert summary['learner_restore_probe'].startswith('UNVERIFIED')
    assert summary['recorded_training_environment_contract'] == saved
    assert summary['unverified']
    assert checkpoint.read_bytes() == b'unchanged fixture'
    corrupted = deepcopy(payload)
    corrupted['environment_contract']['reward_mode'] = 'wrong'
    with pytest.raises(ValueError, match='contract differs'):
        evaluate_loaded_checkpoint(agent, corrupted, config, checkpoint, out,
            evaluation_overrides={'sim_backend': 'cpu', 'num_envs': 1})


def test_learner_mutation_is_rejected():
    agent = learner()
    before = deepcopy(agent.state_dict())
    with torch.no_grad():
        next(agent.critic.parameters()).add_(1)
    with pytest.raises(RuntimeError, match='learner state'):
        assert_state_equal(before, agent.state_dict())


def test_cuda_rng_restored_on_exception():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for RNG coverage')
    device = torch.device('cuda:0')
    torch.cuda.init()
    before = torch.cuda.get_rng_state_all()
    with pytest.raises(RuntimeError):
        with isolated_rng(device):
            for index in range(torch.cuda.device_count()):
                torch.rand(3, device=f'cuda:{index}')
            raise RuntimeError('controlled failure')
    assert_state_equal(before, torch.cuda.get_rng_state_all())


def test_legacy_halfcheetah_config():
    config = load_run_config(CONFIG.parent / 'sac_smoke.yaml')
    assert config.training.updates_per_env_step == 1
    assert config.training.utd == 1
