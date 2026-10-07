"""Server-only regression tests; controlled success fixtures are not policy success evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
import torch
from dynamics_shift.algorithms.sac.config import SACConfig
from dynamics_shift.algorithms.sac.learner import SACLearner
from dynamics_shift.evaluation.contracts import pushcube_contract
from dynamics_shift.evaluation.pushcube import collect_episodes, evaluate_loaded_checkpoint
from dynamics_shift.experiments.config import load_run_config, RunConfig
from dynamics_shift.experiments.train_sac_source import train_source
from dynamics_shift.utils.checkpoint import save_checkpoint, load_checkpoint
from test_sac_checkpoint import assert_nested_equal

CONFIG = Path(__file__).resolve().parents[1] / "tests/fixtures/sac_pushcube_smoke.yaml"


class BoundaryFixture(gym.Env):
    """Deterministic contract fixture: neither ManiSkill nor a trained policy."""
    observation_space = gym.spaces.Box(-10, 10, (3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    spec = SimpleNamespace(id="PushCube-v1", max_episode_steps=2)
    ignore_terminations = False
    num_envs = 1
    gpu_sim_enabled = False

    def __init__(self, success=True):
        self.success = success
        self.steps = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        return np.zeros(3, np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.ones(3, np.float32), 1., self.success, self.steps == 2, {"success": self.success}

    def close(self):
        self.closed = True


def learner():
    return SACLearner(3, -np.ones(2), np.ones(2), SACConfig(hidden_dims=(8,)), "cpu")


@pytest.mark.parametrize("success,length", [(True, 1), (False, 2)])
def test_controlled_success_and_time_limit_recording(success, length):
    agent = learner()
    rows = collect_episodes(agent, BoundaryFixture(success), seed=0, episodes=2, horizon=2)
    assert agent.actor.training
    assert all(r["episode_length"] == length and r["per_episode_return"] == length for r in rows)
    assert all(r["success_once"] == r["success_at_end"] == r["terminated"] == success for r in rows)
    assert all(r["truncated"] == (not success) for r in rows)


def test_pushcube_config_roundtrip_and_resume_rejection(tmp_path):
    config = load_run_config(CONFIG)
    assert RunConfig.from_dict(config.to_dict()) == config
    assert config.dynamics is None
    with pytest.raises(ValueError, match="ManiSkill training resume is unsupported"):
        train_source(config, tmp_path / "must_not_exist", resume="not-even-loaded.pt")
    assert not (tmp_path / "must_not_exist").exists()


def test_checkpoint_learner_only_and_evaluation_contract(tmp_path, monkeypatch):
    config = load_run_config(CONFIG)
    agent = learner()
    env = BoundaryFixture()
    contract = pushcube_contract(config, env)
    probe_obs = np.zeros(3, np.float32)
    probe = dict(observation=probe_obs.tolist(), action=agent.act(probe_obs, deterministic=True).tolist())
    checkpoint = tmp_path / "learner.pt"
    counters = dict(real_env_steps=0, policy_gradient_steps=0, episodes=0)
    save_checkpoint(checkpoint, agent, counters, config.to_dict(), environment_contract=contract, learner_probe=probe)
    restored, payload = load_checkpoint(checkpoint)
    assert_nested_equal(agent.state_dict(), restored.state_dict())
    assert payload["training_state"] is None
    assert not payload["simulator_persisted"] and not payload["replay_persisted"]
    assert not payload["training_resume_supported"]
    monkeypatch.setattr("dynamics_shift.evaluation.pushcube.make_env", lambda *a, **k: env)
    before = deepcopy(restored.state_dict())
    output = tmp_path / "eval"
    output.mkdir()
    summary = evaluate_loaded_checkpoint(restored, payload, config, checkpoint, output)
    assert summary["learner_restore_probe"] == "passed"
    assert summary["success_termination_episodes"] == 3
    assert summary["learner_updates_during_evaluation"] == 0
    assert_nested_equal(before, restored.state_dict())
    assert env.closed
    corrupted = deepcopy(payload)
    corrupted["environment_contract"]["control_mode"] = "wrong"
    with pytest.raises(ValueError, match="contract differs"):
        evaluate_loaded_checkpoint(restored, corrupted, config, checkpoint, output)
    corrupted = deepcopy(payload)
    corrupted["learner_probe"]["action"][0] += .5
    with pytest.raises(AssertionError):
        evaluate_loaded_checkpoint(restored, corrupted, config, checkpoint, output)


@pytest.mark.training
def test_controlled_success_runner_preserves_terminal_next_obs(tmp_path, monkeypatch):
    from dynamics_shift.data.replay_buffer import ReplayBuffer
    config = load_run_config(CONFIG)
    config = replace(config, training=replace(config.training, real_env_steps=4,
                     learning_starts=2, batch_size=2, checkpoint_every=2))
    monkeypatch.setattr("dynamics_shift.experiments.train_pushcube_sac.make_env", lambda *a, **k: BoundaryFixture())
    monkeypatch.setattr("dynamics_shift.evaluation.pushcube.make_env", lambda *a, **k: BoundaryFixture())
    recorded = []
    original_add = ReplayBuffer.add

    def record(self, obs, action, reward, next_obs, terminated, truncated):
        recorded.append((obs.copy(), next_obs.copy(), terminated, truncated))
        original_add(self, obs, action, reward, next_obs, terminated, truncated)

    monkeypatch.setattr(ReplayBuffer, "add", record)
    run = train_source(config, tmp_path, show_progress=False)
    for obs, next_obs, terminated, truncated in recorded:
        np.testing.assert_array_equal(obs, np.zeros(3))
        np.testing.assert_array_equal(next_obs, np.ones(3))
        assert terminated and not truncated
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["real_env_steps"] == 4 and metadata["policy_gradient_steps"] == 3
    assert metadata["episodes"] == 4 and metadata["status"] == "complete"
    assert (run / "checkpoints/initial.pt").exists()
    assert metadata["evaluation"]["learner_restore_probe"] == "passed"
