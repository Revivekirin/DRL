"""Frozen inference only: never construct a learner or execute an update."""
import csv
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from dynamics_shift.algorithms.sac.networks import GaussianActor
from dynamics_shift.config import ExperimentConfig, DynamicsConfig
from dynamics_shift.envs import make_env, DynamicsShiftController
from dynamics_shift.evaluation.frozen_actor import FrozenActor, load_frozen_actor
from dynamics_shift.evaluation.severity_sweep import SweepConfig, collect_sweep, summarize
from dynamics_shift.evaluation.policy_eval import evaluate_policy
from dynamics_shift.experiments.frozen_severity_sweep import write_csv, sha256


@pytest.fixture
def actor_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        actor = GaussianActor(17, -np.ones(6), np.ones(6), (8,))
    return {"obs_dim": 17, "action_low": [-1.] * 6, "action_high": [1.] * 6,
            "config": {"hidden_dims": [8]}, "actor": actor.state_dict()}


def test_paired_constant_conditions_and_unchanged_actor(actor_state):
    policy = FrozenActor(actor_state, "cpu")
    config = SweepConfig(checkpoint="unused-in-unit-test", evaluation_seeds=(100, 101), device="cpu")
    calls = []
    def factory(env_config):
        env = make_env(env_config)
        original_apply, original_reset, original_step = env.apply_dynamics_shift, env.reset, env.step
        seen = {"scale": None, "seeds": []}
        calls.append(seen)
        def apply(parameter, value):
            assert parameter == "actuator_strength"
            seen["scale"] = value
            original_apply(parameter, value)
        def reset(*, seed=None, **kwargs):
            seen["seeds"].append(seed)
            return original_reset(seed=seed, **kwargs)
        def step(action):
            assert env.get_shift_parameter("actuator_strength") == seen["scale"]
            return original_step(action)
        env.apply_dynamics_shift, env.reset, env.step = apply, reset, step
        return env
    before = {k: v.clone() for k, v in policy.actor.state_dict().items()}
    with patch("dynamics_shift.evaluation.severity_sweep.make_env", side_effect=factory), \
         patch.object(DynamicsShiftController, "__init__", side_effect=AssertionError("No temporal controller")), \
         patch("dynamics_shift.algorithms.sac.learner.SACLearner.__init__", side_effect=AssertionError("No learner")), \
         patch("torch.optim.Adam.__init__", side_effect=AssertionError("No optimizer")):
        rows = collect_sweep(policy, config)
    assert len(rows) == 16
    assert [c["scale"] for c in calls] == list(config.actuator_strengths)
    assert all(c["seeds"] == [100, 101] for c in calls)
    assert all(torch.equal(v, policy.actor.state_dict()[k]) for k, v in before.items())
    assert all(not p.requires_grad for p in policy.actor.parameters())
    # Match the established frozen evaluator at both original reference conditions.
    for scale in (1., .7):
        env = make_env(ExperimentConfig(dynamics=DynamicsConfig(scale)))
        try:
            reference = evaluate_policy(policy, env, config.evaluation_seeds, "reference", scale)
            actual = [r for r in rows if r["actuator_strength"] == scale]
            np.testing.assert_allclose([r["episode_return"] for r in actual],
                                       [r.per_episode_return for r in reference], rtol=0, atol=1e-12)
        finally:
            env.close()


def test_statistics_from_csv_and_seed_pairing(tmp_path):
    rows = [dict(actuator_strength=scale, evaluation_seed=seed, episode_return=value)
            for scale, seed, value in [(1., 101, 20), (.7, 100, 4), (1., 100, 10), (.7, 101, 8)]]
    path = tmp_path / "per_episode.csv"
    write_csv(path, rows)
    with path.open() as stream:
        summary, paired, analysis = summarize(list(csv.DictReader(stream)))
    write_csv(tmp_path / "summary.csv", summary)
    nominal, shifted = summary
    assert nominal["mean_return"] == 15
    assert nominal["std_return"] == 5
    assert shifted["mean_return"] == shifted["median_return"] == 6
    assert shifted["min_return"] == 4 and shifted["max_return"] == 8
    assert shifted["relative_return"] == .4 and shifted["relative_drop"] == .6
    assert shifted["absolute_drop"] == shifted["paired_mean_delta"] == -9
    assert [r["paired_delta_return"] for r in paired if r["actuator_strength"] == .7] == [-6, -12]
    assert analysis["monotonic_nondecreasing_with_strength"]
    assert analysis["steepest_decrease_region"]["to_strength"] == .7
    with (tmp_path / "summary.csv").open() as stream:
        persisted = list(csv.DictReader(stream))
    assert float(persisted[1]["mean_return"]) == 6
    with pytest.raises(ValueError, match="same evaluation seeds"):
        summarize(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(rows + [rows[0]])


def test_zero_nominal_and_nonmonotonic_results():
    rows = [dict(actuator_strength=s, evaluation_seed=100, episode_return=r)
            for s, r in [(1., 0), (.9, 1), (.7, -2)]]
    summary, _, analysis = summarize(rows)
    assert all(r["relative_return"] is None for r in summary)
    assert not analysis["monotonic_nondecreasing_with_strength"]


def test_actor_only_checkpoint_loading_does_not_touch_source(tmp_path, actor_state):
    path = tmp_path / "final.pt"
    torch.save({"schema_version": 1, "learner": actor_state,
                "config": {"seed": 0, "env": {"id": "HalfCheetah-v5"},
                           "dynamics": {"actuator_scale": 1.0}}, "counters": {}}, path)
    before = sha256(path)
    with patch("dynamics_shift.algorithms.sac.learner.SACLearner.__init__", side_effect=AssertionError("No learner")), \
         patch("torch.optim.Adam.__init__", side_effect=AssertionError("No optimizer")):
        policy, _ = load_frozen_actor(path, "cpu")
        expected = FrozenActor(actor_state, "cpu").act(np.zeros(17))
        np.testing.assert_array_equal(policy.act(np.zeros(17)), expected)
    assert sha256(path) == before


def test_actor_mutation_guard(actor_state):
    policy = FrozenActor(actor_state, "cpu")
    def corrupt(*args):
        with torch.no_grad():
            next(policy.actor.parameters()).add_(1)
        return []
    with patch("dynamics_shift.evaluation.severity_sweep.evaluate_policy", side_effect=corrupt):
        with pytest.raises(RuntimeError, match="Frozen actor changed"):
            collect_sweep(policy, SweepConfig(checkpoint="unused", actuator_strengths=(1.,), evaluation_seeds=(100,)))
