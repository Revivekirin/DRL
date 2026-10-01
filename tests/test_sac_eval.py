from copy import deepcopy
import numpy as np
import pytest
import torch
from dynamics_shift.algorithms.sac import SACConfig, SACLearner
from dynamics_shift.config import ExperimentConfig, EnvConfig
from dynamics_shift.envs import make_env
from dynamics_shift.data.replay_buffer import TransitionBatch
from dynamics_shift.evaluation.policy_eval import assert_matching_contract, evaluate_shift
from test_sac_checkpoint import assert_nested_equal


@pytest.mark.training
def test_frozen_deterministic_paired_evaluation():
    learner = SACLearner(17, -np.ones(6), np.ones(6), SACConfig(hidden_dims=(16,)))
    # Populate optimizer state before proving that evaluation leaves it untouched.
    learner.update(TransitionBatch(np.ones((4, 17)), np.zeros((4, 6)), np.ones((4, 1)),
                                   np.ones((4, 17)), np.zeros((4, 1), bool), np.zeros((4, 1), bool)))
    before = deepcopy(learner.state_dict())
    rng = torch.get_rng_state().clone()
    rows, summary = evaluate_shift(learner, EnvConfig(), (123,), 0.7)
    repeated, again = evaluate_shift(learner, EnvConfig(), (123,), 0.7)
    assert rows == repeated
    assert summary == again
    assert_nested_equal(before, learner.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert learner.actor.training
    assert [row["condition"] for row in rows] == ["source", "target"]
    assert all(row["episode_length"] == 1000 and row["truncated"] and not row["terminated"] for row in rows)
    assert summary["delta_return"] == summary["target"]["mean_return"] - summary["source"]["mean_return"]


def test_reward_contract_mismatch_rejected():
    source, target = make_env(ExperimentConfig()), make_env(ExperimentConfig())
    try:
        target.unwrapped._ctrl_cost_weight = 0.2
        with pytest.raises(ValueError, match="_ctrl_cost_weight"):
            assert_matching_contract(source, target)
    finally:
        source.close()
        target.close()
