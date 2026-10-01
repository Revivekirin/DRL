import random
import numpy as np
import pytest
import torch
from dynamics_shift.algorithms.sac import SACConfig, SACLearner
from dynamics_shift.data.replay_buffer import TransitionBatch
from dynamics_shift.utils.checkpoint import load_checkpoint, save_checkpoint


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.training
def test_checkpoint_actions_optimizer_continuation_and_rng(tmp_path):
    torch.manual_seed(0)
    learner = SACLearner(3, -np.ones(2), np.ones(2), SACConfig(hidden_dims=(16,)))
    batch = TransitionBatch(np.ones((8, 3)), np.zeros((8, 2)), np.ones((8, 1)),
                            np.ones((8, 3)), np.zeros((8, 1), bool), np.ones((8, 1), bool))
    learner.update(batch)
    obs = np.array([0.1, 0.2, 0.3])
    expected_action = learner.act(obs, deterministic=True)
    path = tmp_path / "checkpoint.pt"
    counters = dict(real_env_steps=8, policy_gradient_steps=1, episodes=0)
    save_checkpoint(path, learner, counters, {"test": True})
    expected_random = (random.random(), np.random.rand(), torch.rand(2))
    torch_state = torch.get_rng_state().clone()
    loaded, payload = load_checkpoint(path)
    assert torch.equal(torch_state, torch.get_rng_state())
    np.testing.assert_array_equal(expected_action, loaded.act(obs, deterministic=True))
    assert_nested_equal(learner.state_dict(), loaded.state_dict())
    assert payload["counters"] == counters
    assert not payload["replay_persisted"]
    load_checkpoint(path, restore_random_state=True)
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    assert torch.equal(torch.rand(2), expected_random[2])
    torch.manual_seed(27)
    first = learner.update(batch)
    torch.manual_seed(27)
    second = loaded.update(batch)
    assert first == second
    assert_nested_equal(learner.state_dict(), loaded.state_dict())
    with pytest.raises(FileExistsError):
        save_checkpoint(path, learner, dict(counters, policy_gradient_steps=2), {})
