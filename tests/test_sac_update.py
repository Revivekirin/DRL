from copy import deepcopy
import math
from unittest.mock import patch
import numpy as np
import torch
from dynamics_shift.algorithms.sac import SACConfig, SACLearner
from dynamics_shift.algorithms.sac.networks import GaussianActor
from dynamics_shift.data.replay_buffer import TransitionBatch


def make_learner():
    torch.manual_seed(7)
    return SACLearner(3, np.array([-2., 0.]), np.array([2., 3.]), SACConfig(hidden_dims=(16, 16)))


def make_batch():
    rng = np.random.default_rng(3)
    return TransitionBatch(rng.normal(size=(16, 3)), rng.uniform(0, 1, (16, 2)),
                           rng.normal(size=(16, 1)), rng.normal(size=(16, 3)),
                           np.zeros((16, 1), bool), np.ones((16, 1), bool))


def test_updates_all_parameters_and_polyak():
    learner = make_learner()
    old_actor = deepcopy(learner.actor.state_dict())
    old_critic = deepcopy(learner.critic.state_dict())
    old_target = deepcopy(learner.target_critic.state_dict())
    old_alpha = learner.log_alpha.detach().clone()
    metrics = learner.update(make_batch())
    assert all(math.isfinite(v) for v in metrics.values())
    assert learner.policy_gradient_steps == 1
    assert any(not torch.equal(old_actor[k], v) for k, v in learner.actor.state_dict().items())
    assert any(not torch.equal(old_critic[k], v) for k, v in learner.critic.state_dict().items())
    assert not torch.equal(old_alpha, learner.log_alpha)
    for key, value in learner.target_critic.state_dict().items():
        expected = old_target[key] * (1 - learner.config.tau) + learner.critic.state_dict()[key] * learner.config.tau
        torch.testing.assert_close(value, expected)
    assert all(p.grad is None for p in learner.target_critic.parameters())


def test_terminal_mask_and_entropy_bellman_equation():
    learner = make_learner()
    with patch.object(learner.actor, "sample", return_value=(torch.zeros(2, 2), torch.full((2, 1), -2.))), \
         patch.object(learner.target_critic, "forward", return_value=(torch.full((2, 1), 3.), torch.full((2, 1), 5.))):
        result = learner.bellman_target(torch.ones(2, 1), torch.zeros(2, 3), torch.tensor([[1.], [0.]]))
    torch.testing.assert_close(result, torch.tensor([[1.], [1. + .99 * (3 + .2 * 2)]]))
    # Identical RNG and learners: changing only truncated must not alter any update.
    a, b = make_learner(), make_learner()
    batch = make_batch()
    torch.manual_seed(9)
    ma = a.update(batch)
    torch.manual_seed(9)
    mb = b.update(TransitionBatch(batch.obs, batch.action, batch.reward, batch.next_obs,
                                 batch.terminated, ~batch.truncated))
    assert ma == mb


def test_action_bounds_and_change_of_variables():
    actor = GaussianActor(3, np.array([-2., 1.]), np.array([4., 5.]), (16,))
    obs = torch.zeros(128, 3)
    actions, log_prob = actor.sample(obs)
    assert torch.all(actions >= torch.tensor([-2., 1.]))
    assert torch.all(actions <= torch.tensor([4., 5.]))
    # Independent inverse-transform calculation away from saturation.
    squashed = (actions - actor.action_bias) / actor.action_scale
    pre_tanh = torch.atanh(squashed)
    mean, log_std = actor(obs)
    expected = (torch.distributions.Normal(mean, log_std.exp()).log_prob(pre_tanh)
                - torch.log1p(-squashed.square()) - actor.action_scale.log()).sum(-1, keepdim=True)
    torch.testing.assert_close(log_prob, expected, atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        actor.net[-1].weight.zero_()
        actor.net[-1].bias[:2].fill_(100)
    _, saturated_log_prob = actor.sample(obs)
    assert torch.isfinite(saturated_log_prob).all()
