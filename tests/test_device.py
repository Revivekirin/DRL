"""Remote device validation tests; CUDA tests skip on hosts without a GPU."""
from unittest.mock import patch
import numpy as np
import pytest
import torch
from dynamics_shift.utils.device import check_device
from dynamics_shift.algorithms.sac import SACConfig, SACLearner
from dynamics_shift.data.replay_buffer import TransitionBatch
from dynamics_shift.utils.checkpoint import save_checkpoint, load_checkpoint


def test_cuda_unavailable_fails_without_fallback():
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="No CPU fallback"):
            check_device("cuda:0")


def test_cuda_kernel_failure_is_reported():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.device", side_effect=RuntimeError("driver failure")):
        with pytest.raises(RuntimeError, match="driver failure"):
            check_device("cuda:0")


def test_cuda_update_checkpoint_roundtrip(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA server")
    device = check_device("cuda:0")
    learner = SACLearner(3, -np.ones(2), np.ones(2), SACConfig(hidden_dims=(16,)), device=device)
    batch = TransitionBatch(np.ones((4, 3)), np.zeros((4, 2)), np.ones((4, 1)),
                            np.ones((4, 3)), np.zeros((4, 1), bool), np.zeros((4, 1), bool))
    assert all(p.is_cuda for module in (learner.actor, learner.critic, learner.target_critic)
               for p in module.parameters())
    assert learner.log_alpha.is_cuda
    assert all(np.isfinite(value) for value in learner.update(batch).values())
    expected = learner.act(np.ones(3), deterministic=True)
    path = tmp_path / "gpu.pt"
    save_checkpoint(path, learner, dict(real_env_steps=4, policy_gradient_steps=1, episodes=0), {})
    restored, payload = load_checkpoint(path, device=device)
    assert payload["rng"]["torch_cuda"] is not None
    np.testing.assert_array_equal(expected, restored.act(np.ones(3), deterministic=True))
    assert all(np.isfinite(value) for value in restored.update(batch).values())
    cpu, _ = load_checkpoint(path, device="cpu")
    np.testing.assert_allclose(expected, cpu.act(np.ones(3), deterministic=True), atol=1e-5, rtol=1e-5)
