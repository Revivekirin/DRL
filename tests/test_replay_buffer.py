import numpy as np
import pytest
from dynamics_shift.data.replay_buffer import ReplayBuffer


def test_flags_shapes_and_defensive_storage():
    replay = ReplayBuffer(4, 3, 2, seed=0)
    obs = np.ones(3)
    replay.add(obs, np.zeros(2), 2, obs + 1, False, True)
    obs[:] = 99
    batch = replay.sample(8)
    assert batch.obs.shape == (8, 3)
    assert batch.reward.shape == (8, 1)
    assert not batch.terminated.any()
    assert batch.truncated.all()
    np.testing.assert_array_equal(batch.obs, np.ones((8, 3)))
    batch.obs[:] = 0
    assert replay.sample(1).obs[0, 0] == 1
    replay = ReplayBuffer(1, 3, 2, seed=0)
    replay.add(np.zeros(3), np.zeros(2), 0, np.zeros(3), True, False)
    assert replay.sample(1).terminated.all()
    assert not replay.sample(1).truncated.any()


def test_ring_and_validation():
    replay = ReplayBuffer(2, 1, 1, seed=0)
    with pytest.raises(ValueError):
        replay.sample(1)
    for i in range(5):
        replay.add([i], [0], i, [i + 1], False, False)
    assert len(replay) == 2
    assert set(replay.sample(100).obs.ravel()) == {3, 4}
    with pytest.raises(ValueError):
        replay.add([0, 1], [0], 0, [0], False, False)
    with pytest.raises(ValueError):
        replay.add([0], [0], 0, [0], 0.2, False)
