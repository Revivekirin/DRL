"""Backend-independent frozen learner checks and global RNG isolation."""
from copy import deepcopy
from functools import wraps
import numpy as np
import torch
from dynamics_shift.utils.checkpoint import isolated_rng

def assert_state_equal(before, after):
    """Include optimizer, temperature, targets and counters, not just actor weights."""
    if isinstance(before, torch.Tensor):
        equal = torch.equal(before, after)
    elif isinstance(before, np.ndarray):
        equal = np.array_equal(before, after)
    elif isinstance(before, dict):
        equal = before.keys() == after.keys()
        if equal:
            for key in before:
                assert_state_equal(before[key], after[key])
    elif isinstance(before, (tuple, list)):
        equal = len(before) == len(after)
        if equal:
            for a, b in zip(before, after):
                assert_state_equal(a, b)
    else:
        equal = before == after
    if not equal:
        raise RuntimeError("Evaluation changed learner state")



def frozen_learner(function):
    @wraps(function)
    def guarded(learner, *args, **kwargs):
        before = deepcopy(learner.state_dict())
        with isolated_rng(learner.device):
            try:
                return function(learner, *args, **kwargs)
            finally:
                assert_state_equal(before, learner.state_dict())
    return guarded
