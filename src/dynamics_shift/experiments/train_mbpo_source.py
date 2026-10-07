"""Compatibility import; implementation moved to mbpo_mujoco."""
import sys
from . import mbpo_mujoco as _implementation
sys.modules[__name__] = _implementation
