"""Compatibility import; implementation moved to sac_mujoco."""
import sys
from . import sac_mujoco as _implementation
sys.modules[__name__] = _implementation
