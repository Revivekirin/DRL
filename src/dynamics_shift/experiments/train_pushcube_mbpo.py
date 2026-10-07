"""Compatibility import; implementation moved to mbpo_maniskill."""
import sys
from . import mbpo_maniskill as _implementation
sys.modules[__name__] = _implementation
