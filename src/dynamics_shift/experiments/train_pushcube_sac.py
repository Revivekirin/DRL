"""Compatibility import; implementation moved to sac_maniskill."""
import sys
from . import sac_maniskill as _implementation
sys.modules[__name__] = _implementation
