"""Compatibility alias for the common checkpoint evaluator."""
import sys
from . import evaluate as _implementation
sys.modules[__name__] = _implementation
