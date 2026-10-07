"""Legacy module alias for existing evaluation integrations."""
import sys
from . import maniskill
sys.modules[__name__] = maniskill
