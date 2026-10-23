"""Compatibility shim; the Metal implementation lives in ``tcenv.backends.metal.testing``."""

from .backends.metal.testing import *  # noqa: F401,F403

from .backends.metal.testing import _floor  # noqa: F401,E402
