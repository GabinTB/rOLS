"""
rOLS
===========

Vectorized rolling time-series OLS (or Ridge) regression.

Public API
----------
RollingOLS      : main model class
RollingOLSResult: result container with getters
"""

from .model import RollingOLS
from .results import RollingOLSResult

__version__ = "0.4.0"
__all__ = ["RollingOLS", "RollingOLSResult", "__version__"]
