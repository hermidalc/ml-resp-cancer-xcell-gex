"""
The :mod:`sklearn.metrics` module includes score functions, performance metrics
and pairwise metrics and distance computations.
"""

from .scorer import check_scoring
from .scorer import make_scorer
from .scorer import SCORERS
from .scorer import get_scorer


__all__ = ['check_scoring',
           'get_scorer',
           'make_scorer',
           'SCORERS']
