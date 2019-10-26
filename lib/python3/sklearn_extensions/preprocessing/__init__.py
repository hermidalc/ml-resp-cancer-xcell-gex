"""
The :mod:`sklearn.preprocessing` module includes scaling, centering,
normalization, binarization methods.
"""

from .custom_data import DESeq2RLEVST, EdgeRTMMLogCPM, LimmaRemoveBatchEffect


__all__ = ['DESeq2RLEVST',
           'EdgeRTMMLogCPM',
           'LimmaRemoveBatchEffect']
