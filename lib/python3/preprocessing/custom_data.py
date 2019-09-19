import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

r_base = importr('base')
r_base.source('functions.R')
r_edger_tmm_logcpm_transform = robjects.globalenv['edger_tmm_logcpm_transform']
numpy2ri.activate()


class EdgeRTMMLogCPMTransformer(TransformerMixin, BaseEstimator):
    """EdgeR TMM normalization and log-CPM transformation for RNA-seq data.

    Parameters
    ----------
    prior_count : int (default = 1)
        Average count to be added to each observation to avoid taking log of
        zero. Larger values for prior.count produce stronger moderation of the
        values for low counts and more shrinkage of the corresponding log
        fold-changes.

    Attributes
    ----------
    ref_sample_ : array, shape (n_features,)
        TMM normalization reference sample feature vector.
    """
    def __init__(self, prior_count=1):
        self.prior_count = prior_count

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like
        """
        check_array(X)
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The input counts data matrix.

        Returns
        -------
        X_r : array of shape (n_samples, n_selected_features)
            The TMM normalized log-CPM transformed input data matrix.
        """
        X = check_array(X, dtype=None)
        if hasattr(self, 'ref_sample_'):
            X = np.array(r_edger_tmm_logcpm_transform(
                X, ref_sample=self.ref_sample_,
                prior_count=self.prior_count)[0],
                         dtype=float)
        else:
            xt, rs = r_edger_tmm_logcpm_transform(
                X, prior_count=self.prior_count)
            X = np.array(xt, dtype=float)
            self.ref_sample_ = np.array(rs, dtype=float)
        return X

    def inverse_transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The input data matrix.

        Returns
        -------
        X_r : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")
