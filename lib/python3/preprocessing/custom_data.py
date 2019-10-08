import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted, check_memory

r_base = importr('base')
r_base.source('functions.R')
r_deseq2_vst_transform = robjects.globalenv['deseq2_vst_transform']
r_edger_tmm_logcpm_transform = robjects.globalenv['edger_tmm_logcpm_transform']
numpy2ri.activate()

def deseq2_vst_transform(X, y, geo_means, size_factors, disp_func, blind,
                         fit_type):
    if y is not None:
        xt, gm, sf, df = r_deseq2_vst_transform(
            X, y, blind=blind, fit_type=fit_type)
    else:
        xt, gm, sf, df = r_deseq2_vst_transform(
            X, geo_means=geo_means, size_factors=size_factors,
            disp_func=disp_func, blind=blind, fit_type=fit_type)
    return (np.array(xt, dtype=float), np.array(gm, dtype=float),
            np.array(sf, dtype=float), df)

def edger_tmm_logcpm_transform(X, ref_sample, prior_count):
    xt, rs = r_edger_tmm_logcpm_transform(X, ref_sample=ref_sample,
                                          prior_count=prior_count)
    return np.array(xt, dtype=float), np.array(rs, dtype=float)


class DESeq2RLEVSTransformer(TransformerMixin, BaseEstimator):
    """DESeq2 median-of-ratios normalization and VST transformation

    Parameters
    ----------
    blind : bool (default = False)
        varianceStabilizingTransformation() blind option

    fit_type : str (default = local)
        estimateDispersions() fitType option

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    geo_means_ : array, shape (n_features,)
        Feature geometric means.

    size_factors_ : array, shape (n_features,)
        RLE normalization size factors.

    disp_func_ : R/rpy2 function
        RLE normalization dispersion function.
    """
    def __init__(self, blind=False, fit_type='local', memory=None):
        self.blind = blind
        self.fit_type = fit_type
        self.memory = memory

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The input counts data matrix.

        y : array-like, shape = (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        """
        X, y = check_X_y(X, y, dtype=np.int64)
        memory = check_memory(self.memory)
        (self._vst_data, self.geo_means_, self.size_factors_,
         self.disp_func_) = (memory.cache(deseq2_vst_transform)(
                X, y, blind=self.blind, fit_type=self.fit_type))
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The input counts data matrix.

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            DESeq2 median-of-ratios normalized VST transformed input data
            matrix.
        """
        check_is_fitted(self, '_vst_data')
        X = check_array(X, dtype=np.int64)
        if hasattr(self, '_train_done'):
            memory = check_memory(self.memory)
            X = memory.cache(deseq2_vst_transform)(
                X, y=None, geo_means=self.geo_means_,
                size_factors=self.size_factors_, disp_func=self.disp_func_,
                blind=self.blind, fit_type=self.fit_type)[0]
        else:
            X = self._vst_data
            self._train_done = True
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
        """
        raise NotImplementedError("inverse_transform not implemented.")


class EdgeRTMMLogCPMTransformer(TransformerMixin, BaseEstimator):
    """edgeR TMM normalization and log-CPM transformation for RNA-seq data

    Parameters
    ----------
    prior_count : int (default = 1)
        Average count to add to each observation to avoid taking log of zero.
        Larger values for produce stronger moderation of the values for low
        counts and more shrinkage of the corresponding log fold changes.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    ref_sample_ : array, shape (n_features,)
        TMM normalization reference sample feature vector.
    """
    def __init__(self, prior_count=1, memory=None):
        self.prior_count = prior_count
        self.memory = memory

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The input counts data matrix.

        y : ignored
        """
        X = check_array(X, dtype=np.int64)
        memory = check_memory(self.memory)
        self._log_cpms, self.ref_sample_ = (
            memory.cache(edger_tmm_logcpm_transform)(
                X, prior_count=self.prior_count))
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The input counts data matrix.

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            edgeR TMM normalized log-CPM transformed input data matrix.
        """
        check_is_fitted(self, '_log_cpms')
        X = check_array(X, dtype=np.int64)
        if hasattr(self, '_train_done'):
            memory = check_memory(self.memory)
            X = memory.cache(edger_tmm_logcpm_transform)(
                X, ref_sample=self.ref_sample_,
                prior_count=self.prior_count)[0]
        else:
            X = self._log_cpms
            self._train_done = True
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
        """
        raise NotImplementedError("inverse_transform not implemented.")
