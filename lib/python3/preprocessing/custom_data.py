import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted, check_memory

r_base = importr('base')
r_base.source(sys.path[0] + '/lib/R/functions.R')
r_deseq2_vst_fit = robjects.globalenv['deseq2_vst_fit']
r_deseq2_vst_transform = robjects.globalenv['deseq2_vst_transform']
r_edger_tmm_logcpm_fit = robjects.globalenv['edger_tmm_logcpm_fit']
r_edger_tmm_logcpm_transform = robjects.globalenv['edger_tmm_logcpm_transform']
r_limma_remove_ba_fit = robjects.globalenv['limma_remove_ba_fit']
r_limma_remove_ba_transform = robjects.globalenv['limma_remove_ba_transform']
numpy2ri.activate()


def deseq2_vst_fit(X, y, y_meta, blind, fit_type, model_batch):
    xt, gm, sf, df = r_deseq2_vst_fit(X, y, y_meta=y_meta, blind=blind,
                                      fit_type=fit_type,
                                      model_batch=model_batch)
    return (np.array(xt, dtype=float), np.array(gm, dtype=float),
            np.array(sf, dtype=float), df)


def deseq2_vst_transform(X, geo_means, size_factors, disp_func, fit_type):
    return np.array(r_deseq2_vst_transform(
        X, geo_means=geo_means, size_factors=size_factors, disp_func=disp_func,
        fit_type=fit_type), dtype=float)


def edger_tmm_logcpm_fit(X, prior_count):
    return np.array(r_edger_tmm_logcpm_fit(X, prior_count=prior_count),
                    dtype=float)


def edger_tmm_logcpm_transform(X, ref_sample, prior_count):
    return np.array(r_edger_tmm_logcpm_transform(
        X, ref_sample=ref_sample, prior_count=prior_count), dtype=float)


class DESeq2RLEVST(TransformerMixin, BaseEstimator):
    """DESeq2 median-of-ratios normalization and VST transformation for count
    data

    Parameters
    ----------
    blind : bool (default = False)
        varianceStabilizingTransformation blind option.

    fit_type : str (default = "local")
        estimateDispersions and varianceStabilizingTransformation fitType
        option.

    model_batch : bool (default = False)
        Model batch effect if y_meta passed to fit and Batch column exists.

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

    def __init__(self, blind=False, fit_type='local', model_batch=False,
                 memory=None):
        self.blind = blind
        self.fit_type = fit_type
        self.model_batch = model_batch
        self.memory = memory

    def fit(self, X, y, y_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        y_meta : array-like, list, pandas.DataFrame, pandas.Series, \
            (default = None), shape = (n_samples, n_metadata)
            Target metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=int)
        memory = check_memory(self.memory)
        if y_meta is None:
            y_meta = robjects.NULL
        (self._vst_data, self.geo_means_, self.size_factors_,
         self.disp_func_) = memory.cache(deseq2_vst_fit)(
             X, y, y_meta=y_meta, blind=self.blind, fit_type=self.fit_type,
             model_batch=self.model_batch)
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            DESeq2 median-of-ratios normalized VST transformed counts data
            matrix.
        """
        check_is_fitted(self, '_vst_data')
        X = check_array(X, dtype=int)
        if hasattr(self, '_train_done'):
            memory = check_memory(self.memory)
            X = memory.cache(deseq2_vst_transform)(
                X, geo_means=self.geo_means_, size_factors=self.size_factors_,
                disp_func=self.disp_func_, fit_type=self.fit_type)
        else:
            X = self._vst_data
            self._train_done = True
        return X

    def inverse_transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError("inverse_transform not implemented.")


class EdgeRTMMLogCPM(TransformerMixin, BaseEstimator):
    """edgeR TMM normalization and log-CPM transformation for count data

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
            Input counts data matrix.

        y : ignored
        """
        X = check_array(X, dtype=int)
        memory = check_memory(self.memory)
        self._log_cpms, self.ref_sample_ = memory.cache(edger_tmm_logcpm_fit)(
            X, prior_count=self.prior_count)
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            edgeR TMM normalized log-CPM transformed counts data matrix.
        """
        check_is_fitted(self, '_log_cpms')
        X = check_array(X, dtype=int)
        if hasattr(self, '_train_done'):
            memory = check_memory(self.memory)
            X = memory.cache(edger_tmm_logcpm_transform)(
                X, ref_sample=self.ref_sample_, prior_count=self.prior_count)
        else:
            X = self._log_cpms
            self._train_done = True
        return X

    def inverse_transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError("inverse_transform not implemented.")


class LimmaRemoveBatchEffectTransformer(TransformerMixin, BaseEstimator):
    """limma removeBatchEffect transformer for log-transformed expression data

    Parameters
    ----------
    preserve_design : bool (default = True)
        Whether batch effect correction should protect target design from
        being removed.

    Attributes
    ----------
    beta_ : array, shape (n_features, n_batches - 1)
        removeBatchEffect linear model coefficents
    """

    def __init__(self, preserve_design=True):
        self.preserve_design = preserve_design

    def fit(self, X, y=None, y_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input log-transformed data matrix.

        y : ignored

        y_meta : array-like, list, pandas.DataFrame, pandas.Series, \
            shape = (n_samples, n_metadata)
            Target metadata with Batch and Class columns.
        """
        X = check_array(X)
        if y_meta is None:
            y_meta = robjects.NULL
            self.batch_ = None
        self.beta_ = np.array(r_limma_remove_ba_fit(
            X, y_meta, preserve_design=self.preserve_design), dtype=float)
        return self

    def transform(self, X, y_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input log-transformed data matrix.

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            Batched corrected log-transformed input data matrix.
        """
        check_is_fitted(self, 'beta_')
        X = check_array(X)
        X = np.array(r_limma_remove_ba_transform(X, y_meta, beta=self.beta_),
                     dtype=float)
        return X

    def inverse_transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input batched corrected log-transformed data matrix.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError("inverse_transform not implemented.")
