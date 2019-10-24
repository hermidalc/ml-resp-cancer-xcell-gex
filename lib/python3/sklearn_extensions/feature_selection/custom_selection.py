import sys
import warnings
import numpy as np
import rpy2.robjects as robjects
from rpy2.rinterface import RRuntimeWarning
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted, check_memory
from .univariate_selection import BaseScorer

r_base = importr('base')
r_base.source(sys.path[0] + '/lib/R/functions.R')
r_deseq2_vst_transform = robjects.globalenv['deseq2_vst_transform']
r_deseq2_feature_score = robjects.globalenv['deseq2_feature_score']
r_edger_filterbyexpr_mask = robjects.globalenv['edger_filterbyexpr_mask']
r_edger_tmm_logcpm_transform = robjects.globalenv['edger_tmm_logcpm_transform']
r_edger_feature_score = robjects.globalenv['edger_feature_score']
r_limma_voom_feature_score = robjects.globalenv['limma_voom_feature_score']
r_dream_voom_feature_score = robjects.globalenv['dream_voom_feature_score']
r_limma_feature_score = robjects.globalenv['limma_feature_score']
r_cfs_feature_idxs = robjects.globalenv['cfs_feature_idxs']
r_fcbf_feature_idxs = robjects.globalenv['fcbf_feature_idxs']
r_relieff_feature_score = robjects.globalenv['relieff_feature_score']
numpy2ri.activate()


def deseq2_vst_transform(X, geo_means, size_factors, disp_func, fit_type):
    return np.array(r_deseq2_vst_transform(
        X, geo_means=geo_means, size_factors=size_factors, disp_func=disp_func,
        fit_type=fit_type), dtype=float)


def deseq2_feature_score(X, y, y_meta, lfc, blind, fit_type, model_batch,
                         n_threads):
    sv, xt, gm, sf, df = r_deseq2_feature_score(
        X, y, y_meta=y_meta, lfc=lfc, blind=blind, fit_type=fit_type,
        model_batch=model_batch, n_threads=n_threads)
    return (np.array(sv, dtype=float), np.array(xt, dtype=float),
            np.array(gm, dtype=float), np.array(sf, dtype=float), df)


def edger_tmm_logcpm_transform(X, ref_sample, prior_count):
    return np.array(r_edger_tmm_logcpm_transform(
        X, ref_sample=ref_sample, prior_count=prior_count), dtype=float)


def edger_feature_score(X, y, y_meta, lfc, robust, prior_count, model_batch):
    pv, pa, xt, rs = r_edger_feature_score(
        X, y, y_meta=y_meta, lfc=lfc, robust=robust, prior_count=prior_count,
        model_batch=model_batch)
    return (np.array(pv, dtype=float), np.array(pa, dtype=float),
            np.array(xt, dtype=float), np.array(rs, dtype=float))


def limma_voom_feature_score(X, y, y_meta, lfc, robust, prior_count,
                             model_batch, model_dupcor):
    pv, pa, xt, rs = r_limma_voom_feature_score(
        X, y, y_meta=y_meta, lfc=lfc, robust=robust, prior_count=prior_count,
        model_batch=model_batch, model_dupcor=model_dupcor)
    return (np.array(pv, dtype=float), np.array(pa, dtype=float),
            np.array(xt, dtype=float), np.array(rs, dtype=float))


def dream_voom_feature_score(X, y, y_meta, lfc, prior_count, model_batch,
                             n_threads):
    pv, pa, xt, rs = r_dream_voom_feature_score(
        X, y, y_meta, lfc=lfc, prior_count=prior_count,
        model_batch=model_batch, n_threads=n_threads)
    return (np.array(pv, dtype=float), np.array(pa, dtype=float),
            np.array(xt, dtype=float), np.array(rs, dtype=float))


def fcbf_feature_idxs(X, y, threshold):
    idxs, scores = r_fcbf_feature_idxs(X, y, threshold=threshold)
    return np.array(idxs, dtype=int), np.array(scores, dtype=float)


def relieff_feature_score(X, y, num_neighbors, sample_size):
    return np.array(r_relieff_feature_score(
        X, y, num_neighbors=num_neighbors, sample_size=sample_size),
                    dtype=float)


class DESeq2(BaseEstimator, SelectorMixin):
    """DESeq2 differential expression feature selector and
    normalizer/transformer for RNA-seq count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. The "all" option bypasses selection,
        for use in a parameter search.

    fc : float (default = 1.0)
        lfcShrink absolute fold change minimum threshold.

    blind : bool (default = False)
        varianceStabilizingTransformation blind option.

    fit_type : str (default = "local")
        estimateDispersions and varianceStabilizingTransformation fitType
        option.

    model_batch : bool (default = False)
        Model batch effect if y_meta passed to fit and Batch column exists.

    n_threads : int (default = 1)
        Number of DESeq2 parallel threads. This should be carefully selected
        when using within Grid/RandomizedSearchCV to not oversubscribe CPU
        and memory resources.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    svals_ : array, shape (n_features,)
        Feature s-values.

    geo_means_ : array, shape (n_features,)
        Feature geometric means.

    size_factors_ : array, shape (n_features,)
        RLE normalization size factors.

    disp_func_ : R/rpy2 function
        RLE normalization dispersion function.
    """

    def __init__(self, k='all', fc=1, blind=False, fit_type='local',
                 model_batch=False, n_threads=1, memory=None):
        self.k = k
        self.fc = fc
        self.blind = blind
        self.fit_type = fit_type
        self.model_batch = model_batch
        self.n_threads = n_threads
        self.memory = memory

    def fit(self, X, y, y_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training sample class labels.

        y_meta : array-like, list, pandas.DataFrame, pandas.Series, \
            (default = None), shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=int)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        if y_meta is None:
            y_meta = robjects.NULL
        (self.svals_, self._vst_data, self.geo_means_, self.size_factors_,
         self.disp_func_) = memory.cache(deseq2_feature_score)(
             X, y, y_meta=y_meta, lfc=np.log2(self.fc), blind=self.blind,
             fit_type=self.fit_type, model_batch=self.model_batch,
             n_threads=self.n_threads)
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            DESeq2 median-of-ratios normalized VST transformed counts data
            matrix with only the selected features.
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
        return super().transform(X)

    def inverse_transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features."
                % self.k)
        if self.fc < 1:
            raise ValueError(
                "fold change threshold should be >= 1; got %r." % self.fc)

    def _get_support_mask(self):
        check_is_fitted(self, 'svals_')
        mask = np.zeros_like(self.svals_, dtype=bool)
        if self.k == 'all':
            mask = np.ones_like(self.svals_, dtype=bool)
        elif self.k > 0:
            mask[np.argsort(self.svals_, kind='mergesort')[:self.k]] = True
        return mask


class EdgeR(BaseEstimator, SelectorMixin):
    """edgeR differential expression feature selector and
    normalizer/transformer for RNA-seq count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. The "all" option bypasses selection,
        for use in a parameter search.

    fc : float (default = 1.0)
        glmTreat absolute fold change minimum threshold.  Default value of 1.0
        gives glmQLFTest results.

    robust : bool (default = True)
        estimateDisp and glmQLFit robust option.

    prior_count : int (default = 1)
        Average count to add to each observation to avoid taking log of zero.
        Larger values for produce stronger moderation of the values for low
        counts and more shrinkage of the corresponding log fold changes.

    model_batch : bool (default = False)
        Model batch effect if y_meta passed to fit and Batch column exists.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    pvals_ : array, shape (n_features,)
        Feature raw p-values.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

    ref_sample_ : array, shape (n_features,)
        TMM normalization reference sample feature vector.
    """

    def __init__(self, k='all', fc=1, robust=True, prior_count=1,
                 model_batch=False, memory=None):
        self.k = k
        self.fc = fc
        self.robust = robust
        self.prior_count = prior_count
        self.model_batch = model_batch
        self.memory = memory

    def fit(self, X, y, y_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training sample class labels.

        y_meta : array-like, list, pandas.DataFrame, pandas.Series, \
            (default = None), shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=int)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        if y_meta is None:
            y_meta = robjects.NULL
        self.pvals_, self.padjs_, self._log_cpms, self.ref_sample_ = (
            memory.cache(edger_feature_score)(
                X, y, y_meta=y_meta, lfc=np.log2(self.fc), robust=self.robust,
                prior_count=self.prior_count, model_batch=self.model_batch))
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            edgeR TMM normalized log-CPM transformed counts data matrix with
            only the selected features.
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
        return super().transform(X)

    def inverse_transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features."
                % self.k)
        if self.fc < 1:
            raise ValueError(
                "fold change threshold should be >= 1; got %r." % self.fc)

    def _get_support_mask(self):
        check_is_fitted(self, 'pvals_')
        mask = np.zeros_like(self.pvals_, dtype=bool)
        if self.k == 'all':
            mask = np.ones_like(self.pvals_, dtype=bool)
        elif self.k > 0:
            mask[np.argsort(self.pvals_, kind='mergesort')[:self.k]] = True
        return mask


class EdgeRFilterByExpr(BaseEstimator, SelectorMixin):
    """edgeR filterByExpr feature selector for count data

    Parameters
    ----------
    model_batch : bool (default = False)
        Model batch effect if y_meta passed to fit and Batch column exists.
    """

    def __init__(self, model_batch=False):
        self.model_batch = model_batch

    def fit(self, X, y, y_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training sample class labels.

        y_meta : array-like, list, pandas.DataFrame, pandas.Series, \
            (default = None), shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=int)
        if y_meta is None:
            y_meta = robjects.NULL
        self._mask = np.array(r_edger_filterbyexpr_mask(
            X, y, y_meta=y_meta, model_batch=self.model_batch), dtype=bool)
        return self

    def _get_support_mask(self):
        check_is_fitted(self, '_mask')
        return self._mask


class LimmaVoom(BaseEstimator, SelectorMixin):
    """limma-voom differential expression feature selector and
    normalizer/transformer for RNA-seq count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. The "all" option bypasses selection,
        for use in a parameter search.

    fc : float (default = 1.0)
        treat absolute fold change minimum threshold.  Default value of 1.0
        gives eBayes results.

    robust : bool (default = True)
        limma treat/eBayes robust option.

    prior_count : int (default = 1)
        Average count to add to each observation to avoid taking log of zero.
        Larger values for produce stronger moderation of the values for low
        counts and more shrinkage of the corresponding log fold changes.

    model_batch : bool (default = False)
        Model batch effect if y_meta passed to fit and Batch column exists.

    model_dupcor : bool (default = False)
        Model limma duplicateCorrelation if y_meta passed to fit and Group
        column exists.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    pvals_ : array, shape (n_features,)
        Feature raw p-values.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

    ref_sample_ : array, shape (n_features,)
        TMM normalization reference sample feature vector.
    """

    def __init__(self, k='all', fc=1, robust=True, prior_count=1,
                 model_batch=False, model_dupcor=False, memory=None):
        self.k = k
        self.fc = fc
        self.robust = robust
        self.prior_count = prior_count
        self.model_batch = model_batch
        self.model_dupcor = model_dupcor
        self.memory = memory

    def fit(self, X, y, y_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training sample class labels.

        y_meta : array-like, list, pandas.DataFrame, pandas.Series, \
            (default = None), shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=int)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        if y_meta is None:
            y_meta = robjects.NULL
        self.pvals_, self.padjs_, self._log_cpms, self.ref_sample_ = (
            memory.cache(limma_voom_feature_score)(
                X, y, y_meta=y_meta, lfc=np.log2(self.fc), robust=self.robust,
                prior_count=self.prior_count, model_batch=self.model_batch,
                model_dupcor=self.model_dupcor))
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            edgeR TMM normalized log-CPM transformed counts data matrix with
            only the selected features.
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
        return super().transform(X)

    def inverse_transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features."
                % self.k)
        if self.fc < 1:
            raise ValueError(
                "fold change threshold should be >= 1; got %r." % self.fc)

    def _get_support_mask(self):
        check_is_fitted(self, 'pvals_')
        mask = np.zeros_like(self.pvals_, dtype=bool)
        if self.k == 'all':
            mask = np.ones_like(self.pvals_, dtype=bool)
        elif self.k > 0:
            mask[np.argsort(self.pvals_, kind='mergesort')[:self.k]] = True
        return mask


class DreamVoom(BaseEstimator, SelectorMixin):
    """dream limma-voom differential expression feature selector and
    normalizer/transformer for RNA-seq count data repeated measures designs

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. The "all" option bypasses selection,
        for use in a parameter search.

    fc : float (default = 1.0)
        Absolute fold-change minimum threshold.

    prior_count : int (default = 1)
        Average count to add to each observation to avoid taking log of zero.
        Larger values for produce stronger moderation of the values for low
        counts and more shrinkage of the corresponding log fold changes.

    model_batch : bool (default = False)
        Model batch effect if y_meta passed to fit and Batch column exists.

    n_threads : int (default = 1)
        Number of dream parallel threads. This should be carefully selected
        when using within Grid/RandomizedSearchCV to not oversubscribe CPU
        and memory resources.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    pvals_ : array, shape (n_features,)
        Feature raw p-values.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

    ref_sample_ : array, shape (n_features,)
        TMM normalization reference sample feature vector.
    """

    def __init__(self, k='all', fc=1, prior_count=1, model_batch=False,
                 n_threads=1, memory=None):
        self.k = k
        self.fc = fc
        self.prior_count = prior_count
        self.model_batch = model_batch
        self.n_threads = n_threads
        self.memory = memory

    def fit(self, X, y, y_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training sample class labels.

        y_meta : array-like, list, pandas.DataFrame, pandas.Series, \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=int)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        self.pvals_, self.padjs_, self._log_cpms, self.ref_sample_ = (
            memory.cache(dream_voom_feature_score)(
                X, y, y_meta, lfc=np.log2(self.fc),
                prior_count=self.prior_count, model_batch=self.model_batch,
                n_threads=self.n_threads))
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            edgeR TMM normalized log-CPM transformed counts data matrix with
            only the selected features.
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
        return super().transform(X)

    def inverse_transform(self, X):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features."
                % self.k)
        if self.fc < 1:
            raise ValueError(
                "fold change threshold should be >= 1; got %r." % self.fc)

    def _get_support_mask(self):
        check_is_fitted(self, 'pvals_')
        mask = np.zeros_like(self.pvals_, dtype=bool)
        if self.k == 'all':
            mask = np.ones_like(self.pvals_, dtype=bool)
        elif self.k > 0:
            mask[np.argsort(self.pvals_, kind='mergesort')[:self.k]] = True
        return mask


class LimmaScorerClassification(BaseScorer):
    """limma differential expression feature scorer for classification tasks

    Parameters
    ----------
    fc : float (default = 1.0)
        treat absolute fold change minimum threshold.  Default value of 1.0
        gives eBayes results.

    robust : bool (default = False)
        limma treat/eBayes robust option

    trend : bool (default = False)
        limma treat/eBayes trend option

    model_batch : bool (default = False)
        Model batch effect if y_meta passed to fit and Batch column exists

    Attributes
    ----------
    scores_ : array, shape (n_features,)
        Feature F values.

    pvalues_ : array, shape (n_features,)
        Feature adjusted p-values.
    """

    def __init__(self, fc=1, robust=False, trend=False, model_batch=False):
        self.fc = fc
        self.robust = robust
        self.trend = trend
        self.model_batch = model_batch

    def fit(self, X, y, y_meta=None):
        """
        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training feature data matrix.

        y : array_like, shape (n_samples,)
            Training sample class labels.

        y_meta : array-like, list, pandas.DataFrame, pandas.Series, \
            (default = None), shape = (n_samples, n_metadata)
            Target sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        self._check_params(X, y)
        if y_meta is None:
            y_meta = robjects.NULL
        pv, pa = r_limma_feature_score(
            X, y, y_meta=y_meta, lfc=np.log2(self.fc), robust=self.robust,
            trend=self.trend, model_batch=self.model_batch)
        # convert to scores for sklearn Select* feature selectors
        self.scores_ = np.reciprocal(np.array(pv, dtype=float))
        self.pvalues_ = np.array(pa, dtype=float)
        return self

    def _check_params(self, X, y):
        if self.fc < 1:
            raise ValueError(
                "fold change threshold should be >= 1; got %r." % self.fc)


class ColumnSelector(BaseEstimator, SelectorMixin):
    """Column feature selector

    Parameters
    ----------
    cols : array-like (default = None)
        A list specifying the feature indices to be selected. For example,
        [1, 4, 5] to select the 2nd, 5th, and 6th feature columns, and
        ['A','C','D'] to select the name of feature columns A, C and D.
        If None, returns all columns in the array.
    """

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training input data matrix.

        y : array-like, shape = (n_samples)
            Target values (class labels in classification, real numbers in
            regression).

        Returns
        ---------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=None)
        self._check_params(X, y)
        if self.cols is None:
            mask = np.ones(X.shape[1], dtype=bool)
        elif hasattr(X, 'iloc') and isinstance(self.cols[0], str):
            mask = X.columns.isin(self.cols)
        else:
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[list(self.cols)] = True
        self._mask = mask
        return self

    def _check_params(self, X, y):
        if self.cols is not None:
            types = {type(i) for i in self.cols}
            if len(types) > 1:
                raise ValueError("cols should be all of the same data type.")
            if hasattr(X, 'iloc') and isinstance(self.cols[0], str):
                for col in self.cols:
                    if col not in X:
                        raise ValueError("'%r' column does not exist." % col)
            else:
                for col in self.cols:
                    if not 0 <= col <= X.shape[1]:
                        raise ValueError(
                            "cols should be 0 <= col <= n_features; got %r."
                            "Use cols=None to return all features."
                            % col)

    def _get_support_mask(self):
        check_is_fitted(self, '_mask')
        return self._mask


class CFS(BaseEstimator, SelectorMixin):
    """Feature selector using Correlation Feature Selection (CFS) algorithm

    Attributes
    ----------
    selected_idxs_ : array-like, 1d
        CFS selected feature indexes
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training input data matrix.

        y : array-like, shape = (n_samples)
            Target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=None)
        self._n_features = X.shape[1]
        warnings.filterwarnings('ignore', category=RRuntimeWarning,
                                message="^Rjava\.init\.warning")
        self.selected_idxs_ = np.array(r_cfs_feature_idxs(X, y), dtype=int)
        warnings.filterwarnings('always', category=RRuntimeWarning)
        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'selected_idxs_')
        mask = np.zeros(self._n_features, dtype=bool)
        mask[self.selected_idxs_] = True
        return mask


class FCBF(BaseEstimator, SelectorMixin):
    """Feature selector using Fast Correlation-Based Filter (FCBF) algorithm

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. The "all" option bypasses selection,
        for use in a parameter search. If k is specified threshold is ignored.

    threshold : float (default = 0)
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    selected_idxs_ : array-like, 1d
        FCBF selected feature indexes
    """

    def __init__(self, k='all', threshold=0, memory=None):
        self.k = k
        self.threshold = threshold
        self.memory = memory
        self.selected_idxs_ = np.array([], dtype=int)
        self.scores_ = np.array([], dtype=float)

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training input data matrix.

        y : array-like, shape = (n_samples)
            Target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=None)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        self._n_features = X.shape[1]
        if self.k == 'all' or self.k > 0:
            warnings.filterwarnings('ignore', category=RRuntimeWarning,
                                    message="^Rjava\.init\.warning")
            feature_idxs, scores = memory.cache(fcbf_feature_idxs)(
                X, y, threshold=self.threshold)
            warnings.filterwarnings('always', category=RRuntimeWarning)
            if self.k != 'all':
                feature_idxs = feature_idxs[
                    np.argsort(scores, kind='mergesort')[-self.k:]]
                scores = np.sort(scores, kind='mergesort')[-self.k:]
            self.selected_idxs_ = np.sort(feature_idxs, kind='mergesort')
            self.scores_ = scores[np.argsort(feature_idxs, kind='mergesort')]
        return self

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features."
                % self.k)

    def _get_support_mask(self):
        check_is_fitted(self, 'selected_idxs_')
        mask = np.zeros(self._n_features, dtype=bool)
        if self.k == 'all' or self.k > 0:
            mask[self.selected_idxs_] = True
        return mask


class ReliefF(BaseEstimator, SelectorMixin):
    """Feature selector using ReliefF algorithm

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. The "all" option bypasses selection,
        for use in a parameter search.

    n_neighbors : int (default = 10)
        Number of neighbors for ReliefF algorithm

    sample_size : int (default = 5)
        Sample size for ReliefF algorithm

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    scores_ : array-like, shape=(n_features,)
        Feature scores
    """

    def __init__(self, k='all', n_neighbors=10, sample_size=5, memory=None):
        self.k = k
        self.n_neighbors = n_neighbors
        self.sample_size = sample_size
        self.memory = memory

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training input data matrix.

        y : array-like, shape = (n_samples)
            Target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=None)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        warnings.filterwarnings('ignore', category=RRuntimeWarning,
                                message="^Rjava\.init\.warning")
        self.scores_ = memory.cache(relieff_feature_score)(X, y)
        warnings.filterwarnings('always', category=RRuntimeWarning)
        return self

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features."
                % self.k)

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.k == 'all':
            mask = np.ones_like(self.scores_, dtype=bool)
        elif self.k > 0:
            mask[np.argsort(self.scores_, kind='mergesort')[-self.k:]] = True
        return mask
