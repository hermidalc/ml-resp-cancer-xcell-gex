# Authors: Leandro Hermida <hermidal@cs.umd.edu>
#
# License: BSD 3 clause

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.validation import check_array


class StratifiedGroupShuffleSplit(StratifiedShuffleSplit):
    """Stratified GroupShuffleSplit cross-validator

    Provides randomized train/test indices to split data according to a
    third-party provided group. This group information can be used to encode
    arbitrary domain specific stratifications of the samples as integers.

    This cross-validation object is a merge of GroupShuffleSplit and
    StratifiedShuffleSplit, which returns stratified randomized folds. The
    folds are made by preserving the percentage of groups for each class.

    Note: like the StratifiedShuffleSplit strategy, stratified random group
    splits do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int (default 5)
        Number of re-shuffling & splitting iterations.

    test_size : float, int, None, optional (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the test split (rounded up). If int,
        represents the absolute number of test groups. If None, the value is
        set to the complement of the train size. By default, the value is set
        to 0.1.

    train_size : float, int, or None, default is None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the groups to include in the train split. If
        int, represents the absolute number of train groups. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupShuffleSplit
    >>> X = np.ones(shape=(15, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6])
    >>> print(groups.shape)
    (15,)
    >>> sgss = StratifiedGroupShuffleSplit(n_splits=3, train_size=.7,
    ...                                    random_state=43)
    >>> sgss.get_n_splits()
    3
    >>> for train_idx, test_idx in sgss.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idx])
    ...     print("      ", y[train_idx])
    ...     print(" TEST:", groups[test_idx])
    ...     print("      ", y[test_idx])
    TRAIN: [2 2 2 4 5 5 5 5 6 6]
           [1 1 1 0 1 1 1 1 0 0]
     TEST: [1 1 3 3 3]
           [0 0 1 1 1]
    TRAIN: [1 1 2 2 2 3 3 3 4]
           [0 0 1 1 1 1 1 1 0]
     TEST: [5 5 5 5 6 6]
           [1 1 1 1 0 0]
    TRAIN: [1 1 2 2 2 3 3 3 6 6]
           [0 0 1 1 1 1 1 1 0 0]
     TEST: [4 5 5 5 5]
           [0 1 1 1 1]
    """

    def __init__(self, n_splits=5, test_size=None, train_size=None,
                 random_state=None):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state)
        self._default_test_size = 0.1

    def _iter_indices(self, X, y, groups):
        y = check_array(y, ensure_2d=False, dtype=None)
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)
        (unique_groups, unique_groups_y), group_indices = np.unique(
            np.stack((groups, y)), axis=1, return_inverse=True)
        if unique_groups.shape[0] != np.unique(groups).shape[0]:
            raise ValueError("Members of each group must all be of the same "
                             "class.")
        for group_train, group_test in super()._iter_indices(
                X=unique_groups, y=unique_groups_y):
            # these are the indices of unique_groups in the partition invert
            # them into data indices
            train = np.flatnonzero(np.in1d(group_indices, group_train))
            test = np.flatnonzero(np.in1d(group_indices, group_test))
            yield train, test

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : array-like, with shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        return super().split(X, y, groups)
