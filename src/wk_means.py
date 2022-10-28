"""WK-means clustering."""

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.exceptions import ConvergenceWarning


class WKMeans(ClusterMixin, BaseEstimator):
    """WK-Means clustering.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='random'
        Method for initialization:

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    p : int, default=1
        order of W distance

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    n_iter_ : int
        Number of iterations run.

    Notes
    -----

    """

    def __init__(
        self,
        n_clusters=2,
        *,
        init='random',
        p=1,
        max_iter=300,
        tol=1e-16,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="auto",
    ):

        self.n_clusters = n_clusters
        self.init = init
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

    def _check_params(self, X):
        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter should be > 0, got {self.max_iter} instead.")

        # n_clusters
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        # init
        if not (
            hasattr(self.init, "__array__")
            or callable(self.init) or self.init == 'random'
        ):
            raise ValueError(
                "init should be either a ndarray or a"
                f"callable, got '{self.init}' instead."
            )

    def _validate_center_shape(self, X, centers):
        """Check if centers is compatible with X and n_clusters."""
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}."
            )
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )

    def _check_test_data(self, X):
        X = self._validate_data(
            X,
            accept_sparse="csr",
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        return X

    def _init_centroids(self, X, init, random_state, init_size=None):
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            n_samples = X.shape[0]

        if isinstance(init, str) and init == "random":
            seeds = random_state.permutation(n_samples)[:n_clusters]
            centers = X[seeds]
        elif hasattr(init, "__array__"):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(centers, dtype=X.dtype,
                                  copy=False, order="C")
            self._validate_center_shape(X, centers)

        if sp.issparse(centers):
            centers = centers.toarray()

        return centers

    def fit(self, X, y=None, sample_weight=None):
        """Compute wk-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Validate init array
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        # Initialize centers
        centers_init = self._init_centroids(
            X, init=init, random_state=random_state
        )
        if self.verbose:
            print("Initialization complete")

        # wk-means
        centroids_before = centers_init
        for i in range(self.max_iter):
            # assign closest centroid
            labels = pd.DataFrame(X).apply(
                lambda series: closest(series, centroids_before, self.p), axis=1)
            # update centroids
            centroids = [W_barycenter(X[labels == i], self.p)
                         for i in range(self.n_clusters)]
            # loss
            loss = loss_function(centroids_before, centroids, self.p)
            centroids_before = centroids
            if self.verbose:
                print(f'loss : {loss}')
            if loss <= self.tol:
                break

        distinct_clusters = len(set(labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.n_iter_ = i
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        return pd.DataFrame(X).apply(lambda series: closest(series, self.cluster_centers_), axis=1)

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            },
        }


# ------------------------------------------------ #
# ------------------- Helpers -------------------- #
# ------------------------------------------------ #

def W(alpha, beta, p=1):
    """W distance

    Parameters
    ----------
    alpha : {ndarray, sparse matrix} of shape (n_features,)
        atoms of the empirical measure
    beta : {ndarray, sparse matrix} of shape (n_features,)
        atoms of the empirical measure
    p : int, optional
        W distance order, by default 1

    Returns
    -------
    float
        W distancte
    """

    assert len(alpha) == len(beta)
    N = len(alpha)

    return 1/N * np.sum([abs(alpha[i] - beta[i])**p for i in range(N)])


def closest(mu, centroids, p=1):
    """Returns the closest centroid to a measure

    Parameters
    ----------
    mu : {ndarray, sparse matrix} of shape (n_features,)
        atoms of a measure
    centroids : {ndarray, sparse matrix} of shape (n_clusters,n_features)
        centroids

    Returns
    -------
    array
        closest centroid
    """
    return np.argmin([W(mu, centroids[i], p) for i in range(len(centroids))])


def W_barycenter(atoms, p=1):
    """W barycenter

    Parameters
    ----------
    atoms : {ndarray, sparse matrix} of shape (n_features,)
        cluster od measures
    p : int, optional
        W distance order, by default 1

    Returns
    -------
    array
        barycenter
    """
    if p > 1:
        atoms_j = np.mean(atoms, axis=0)
    else:
        atoms_j = np.median(atoms, axis=0)
    return atoms_j


# loss function ~ distance between "k" centroids
def loss_function(centroids_before, centroids, p=1):
    """loss function

    Parameters
    ----------
    centroids_before : {ndarray, sparse matrix} of shape (n_clusters,n_features)

    centroids : {ndarray, sparse matrix} of shape (n_clusters,n_features)


    Returns
    -------
    float
        distance between centroids (how much did the centroids move)
    """
    k = len(centroids_before)
    return np.sum([W(centroids_before[i], centroids[i], p) for i in range(k)])


def Q(x, j):
    """Computes the j_th order stat

    Parameters
    ----------
    x : array or list like
        Sample of the distribution
    j : int
        order

    Returns
    -------
    float
        the j_th order stat
    """
    assert j > 0
    return np.partition(np.asarray(x), j-1)[j-1]
