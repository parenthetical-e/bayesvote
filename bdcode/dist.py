import numpy as np

from itertools import product
from copy import deepcopy

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


class ConditionalKernelDensity:
    """Extends KernelDensity, creating conditional distributions
    on y."""
    def __init__(self, **kd_kwargs):
        self.kd_kwargs = kd_kwargs
        self._lookup = dict()

        # Set CV alg
        self._set_cv(GridSearchCV)

    def _set_cv(self, CV):
        self.CV = CV

    def _parse(self, X, y):
        channels = list(range(X.shape[1]))
        targets = np.unique(y)
        return targets, channels

    def _check(self, X, y):
        X = np.atleast_2d(X)
        if X.ndim > 2:
            raise ValueError("X must be 2d")
        return X

    def fit(self, X, y, cv_kwargs=None):
        # Sanity
        X = self._check(X, y)
        # Init meta
        self.targets, self.channels = self._parse(X, y)
        self.num_targets = len(self.targets)
        self.num_channels = len(self.channels)

        # Fit, save in lookup
        self._lookup = dict()
        for c, t in product(self.channels, self.targets):
            # Select
            mask = t == y
            X_filt = X[mask, c].reshape(np.sum(mask), 1)  # force 2d

            # Tune?
            if cv_kwargs is not None:
                grid = self.CV(KernelDensity(), **cv_kwargs)
                grid.fit(X_filt)
                kde = grid.best_estimator_
            else:
                kde = KernelDensity(**self.kd_kwargs)  # Defaults

            # Fit and log
            kde.fit(X_filt)
            self._lookup[(t, c)] = deepcopy(kde)

    def score_samples(self, X, y):
        X = self._check(X, y)
        X_score = np.zeros_like(X)

        # Fit / target
        for c, t in product(self.channels, self.targets):
            # Select
            mask = t == y
            X_filt = X[mask, c].reshape(np.sum(mask), 1)  # force 2d

            # Score
            scores = self._lookup[(t, c)].score_samples(X_filt)
            X_score[mask, c] = scores

        return X_score

    def probs(self, X, y):
        return np.exp(self.score_samples(X, y))
