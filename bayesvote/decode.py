import numpy as np
from copy import deepcopy


def decoder(X, y, prior, kde):
    """Decode X, given classes y.
    
    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds to
        a single data point.

    y : array-like of shape (n_samples,)
        Target values (class labels)

    prior : scalar (0-1)
        The prior
    
    kde: a KernelDensity-like object
        An instance of the ConditionalKernelDensity, or derivative

    Returns
    -------
    X : array-like of shape (n_samples, n_classes)
        List of n_classes-dimensional data points. Each row corresponds to
        a single bayes probability.
    """

    # Sanity check kde.channels and kde.targets against
    # X and y
    assert X.shape[1] == kde.num_channels, "Channel mismatch"

    # Init the return - a prob 2d array (time, target)
    X_decode = np.zeros((y.shape[0] + 1, kde.num_targets))
    X_decode[0, :] = prior

    # Build up indep trace tensor to renorm the updates
    X_like = np.zeros((y.shape[0], kde.num_channels, kde.num_targets))
    for j, c in enumerate(kde.channels):
        for k, t in enumerate(kde.targets):
            X_c = X[:, c].reshape(-1, 1)
            X_like[:, j, k] = np.exp(kde._lookup[(t, c)].score_samples(X_c))

    # !
    # Sequential bayes estimates for each target
    for k, t in enumerate(kde.targets):
        # Reinit for each target
        p_old = np.ones(kde.num_targets) * prior
        p_new = deepcopy(p_old)

        # Step
        for i in range(1, y.shape[0]):
            # Round-robin update over channels
            for j, c in enumerate(kde.channels):
                p = X_like[i, j, :]  # likelihood
                p_new = (p * p_old)  # update
                p_new /= np.sum(p_new)  # renorm
                p_old = deepcopy(p_new)  # shift

            # Update bayesian decode trace *after*
            # all the channels have 'voted'
            X_decode[i, k] = deepcopy(p_new[k])

    return X_decode