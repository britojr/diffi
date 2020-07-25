import numpy as np

# epsilon as defined in the original paper
_EPSILON = 1e-2

def diffi_score(forest, X, inlier_samples="auto"):
    """
    Depth-based Isolation Forest Feature Importance (DIFFI) Algorithm [1].

    Return the feature importance for every feature of a given isolation forest.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        The input samples.

    inlier_samples : 'auto', 'all' or int, default='auto'
        The amount of inlier samples to consider when computing importance coefficient.
            - If 'auto', use the same amount of outliers found.
            - If 'all', use the all the inliers in the dataset.
            - If int, the contamination should be in the range [0, 0.5].

    Returns
    -------
    fi : numpy.ndarray of shape (n_features,)
        Array with the inportance of each feature.

    References
    ----------
    .. [1] Carletti, Mattia, Chiara Masiero, Alessandro Beghi, and Gian Antonio Susto.
           "Explainable machine learning in industry 4.0: evaluating feature importance in anomaly detection to enable root cause analysis."
           IEEE International Conference on Systems, Man and Cybernetics (SMC), pp. 21-26. IEEE, 2019.
    """

    pred = forest.predict(X)
    X_out = X[pred < 0]
    X_in = X[pred > 0]

    if inlier_samples == "all":
        k = X_in.shape[0]
    elif inlier_samples == "auto":
        k = X_out.shape[0]
    else:
        k = int(inlier_samples)
    if k < X_in.shape[0]:
        X_in = X_in[np.random.choice(X_in.shape[0], k, replace=False), :]

    return (_mean_cumulative_importance(forest, X_out) /
            _mean_cumulative_importance(forest, X_in))


def _mean_cumulative_importance(forest, X):
    '''
    Computes mean cumulative importance for every feature of given forest on dataset X
    '''

    f_importance = np.zeros(X.shape[1])
    f_count = np.zeros(X.shape[1])

    if forest._max_features == X.shape[1]:
        subsample_features = False
    else:
        subsample_features = True

    for tree, features in zip(forest.estimators_, forest.estimators_features_):
        X_subset = X[:, features] if subsample_features else X

        importance_t, count_t = _cumulative_ic(tree, X_subset)

        if subsample_features:
            f_importance[features] += importance_t
            f_count[features] += count_t
        else:
            f_importance += importance_t
            f_count += count_t

    return f_importance / f_count


def _cumulative_ic(tree, X):
    '''
    Computes importance and count for every feature of given tree on dataset X
    '''
    importance = np.zeros(X.shape[1])
    count = np.zeros(X.shape[1])

    node_indicator = tree.decision_path(X)
    depth = np.array(node_indicator.sum(axis=1)).reshape(-1)
    node_loads = np.array(node_indicator.sum(axis=0)).reshape(-1)

    iic = _induced_imbalance_coeff(tree, X, node_loads)
    rows, cols = node_indicator.nonzero()
    for i, j in zip(rows, cols):
        f = tree.tree_.feature[j]
        # ignore leaf nodes
        if f < 0:
            continue
        count[f] += 1
        importance[f] += iic[j] / depth[i]

    return importance, count

def _induced_imbalance_coeff(tree, X, node_loads):
    '''
    Computes imbalance coefficient for every *node* of a tree on dataset X
    '''
    iic = np.zeros_like(node_loads)
    for i in range(len(iic)):
        # ignore leaf nodes
        if tree.tree_.children_left[i] < 0:
            continue
        n_left = node_loads[tree.tree_.children_left[i]]
        n_right = node_loads[tree.tree_.children_right[i]]
        if n_left == 0 or n_right == 0:
            iic[i] = _EPSILON
            continue
        if n_left == 1 or n_right == 1:
            iic[i] = 1
            continue
        iic[i] = max(n_left, n_right) / node_loads[i]
    return iic
