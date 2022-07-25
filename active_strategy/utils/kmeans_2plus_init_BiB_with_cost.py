#---------------------------------------------------------------------------------------
# Code adapted from scikit-learn (https://scikit-learn.org/stable/getting_started.html)
# by Huy V. Vo and Oriane Simeoni
# INRIA, Valeo.ai
#---------------------------------------------------------------------------------------

from sklearn.metrics.pairwise import _euclidean_distances
from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.utils import check_array
from sklearn.utils import check_random_state
import numpy as np
import scipy.sparse as sp

def kmeans_plusplus(
    X, init_seeds, group_indices, group_cost, max_cost, 
    *, x_squared_norms=None, random_state=None, n_local_trials=None
):
    """Init n_clusters seeds according to k-means++
    .. versionadded:: 0.24
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds from.
    init_seeds: An initial list of centroids indices.
    group_indices: array or list, same length as X, indices of the groups
        of points in X.
    group_cost: dictionary, keys are group indices, values are group costs.
    x_squared_norms : array-like of shape (n_samples,), default=None
        Squared Euclidean norm of each data point.
    random_state : int or RandomState instance, default=None
        Determines random number generation for centroid initialization. Pass
        an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)).
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.
    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Examples
    --------
    >>> from sklearn.cluster import kmeans_plusplus
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=0)
    >>> centers
    array([[10,  4],
           [ 1,  0]])
    >>> indices
    array([4, 2])
    """

    # Check data
    check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])

    if len(group_indices) != len(X):
        raise ValueError(
            f"The number of group indices ({len(group_indices)}) must be equal "
            f"to the number of data point ({len(X)})!"
        )
    group_indices = np.array(group_indices)

    # Check parameters
    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)
    else:
        x_squared_norms = check_array(x_squared_norms, dtype=X.dtype, ensure_2d=False)

    if x_squared_norms.shape[0] != X.shape[0]:
        raise ValueError(
            f"The length of x_squared_norms {x_squared_norms.shape[0]} should "
            f"be equal to the length of n_samples {X.shape[0]}."
        )

    if n_local_trials is not None and n_local_trials < 1:
        raise ValueError(
            f"n_local_trials is set to {n_local_trials} but should be an "
            "integer value greater than zero."
        )

    random_state = check_random_state(random_state)

    # Call private k-means++
    indices, selected_group_indices = _kmeans_plusplus(
        X, init_seeds, group_indices, group_cost, max_cost,
        x_squared_norms, random_state, n_local_trials
    )

    return indices, selected_group_indices


def _kmeans_plusplus(X, init_seeds, group_indices, group_cost, max_cost,
                     x_squared_norms, random_state, n_local_trials=None):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.
    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.
    n_groups : int
        The number of groups to choose.
    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.
    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, n_features = X.shape

    indices = []
    selected_group_indices = []

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        # Estimate average number of selected groups
        estimate_num_groups = int(max_cost / np.mean(list(group_cost.values())) + 0.5)
        n_local_trials = 2 + int(np.log(estimate_num_groups))

    # Pick first center randomly and track index of point
    if len(init_seeds) == 0:
        center_id = random_state.randint(n_samples)
        center_ids = np.where(group_indices == group_indices[center_id])[0]
        indices += list(center_ids)
        selected_group_indices.append(group_indices[center_id])
        num_init_groups = 1
    else:
        num_init_groups = len(np.unique(group_indices[init_seeds]))
        _, _ids = np.unique(group_indices[init_seeds], return_index=True)
        selected_group_indices = list(group_indices[init_seeds][np.sort(_ids)])
        for gi in selected_group_indices:
            indices += list(np.where(group_indices == gi)[0])
    
    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = _euclidean_distances(
        X[indices], X, Y_norm_squared=x_squared_norms, squared=True
    )
    if len(indices) > 1:
        closest_dist_sq = np.min(closest_dist_sq, axis=0)
    current_pot = closest_dist_sq.sum()
    current_cost = np.sum([group_cost[g] for g in selected_group_indices])

    
    # Pick the remaining n_groups-1 groups
    # for c in range(num_init_groups, n_groups):
    while current_pot > 0 and current_cost < max_cost:
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)
        if len(candidate_ids) == 0:
            break

        # select best candidate
        best_candidate = -1
        best_pot = current_pot
        for cid in candidate_ids:
            new_indices = list(np.where(group_indices == group_indices[cid])[0])
            distance_to_new_indices = _euclidean_distances(
                X[new_indices], X, Y_norm_squared=x_squared_norms, squared=True
            )
            if len(new_indices) > 1:
                np.minimum(closest_dist_sq, distance_to_new_indices, out=distance_to_new_indices)
                candidate_pot = np.min(distance_to_new_indices, axis=0).sum()
            else:
                candidate_pot = np.minimum(closest_dist_sq, distance_to_new_indices[0]).sum()
            if candidate_pot < best_pot:
                best_candidate = cid
                best_pot = candidate_pot

        # add new indices
        new_indices = list(np.where(group_indices == group_indices[best_candidate])[0])
        distance_to_new_indices = _euclidean_distances(
            X[new_indices], X, Y_norm_squared=x_squared_norms, squared=True
        )
        if len(new_indices) > 1:
            np.minimum(closest_dist_sq, distance_to_new_indices, out=distance_to_new_indices)
            closest_dist_sq = np.min(distance_to_new_indices, axis=0)
        else:
            closest_dist_sq = np.minimum(closest_dist_sq, distance_to_new_indices[0])
        current_pot = closest_dist_sq.sum()
        
        # Permanently add best center candidate found in local tries
        indices += list(new_indices)
        selected_group_indices.append(group_indices[best_candidate])
        current_cost += group_cost[group_indices[best_candidate]]
        
    return indices, selected_group_indices
