"""feature transformation of input.

this is primarily for feeding nonlinear input to GLM models.
"""

import numpy as np


def generator_all_valid_pairs(shape, locality=None):
    # this will generate all pairs that should be used in feature preprocessing.
    if locality is None:
        locality = (0,) * len(shape)
    assert len(shape) == len(locality) and isinstance(shape, tuple) and isinstance(locality, tuple) and len(shape) > 0
    locality = np.array(locality)
    assert np.all(locality >= 0) and locality.shape == (len(shape),) and np.all(locality < np.array(shape))
    num_el = int(np.product(shape))
    assert num_el > 0
    # then compute the multi index version.

    multi_index_all = np.asarray(np.unravel_index(np.arange(num_el), shape, order='C')).T
    assert multi_index_all.shape == (num_el, len(shape))

    linear_pair_to_save = []
    total_pair = 0
    for i in range(num_el):
        for j in range(i, num_el):
            index1, index2 = multi_index_all[i], multi_index_all[j]
            index_diff = abs(index1 - index2)
            assert index_diff.shape == locality.shape
            if np.all(index_diff <= locality):
                linear_pair_to_save.append((i, j))
                total_pair += 1

    linear_pair_to_save = np.asarray(linear_pair_to_save)
    assert linear_pair_to_save.shape == (total_pair, 2) and total_pair > 0
    return linear_pair_to_save


def quadratic_features(X, locality=None):
    """generate the quadratic features from X, based on locality

    :param X:
    :param locality:
    :return:
    """

    N, *shape = X.shape
    shape = tuple(shape)
    linear_pair_to_save = generator_all_valid_pairs(shape, locality)
    # I will generate a flat version.
    X_flat = np.reshape(X, (N, -1), order='C')
    # then generate a version on top of this X_flat
    X_quad = []
    for (i, j) in linear_pair_to_save:
        X_quad.append(X_flat[:, i] * X_flat[:, j])
    X_quad = np.asarray(X_quad).T
    assert X_quad.shape == (N, len(linear_pair_to_save))

    return X_quad
