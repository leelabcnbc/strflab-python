import numpy as np


def rta(stimulus_flat_all, response_all):
    """compute response triggered average

    notice that here the algorithm used is more like the naive spike-triggered average
    the STA in original STRFlab is more like linear regression, instead of this.

    it's assumed that stimulus in stimulus_flat_all has an elliptically symmetric distribution.

    :param stimulus_flat_all: T x D np.ndarray.
    :param response_all: T x M np.ndarray.
    :return: a MxD np.ndarray, each row being response triggered average for each neuron.
    """
    t1, d = stimulus_flat_all.shape
    t2, m = response_all.shape

    assert t1 == t2 and d > 1 and m > 0

    fit_kernels = np.matmul(response_all.T, stimulus_flat_all) / response_all.T.sum(axis=1, keepdims=True)

    return fit_kernels


def correct_rta(fit_kernels, svd_of_cov_matrix):
    """correct for bias in rta result due to correlation in stimulus"""
    u, s, _ = svd_of_cov_matrix
    m, d = fit_kernels.shape
    # assumes full SVD
    assert u.shape == (d, d) and s.shape == (d,) and (s > 0).all()
    cov_inv = np.matmul(np.matmul(u, np.diag(1 / s)), u.T)
    # cov_inv.T or cov_inv shouldn't matter, as cov_inv is symmetric. Here just for theoretic correctness.
    return np.matmul(fit_kernels, cov_inv.T)
