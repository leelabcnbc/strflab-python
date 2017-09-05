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
    T1, D = stimulus_flat_all.shape
    T2, M = response_all.shape

    assert T1 == T2 and D > 0 and M > 0

    fit_kernels = np.matmul(response_all.T, stimulus_flat_all) / response_all.T.sum(axis=1, keepdims=True)

    return fit_kernels


def correct_rta(fit_kernels, svd_of_cov_matrix):
    """correct for bias in rta result due to correlation in stimulus"""
    U, S, _ = svd_of_cov_matrix
    M, D = fit_kernels.shape
    # assumes full SVD
    assert U.shape == (D, D) and S.shape == (D,) and (S > 0).all()
    cov_inv = np.matmul(np.matmul(U, np.diag(1 / S)), U.T)
    # cov_inv.T or cov_inv shouldn't matter, as cov_inv is symmetric. Here just for theoretic correctness.
    return np.matmul(fit_kernels, cov_inv.T)
