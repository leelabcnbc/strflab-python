import numpy as np
from numpy.linalg import norm, svd
from . import rta


def rtc(stimulus_flat_all, response_all, correction=False, svd_of_cov_matrix=None,
        project_out_rta=False, subtract_rta=True):
    """compute response triggered covariance

    notice that here the algorithm used is more like the naive spike-triggered covariance
    the STC in original STRFlab is more like linear regression, instead of this.

    it's assumed that stimulus in stimulus_flat_all has an elliptically symmetric distribution.

    :param stimulus_flat_all: T x D np.ndarray.
    :param response_all: T x M np.ndarray.
    :return: a tuple with following elements
         1. MxDxD np.ndarray. For each neuron,
            each DxD matrix stores response triggered covariance directions as rows.
            in the order of decreasing eigenvalues.
         2. MxD np.ndarray. for each neuron, each (D,) vector stores eigenvalues.
         3. MxDxD np.ndarray. relevant response triggered covariance directions, in the original space.
         4. MxDxD np.ndarray. irrelevant response triggered covariance directions, in the original space.

         3 and 4 are same as 1 if correction is False.

    you can also perform analysis after projecting out rta, as in many papers of Schwartz, Rust, and Simoncelli.
    you can also choose to subtract_rta or not. this is performed after projection.

    the rta is assumed to be in the original space.

    mostly this function follows

    Spike-triggered covariance: geometric proof, symmetry properties, and extension beyond Gaussian stimuli

    https://doi.org/10.1007/s10827-012-0411-y

    by Inés Samengo and Tim Gollisch

    we assume that stimulus matrix has full rank. if not, you need to preprocess it, such as PCA.

    """
    t1, d = stimulus_flat_all.shape
    t2, m = response_all.shape

    assert t1 == t2 and d > 1 and m > 0

    # # compute corrected rta.
    # rta_all = rta.rta(stimulus_flat_all, response_all)

    # first, compute stim in the corrected space.
    # check equations (21-30) in the paper.
    if not correction:
        assert svd_of_cov_matrix is None
        stim_to_use = stimulus_flat_all
        relevant_trans = None
        irrelevant_trans = None
    else:
        u, s, _ = svd_of_cov_matrix
        assert np.all(s > 0) and u.shape == (d, d)
        # compute forward mapping. Eq. (22)
        forward_mapping = (u * (s ** (-0.5))).T
        # Eq. (22)
        stim_to_use = np.matmul(stimulus_flat_all, forward_mapping.T)
        relevant_trans = forward_mapping.T
        irrelevant_trans = u * (s ** 0.5)

    # # then compute RTA in the corrected space.
    #

    # collect cov matrices for all data.
    cov_all, _ = _collect_cov(stim_to_use, response_all,
                              project_out_rta, subtract_rta)

    # for each cov, compute svd, as well as relevant directions and irrelevant directions.

    result = _compute_svd_all(cov_all, relevant_trans, irrelevant_trans)
    _compute_svd_all_check_shape(result, m, d)
    # return everything.
    return result


def _compute_svd_all_check_shape(result, m, d):
    vector_original_list, eig_list, vector_relevant_list, vector_irrelevant_list = result
    assert vector_original_list.shape == (m, d, d)
    assert eig_list.shape == (m, d)
    assert vector_relevant_list.shape == (m, d, d)
    assert vector_irrelevant_list.shape == (m, d, d)


def _compute_svd_all(cov_all, relevant_trans, irrelevant_trans):
    eig_list = []
    vector_original_list = []
    vector_relevant_list = []
    vector_irrelevant_list = []

    for cov in cov_all:
        u, d, _ = svd(cov)
        eig_list.append(d)
        vector_original_list.append(u.T)
        if relevant_trans is None:
            vector_relevant_list.append(u.T)
        else:
            vector_relevant_list.append(np.matmul(relevant_trans, u).T)

        if irrelevant_trans is None:
            vector_irrelevant_list.append(u.T)
        else:
            vector_irrelevant_list.append(np.matmul(irrelevant_trans, u).T)

    eig_list = np.array(eig_list)
    vector_original_list = np.array(vector_original_list)
    vector_relevant_list = np.array(vector_relevant_list)
    vector_irrelevant_list = np.array(vector_irrelevant_list)

    return vector_original_list, eig_list, vector_relevant_list, vector_irrelevant_list


def _collect_cov(stim_to_use, response_all, project_out_rta, subtract_rta):
    rta_all = rta.rta(stim_to_use, response_all)  # (M,D)
    m, d = rta_all.shape

    cov_list = []

    for i_neuron in range(m):
        rta_this = rta_all[i_neuron]
        response_this = response_all[:, i_neuron]
        if project_out_rta:
            rta_this_normed = (rta_this / norm(rta_this))[:, np.newaxis]  # (D,1)
            assert rta_this_normed.shape == (d, 1)
            stim_to_use_this = stim_to_use - np.matmul(np.matmul(stim_to_use, rta_this_normed), rta_this_normed.T)
            rta_this[...] = 0
        else:
            stim_to_use_this = stim_to_use

        if subtract_rta:
            # don't do -=. that probably will have side effects.
            stim_to_use_this = stim_to_use_this - rta_this
        stim_to_use_this_weighted = stim_to_use_this * response_this[:, np.newaxis]
        # compute response triggered covariance.
        # Eq. (15).
        cov_this = np.matmul(stim_to_use_this_weighted.T, stim_to_use_this_weighted) / response_this.sum()
        assert cov_this.shape == (d, d)
        cov_list.append(cov_this)
    assert len(cov_list) == m
    return cov_list, rta_all
