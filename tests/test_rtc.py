"""essentially rewrite of /demo/rta.ipynb"""

import unittest
from itertools import product

import numpy as np
from numpy.linalg import svd, matrix_rank, inv, norm

from strflab import test_util, preprocessing, rta, rtc
from strflab.test_util import norm_vector, sigmoid, cos_between

rng_stream_global = test_util.rng_stream


def generate_one_kernel_set(seed, kernelsize):
    rng_stream_global.seed(seed)
    k1_kernel = rng_stream_global.randn(kernelsize)
    # I didn't make excitatory and suppressive vertical.
    # non-vertical seems to make the difference between eigenvalues for subtracting STA vs. not subtracting bigger.
    k2_kernel = rng_stream_global.randn(kernelsize)
    k3_kernel = rng_stream_global.randn(kernelsize)
    kernels_all = np.array([k1_kernel, k2_kernel, k3_kernel])

    return kernels_all


def generate_all_kernels(num_neuron, kernel_size):
    return np.asarray([generate_one_kernel_set(seed, kernel_size) for seed in range(num_neuron)])


def generate_eq7_output(stim_in_this, kernel_to_recover):
    # implement Eq. (7)
    # this should be N x 3.
    stim_out_linear = np.matmul(stim_in_this, kernel_to_recover.T)
    k1_response, k2_response, k3_response = stim_out_linear.T
    k1_response = np.square(np.clip(k1_response, 0, np.inf))
    k2_response = np.square(k2_response)
    k3_response = np.square(k3_response)

    return sigmoid(2 * (1 + k1_response) / (1 + k2_response + 0.4 * k3_response))


def _get_kernels(vector_relevant_list, vector_irrelevant_list, project_sta):
    # get kernels.
    if not project_sta:
        # then excitatory is first one, suppresive are last two.
        relevant_e = vector_relevant_list[0]
        relevant_i1 = vector_relevant_list[-2]
        relevant_i2 = vector_relevant_list[-1]
        # if you change `vector_irrelevant_list` to `vector_relevant_list`.
        # then _check_orthogonal will give large numbers.
        irrelevant_all = vector_irrelevant_list[1:-2]
    else:
        # otherwise, STA is the last one.
        relevant_e = vector_relevant_list[-1]
        relevant_i1 = vector_relevant_list[-3]
        relevant_i2 = vector_relevant_list[-2]
        irrelevant_all = vector_irrelevant_list[:-3]

    relevant_e_flat = norm_vector(relevant_e.ravel())
    relevant_i1_flat = norm_vector(relevant_i1.ravel())
    relevant_i2_flat = norm_vector(relevant_i2.ravel())
    irrelevant_all = np.asarray([norm_vector(x.ravel()) for x in irrelevant_all])

    # I always present them in the order of e, i1, i2.
    relevant_all = np.asarray([relevant_e_flat, relevant_i1_flat, relevant_i2_flat])

    return relevant_all, irrelevant_all


def _check_orthogonal(test_instance: unittest.TestCase, relevant_all, irrelevant_all):
    result = relevant_all @ irrelevant_all.T
    test_instance.assertTrue(abs(result).max() < 1e-6)


def _check_projection(test_instance: unittest.TestCase, relevant_all, kernels_this):
    # see <https://en.wikipedia.org/wiki/Projection_(linear_algebra)>
    project_matrix = relevant_all.T @ inv(relevant_all @ relevant_all.T) @ relevant_all
    kernels_this_projected = (project_matrix @ kernels_this.T).T
    assert kernels_this_projected.shape == kernels_this.shape
    # check cos
    cos_values = np.asarray([cos_between(v1, v2) for v1, v2 in zip(kernels_this, kernels_this_projected)])
    # it's not very reliable.

    # check norm
    norm_diff = np.asarray([norm(v1) / norm(v2) for v1, v2 in zip(kernels_this, kernels_this_projected)])
    test_instance.assertTrue(abs(cos_values - 1).max() < 0.01)
    # print(norm_diff)
    test_instance.assertTrue(abs(norm_diff - 1).max() < 0.01)

    return kernels_this_projected


def _check_one_case(test_instance: unittest.TestCase, relevant_list, irrelevant_list, project_sta, kernels_this):
    relevant_all, irrelevant_all = _get_kernels(relevant_list,
                                                irrelevant_list, project_sta)
    _check_orthogonal(test_instance, relevant_all, irrelevant_all)

    # check projections
    _check_projection(test_instance, relevant_all, kernels_this)


def check_data_stc(test_instance: unittest.TestCase, stim_in_this, kernels, introduce_corr=False, seed=0):
    rng_stream_global.seed(seed)
    num_neuron, _, kernelsize = kernels.shape
    assert _ == 3

    if introduce_corr:
        transform_matrix = test_util.generate_correlation_matrix(kernelsize, 0)
        assert matrix_rank(transform_matrix) == kernelsize
        stim_in_this = np.matmul(stim_in_this, transform_matrix)
        svd_cov = svd(preprocessing.cov_matrix(stim_in_this))
    else:
        svd_cov = None
    response_all_rate = []
    response_all_spike = []
    for kernels_this in kernels:
        data_out = generate_eq7_output(stim_in_this, kernels_this)
        data_out_spike = (rng_stream_global.rand(*data_out.shape) < data_out).astype(np.float64)
        response_all_rate.append(data_out)
        response_all_spike.append(data_out_spike)
    response_all_rate = np.asarray(response_all_rate).T
    response_all_spike = np.asarray(response_all_spike).T
    for use_spike in (True, False):
        if use_spike:
            data_out_this = response_all_spike
        else:
            data_out_this = response_all_rate
        eig_dict = dict()

        rta_all = rta.rta(stim_in_this, data_out_this)
        if introduce_corr:
            rta_all = rta.correct_rta(rta_all, svd_cov)

        for (project_sta, subtract_sta) in product((True, False), (True, False)):
            (vector_original_list, eig_list,
             vector_relevant_list, vector_irrelevant_list) = rtc.rtc(stim_in_this,
                                                                     data_out_this,
                                                                     correction=introduce_corr,
                                                                     svd_of_cov_matrix=svd_cov,
                                                                     project_out_rta=project_sta,
                                                                     subtract_rta=subtract_sta)
            test_instance.assertEqual(vector_original_list.shape, (num_neuron, kernelsize, kernelsize))
            test_instance.assertEqual(eig_list.shape, (num_neuron, kernelsize))
            test_instance.assertEqual(vector_relevant_list.shape, (num_neuron, kernelsize, kernelsize))
            test_instance.assertEqual(vector_irrelevant_list.shape, (num_neuron, kernelsize, kernelsize))

            if not introduce_corr:
                test_instance.assertTrue(np.array_equal(vector_original_list, vector_relevant_list))
                test_instance.assertTrue(np.array_equal(vector_original_list, vector_irrelevant_list))

            if project_sta:
                # shoulc check rta
                # check that the rta returned here (should be last relevant eigenvector)
                # is equivalent as the STA returned by rta.
                rta_from_rtc = vector_relevant_list[:, -1]
                assert rta_from_rtc.shape == rta_all.shape
                cos_values = np.asarray([cos_between(v1, v2) for v1, v2 in zip(rta_from_rtc, rta_all)])
                # it's not very reliable.
                #
                # print(cos_values)
                test_instance.assertTrue(abs(abs(cos_values) - 1).max() < 1e-3)

                # ok. time to check neurons one by one.
                # collect eigs
            eig_dict[(project_sta, subtract_sta)] = eig_list

            # # plot pairs.
            for i_neuron in range(num_neuron):
                _check_one_case(test_instance, vector_relevant_list[i_neuron],
                                vector_irrelevant_list[i_neuron], project_sta, kernels[i_neuron])

                # axes[0].scatter(np.arange(kernelsize) + 1, eig_dict[True, True] - eig_dict[True, False])
                # axes[0].set_title()
                #
                # axes[1].scatter(np.arange(kernelsize) + 1, eig_dict[False, True] - eig_dict[False, False])
                # axes[1].set_title(
                #     '')
                #
                # plt.show()
        # when projecting sta out, subtract - no_subtract (should be all flat)
        test_instance.assertTrue(abs(eig_dict[True, True] - eig_dict[True, False]).max() < 0.001)
        # when NOT projecting sta out, subtract - no_subtract (should be only changed at relevant directions if at all)
        test_instance.assertTrue(abs(eig_dict[False, True][:, 1:-2] - eig_dict[False, False][:, 1:-2]).max() < 0.01)


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        rng_stream_global.seed(0)

    def test_rtc_white(self):
        kernelsize = 12
        for num_neuron in (1, 3):
            kernels_all = generate_all_kernels(num_neuron, kernelsize)
            stim_this = rng_stream_global.randn(2000000, kernelsize)
            check_data_stc(self, stim_this, kernels_all, False)

    def test_rta_nonwhite(self):
        kernelsize = 12
        for num_neuron in (1, 3):
            kernels_all = generate_all_kernels(num_neuron, kernelsize)
            stim_this = rng_stream_global.randn(2000000, kernelsize)
            check_data_stc(self, stim_this, kernels_all, True)


if __name__ == '__main__':
    unittest.main(failfast=True)
