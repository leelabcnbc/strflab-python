"""essentially rewrite of /demo/rta.ipynb"""

import unittest
from collections import OrderedDict

import numpy as np
from numpy.linalg import svd, matrix_rank

from strflab import test_util, preprocessing, rta
from strflab.test_util import norm_vector, sigmoid, cos_between

rng_stream_global = test_util.rng_stream


def generate_out_dict(stim_in, kernel_to_recover):
    stim_out_linear = np.matmul(stim_in, kernel_to_recover.T)
    out_dict = OrderedDict()

    # square
    out_dict['square'] = stim_out_linear * 2
    # 0, 1 at threshold of 0
    out_dict['threshold'] = np.where(stim_out_linear >= 0, 1, 0)
    # sigmoid
    out_dict['sigmoid'] = sigmoid(stim_out_linear)
    # relu
    out_dict['relu'] = np.where(stim_out_linear >= 0, stim_out_linear, 0)

    return out_dict


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        rng_stream_global.seed(0)

    def test_rta_white(self):
        for _ in range(10):
            kernelsize = 32
            kernel_to_recover = rng_stream_global.randn(1, kernelsize)
            stim_in = rng_stream_global.randn(100000, kernelsize)
            out_dict = generate_out_dict(stim_in, kernel_to_recover)
            for out_name, out_this in out_dict.items():
                rta_this = rta.rta(stim_in, out_this)
                self.assertEqual(rta_this.shape, (1, kernelsize))
                # compute cosine between recovered and original
                this_one = norm_vector(rta_this.ravel())
                ref_one = norm_vector(kernel_to_recover.ravel())
                self.assertTrue(abs(abs(cos_between(this_one, ref_one)) - 1) < 0.01)

    def test_rta_white_multiple(self):
        for _ in range(10):
            kernelsize = 32
            kernel_to_recover = rng_stream_global.randn(5, kernelsize)
            stim_in = rng_stream_global.randn(100000, kernelsize)
            out_dict = generate_out_dict(stim_in, kernel_to_recover)
            for out_name, out_this in out_dict.items():
                rta_this = rta.rta(stim_in, out_this)
                self.assertEqual(rta_this.shape, (5, kernelsize))

                rta_this = preprocessing.reshape_kernel(rta_this, (kernelsize,), 1,
                                                        5, False)
                assert rta_this.shape == (5, 1, kernelsize)
                rta_this = rta_this[:, 0]

                for i_vector in range(5):
                    kernel_to_recover_this = kernel_to_recover[i_vector]
                    rta_this_this = rta_this[i_vector]
                    # compute cosine between recovered and original
                    this_one = norm_vector(rta_this_this.ravel())
                    ref_one = norm_vector(kernel_to_recover_this.ravel())
                    self.assertTrue(abs(abs(cos_between(this_one, ref_one)) - 1) < 0.01)

    def test_rta_nonwhite(self):
        for _ in range(10):
            kernelsize = 32
            kernel_to_recover = rng_stream_global.randn(1, kernelsize)
            stim_in = rng_stream_global.randn(100000, kernel_to_recover.size)

            # generate some cov matrix.
            transform_matrix = test_util.generate_correlation_matrix(kernelsize)
            assert matrix_rank(transform_matrix) == kernelsize
            stim_in_nonwhite = np.matmul(stim_in, transform_matrix)
            svd_cov = svd(preprocessing.cov_matrix(stim_in_nonwhite))
            out_dict = generate_out_dict(stim_in_nonwhite, kernel_to_recover)
            for out_name, out_this in out_dict.items():
                rta_this = rta.rta(stim_in_nonwhite, out_this)
                # comment out the line below will make things fail.
                rta_this = rta.correct_rta(rta_this, svd_cov)
                self.assertEqual(rta_this.shape, (1, kernelsize))
                # compute cosine between recovered and original
                this_one = norm_vector(rta_this.ravel())
                ref_one = norm_vector(kernel_to_recover.ravel())
                self.assertTrue(abs(abs(cos_between(this_one, ref_one)) - 1) < 0.01)

    def test_rta_nonwhite_multiple(self):
        for _ in range(10):
            kernelsize = 32  # if too big, may run into some precision issues.
            kernel_to_recover = rng_stream_global.randn(5, kernelsize)
            stim_in = rng_stream_global.randn(100000, kernelsize)

            # generate some cov matrix.
            transform_matrix = test_util.generate_correlation_matrix(kernelsize)

            assert matrix_rank(transform_matrix) == kernelsize
            stim_in_nonwhite = np.matmul(stim_in, transform_matrix)
            svd_cov = svd(preprocessing.cov_matrix(stim_in_nonwhite))

            out_dict = generate_out_dict(stim_in_nonwhite, kernel_to_recover)
            for out_name, out_this in out_dict.items():
                rta_this = rta.rta(stim_in_nonwhite, out_this)
                rta_this = rta.correct_rta(rta_this, svd_cov)
                self.assertEqual(rta_this.shape, (5, kernelsize))

                rta_this = preprocessing.reshape_kernel(rta_this, (kernelsize,), 1,
                                                        5, False)
                assert rta_this.shape == (5, 1, kernelsize)
                rta_this = rta_this[:, 0]

                for i_vector in range(5):
                    kernel_to_recover_this = kernel_to_recover[i_vector]
                    rta_this_this = rta_this[i_vector]
                    # compute cosine between recovered and original
                    this_one = norm_vector(rta_this_this.ravel())
                    ref_one = norm_vector(kernel_to_recover_this.ravel())
                    self.assertTrue(abs(abs(cos_between(this_one, ref_one)) - 1) < 0.01)


if __name__ == '__main__':
    unittest.main(failfast=True)
