"""essentially rewrite of /demo/rta.ipynb"""

import unittest
from strflab import test_util, preprocessing, rta
from collections import OrderedDict
import numpy as np
from numpy.linalg import norm, svd, matrix_rank


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def norm_vector(x):
    assert x.ndim == 1
    return x / norm(x)


def cos_between(v1, v2):
    assert v1.ndim == v2.ndim == 1
    return np.dot(v1, v2) / norm(v1) / norm(v2)


rng_stream_global = test_util.rng_stream


def generate_correlation_matrix(size):
    # <https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
    mix_cov = rng_stream_global.randn(size, size)
    mix_cov = mix_cov * mix_cov.T + np.diag(rng_stream_global.rand(size) * 0.00001)
    # then svd
    U, S, Vh = svd(mix_cov)
    return (U @ np.diag(S ** 0.5)).T
    # stim_in_non_white = stim_in


def generate_out_dict(stim_in, kernel_to_recover):
    stim_out_linear = np.matmul(stim_in, kernel_to_recover.ravel())
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
            kernelsize = 64
            kernel_to_recover = rng_stream_global.randn(kernelsize)
            stim_in = rng_stream_global.randn(100000, kernel_to_recover.size)
            out_dict = generate_out_dict(stim_in, kernel_to_recover)
            for out_name, out_this in out_dict.items():
                rta_this = rta.rta(stim_in, out_this[:, np.newaxis])
                self.assertEqual(rta_this.shape, (1, kernelsize))
                # compute cosine between recovered and original
                this_one = norm_vector(rta_this.ravel())
                ref_one = norm_vector(kernel_to_recover.ravel())
                self.assertTrue(abs(abs(cos_between(this_one, ref_one)) - 1) < 0.01)

    def test_rta_nonwhite(self):
        for _ in range(10):
            kernelsize = 64
            kernel_to_recover = rng_stream_global.randn(kernelsize)
            stim_in = rng_stream_global.randn(100000, kernel_to_recover.size)

            # generate some cov matrix.
            transform_matrix = generate_correlation_matrix(kernelsize)
            assert matrix_rank(transform_matrix) == kernelsize
            stim_in_nonwhite = np.matmul(stim_in, transform_matrix)
            svd_cov = svd(preprocessing.cov_matrix(stim_in_nonwhite))
            out_dict = generate_out_dict(stim_in_nonwhite, kernel_to_recover)
            for out_name, out_this in out_dict.items():
                rta_this = rta.rta(stim_in_nonwhite, out_this[:, np.newaxis])
                # comment out the line below will make things fail.
                rta_this = rta.correct_rta(rta_this, svd_cov)
                self.assertEqual(rta_this.shape, (1, kernelsize))
                # compute cosine between recovered and original
                this_one = norm_vector(rta_this.ravel())
                ref_one = norm_vector(kernel_to_recover.ravel())
                self.assertTrue(abs(abs(cos_between(this_one, ref_one)) - 1) < 0.01)


if __name__ == '__main__':
    unittest.main(failfast=True)
