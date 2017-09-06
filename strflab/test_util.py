from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.linalg import svd, norm

rng_stream = np.random.RandomState(seed=0)


def get_random_shape(minshape=1, maxshape=4, low=1, high=4):
    rand_dim = rng_stream.randint(minshape, maxshape + 1)
    # so at most we have 4x4x4x4 = 256 dimensions to recover.
    # set to be small, so that Travis CI won't crash.
    return rng_stream.randint(low=low, high=high, size=rand_dim)


def get_random_kernel(shape, num_neuron, num_delay):
    return rng_stream.randn(num_neuron, num_delay, *shape)


def get_random_stimulus(shape, num_stimulus):
    return rng_stream.randn(num_stimulus, *shape)


def get_random_response_gaussian_rate(num_neuron, num_stimulus, rate_mean=0, rate_std=1):
    return rng_stream.randn(num_stimulus, num_neuron) * rate_std + rate_mean


def generate_correlation_matrix(size, seed=None):
    # <https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
    if seed is not None:
        rng_stream.seed(seed)
    mix_cov = rng_stream.randn(size, size)
    mix_cov = mix_cov * mix_cov.T + np.diag(rng_stream.rand(size) * 0.00001)
    # then svd
    u, s, vh = svd(mix_cov)
    # using broadcasting, `*S` is same as `@diag(S)`.
    return (u * (s ** 0.5)).T
    # stim_in_non_white = stim_in


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def norm_vector(x):
    assert x.ndim == 1
    return x / norm(x)


def cos_between(v1, v2):
    assert v1.ndim == v2.ndim == 1
    return np.dot(v1, v2) / norm(v1) / norm(v2)
