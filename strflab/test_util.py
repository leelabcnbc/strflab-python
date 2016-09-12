from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

rng_stream = np.random.RandomState(seed=0)


def get_random_shape(minshape=1, maxshape=4):
    rand_dim = rng_stream.randint(minshape, maxshape + 1)
    # so at most we have 4x4x4x4 = 256 dimensions to recover.
    # set to be small, so that Travis CI won't crash.
    return rng_stream.randint(low=1, high=4, size=rand_dim)


def get_random_kernel(shape, num_neuron=1, num_delay=1):
    return rng_stream.randn(num_neuron, num_delay, *shape)


def get_random_stimulus(shape, num_stimulus=10000):
    if num_stimulus is None:
        num_stimulus = np.random.randint(1000, 20000)

    return rng_stream.randn(num_stimulus, *shape)
