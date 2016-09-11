from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
rng_stream = np.random.RandomState(seed=0)


def get_random_shape(minshape = 1, maxshape = 5):
    rand_dim = rng_stream.randint(minshape, maxshape+1)
    return rng_stream.randint(low=1, high=10, size=rand_dim)

