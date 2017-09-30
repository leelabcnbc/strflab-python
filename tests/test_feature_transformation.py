import unittest
import numpy as np
from strflab import test_util, feature_transformation
from itertools import product

rng_stream_global = test_util.rng_stream


class TestFeatureTransformation(unittest.TestCase):
    def setUp(self):
        rng_stream_global.seed(0)

    def test_quadratic_features(self):
        for ndim in (1, 2, 3):
            shape_this = tuple(test_util.get_random_shape(ndim, ndim, low=1, high=10))
            num_el_all = np.product(shape_this)
            assert num_el_all > 1  # so that 'random' locality could be useful
            num_stimulus_this = rng_stream_global.randint(100, 10000)
            stimulus_this = test_util.get_random_stimulus(shape_this, num_stimulus_this)
            assert stimulus_this.shape == (num_stimulus_this,) + shape_this
            # check different locality
            for locality_to_use_scheme in ('all_zero', 'random', None):
                if locality_to_use_scheme == 'all_zero':
                    locality_to_use = (0,) * ndim
                elif locality_to_use_scheme == 'random':
                    locality_to_use = np.full(ndim, fill_value=-1, dtype=np.int64)
                    for dim_this in range(ndim):
                        locality_to_use[dim_this] = rng_stream_global.randint(shape_this[dim_this])
                    locality_to_use = tuple(locality_to_use)
                    # this is not always the case. but it works for this random seed.
                    assert not locality_to_use == (0,) * ndim
                elif locality_to_use_scheme is None:
                    # actually, equivalent to 'all_zero'
                    locality_to_use = None
                else:
                    raise NotImplementedError('unsupported locality')
                    # looked good to me
                    # print('ndim {}, locality {}, {}'.format(ndim, locality_to_use_scheme, locality_to_use))
                # ok. let's compute the quadratic features.
                stimulus_this_quadratic = feature_transformation.quadratic_features(stimulus_this, locality_to_use)


if __name__ == '__main__':
    unittest.main(failfast=True)
