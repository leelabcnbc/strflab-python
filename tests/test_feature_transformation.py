import unittest
import numpy as np
from strflab import test_util, feature_transformation
from itertools import product

rng_stream_global = test_util.rng_stream


# this doesn't work. it's more complicated than this.
# def compute_number_of_quadratic_features(shape_this, locality):
#     ndim = len(shape_this)
#
#     if locality is None:
#         locality = (0,) * len(shape_this)
#
#     assert len(locality) == ndim
#     assert np.all((np.array(shape_this) - np.array(locality)) > 0)
#
#     locality_loop = [range(loc_this + 1) for loc_this in locality]
#     total_pair = 0
#     for local_this_all in product(*locality_loop):
#         print(local_this_all)
#         # first, check how many points are there.
#         local_this_all_np = np.array(local_this_all)
#         assert local_this_all_np.shape == (ndim,)
#         num_pts = 2 ** ((local_this_all_np != 0).sum())
#         if num_pts < 2:
#             pair_num = 1
#         else:
#             pair_num = num_pts * (num_pts - 1) // 2
#         total_pair += (np.product(np.array(shape_this) - local_this_all_np))*pair_num
#     return total_pair

def generator_all_valid_pairs_alternative(shape, locality=None):
    linear_pair_to_save_ref = feature_transformation.generator_all_valid_pairs(shape, locality)

    if locality is None:
        locality = (0,) * len(shape)
    assert len(shape) == len(locality) and isinstance(shape, tuple) and isinstance(locality, tuple) and len(shape) > 0
    locality = np.array(locality)
    assert np.all(locality >= 0) and locality.shape == (len(shape),) and np.all(locality < np.array(shape))
    multi_index_loop = [range(shape_this) for shape_this in shape]

    multi_index_all = np.array(list(product(*multi_index_loop)))
    linear_index_all = np.ravel_multi_index(multi_index_all.T, shape, order='C')

    assert len(multi_index_all) == len(linear_index_all)

    linear_pair_to_save = []
    multi_pair_to_save = []
    total_pair = 0
    for (multi_1, linear_1) in zip(multi_index_all, linear_index_all):
        for (multi_2, linear_2) in zip(multi_index_all, linear_index_all):
            if linear_1 > linear_2:
                continue
            index_diff = abs(multi_1 - multi_2)
            assert index_diff.shape == locality.shape
            if np.all(index_diff <= locality):
                linear_pair_to_save.append((linear_1, linear_2))
                multi_pair_to_save.append(np.concatenate((multi_1, multi_2)))
                total_pair += 1

    linear_pair_to_save = np.asarray(linear_pair_to_save)
    multi_pair_to_save = np.asarray(multi_pair_to_save)
    assert linear_pair_to_save.shape == (total_pair, 2) and total_pair > 0
    assert multi_pair_to_save.shape == (total_pair, 2 * len(shape))

    # check that is linear stuff is the same as the multi one.
    assert np.array_equal(linear_pair_to_save_ref, linear_pair_to_save)

    return multi_pair_to_save


class TestFeatureTransformation(unittest.TestCase):
    def setUp(self):
        rng_stream_global.seed(0)

    def test_quadratic_features(self):
        for ndim in (1, 2, 3):
            shape_this = tuple(test_util.get_random_shape(ndim, ndim, low=1, high=10))
            num_el_all = np.product(shape_this)
            # print(shape_this)
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

                # ok. check that size is correct.
                if locality_to_use_scheme in {'all_zero', None}:
                    stimulus_this_quadratic_ref = stimulus_this.reshape(num_stimulus_this, -1) ** 2
                elif locality_to_use_scheme == 'random':
                    # stimulus_this_quadratic_ref = stimulus_this_quadratic
                    # ok. let's try another way to create this quadratic feature.
                    multi_pair_to_save = generator_all_valid_pairs_alternative(shape_this, locality_to_use)

                    stimulus_this_quadratic_ref = []
                    for pair in multi_pair_to_save:
                        index1, index2 = pair[:ndim], pair[ndim:]
                        index1_elements = stimulus_this[(slice(None),) + tuple(index1)]
                        index2_elements = stimulus_this[(slice(None),) + tuple(index2)]
                        assert index1_elements.shape == index2_elements.shape == (num_stimulus_this,)
                        stimulus_this_quadratic_ref.append(index1_elements * index2_elements)
                    stimulus_this_quadratic_ref = np.asarray(stimulus_this_quadratic_ref).T
                else:
                    raise NotImplementedError('unsupported locality')
                # print(stimulus_this_quadratic_ref.shape, compute_number_of_quadratic_features(shape_this, locality_to_use))
                # check
                self.assertEqual(stimulus_this_quadratic.shape, stimulus_this_quadratic_ref.shape)
                self.assertTrue(np.allclose(stimulus_this_quadratic, stimulus_this_quadratic_ref))


if __name__ == '__main__':
    unittest.main(failfast=True)
