import unittest
import numpy as np
from strflab import test_util, preprocessing
from itertools import product

rng_stream_global = test_util.rng_stream


def delay_one_array_alternative(array_this, delay):
    n, m = array_this.shape
    assert abs(delay) < n
    array_new = np.zeros_like(array_this)
    if delay > 0:
        # see past
        array_new[delay:] = array_this[:-delay]
    elif delay < 0:
        # see future
        array_new[:delay] = array_this[(-delay):]
    else:
        array_new[...] = array_this
    return array_new


def delay_one_array_all_delays(array_this, delays):
    return np.concatenate([delay_one_array_alternative(array_this, delay) for delay in delays], axis=1)


def delay_all_array(stim_list, delays):
    return [delay_one_array_all_delays(stim_this, delays) for stim_this in stim_list]


def truncate_alternative(array_this, truncate_config):
    trun_front, trun_back = truncate_config
    assert trun_front >= 0 and trun_back >= 0 and (trun_front + trun_back) < array_this.shape[0]
    if trun_front == 0 and trun_back == 0:
        return array_this
    elif trun_front > 0 and trun_back == 0:
        return array_this[trun_front:]
    elif trun_front == 0 and trun_back > 0:
        return array_this[:(-trun_back)]
    elif trun_front > 0 and trun_back > 0:
        return array_this[trun_front:(-trun_back)]
    else:
        raise RuntimeError('impossible')


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        rng_stream_global.seed(0)

    def test_check_input(self):
        for _ in range(10):
            shape_this = tuple(test_util.get_random_shape())
            num_neuron_this = rng_stream_global.randint(1, 100)
            num_trial_this = rng_stream_global.randint(1, 5)
            stim_list = []
            resp_list = []
            for i_trial in range(num_trial_this):
                num_stimulus_this = rng_stream_global.randint(100, 10000)
                stimulus_this = test_util.get_random_stimulus(shape_this, num_stimulus_this)
                response_this = rng_stream_global.rand(num_stimulus_this, num_neuron_this)
                stim_list.append(stimulus_this)
                resp_list.append(response_this)

            assert len(resp_list) == len(stim_list) == num_trial_this

            # pass in check input
            (stim_list_new, resp_list_new,
             shape_this_new, num_neuron_this_new) = preprocessing.check_input(stim_list, resp_list)
            self.assertEqual(shape_this, shape_this_new)
            self.assertEqual(num_neuron_this, num_neuron_this_new)
            # check list
            self.check_two_list_equal(stim_list, stim_list_new)
            self.check_two_list_equal(resp_list, resp_list_new)

    def test_flatten_stimulus_list(self):
        ndim_1_checked = False
        ndim_other_checked = False
        for ndim in (1, 4):
            shape_this = tuple(test_util.get_random_shape(ndim, ndim))
            num_trial_this = rng_stream_global.randint(1, 5)
            stim_list = []
            shape_list = []
            for i_trial in range(num_trial_this):
                num_stimulus_this = rng_stream_global.randint(100, 10000)
                stimulus_this = test_util.get_random_stimulus(shape_this, num_stimulus_this)
                stim_list.append(stimulus_this)
                shape_list.append((num_stimulus_this,) + shape_this)

            assert len(shape_list) == len(stim_list) == num_trial_this

            # pass in check input
            stim_list_flat = preprocessing.flatten_stimulus_list(stim_list)
            # check list
            if ndim == 1:
                ndim_1_checked = True
                self.check_two_list_equal(stim_list, stim_list_flat)
            else:
                ndim_other_checked = True
                self.assertEqual(len(stim_list_flat), num_trial_this)
                self.check_two_list_equal(stim_list, [stim_this.reshape(shape_this) for (stim_this, shape_this) in
                                                      zip(stim_list_flat, shape_list)])
        assert ndim_1_checked and ndim_other_checked

    def check_two_list_equal(self, a, b):
        self.assertEqual(len(a), len(b))
        for (x, y) in zip(a, b):
            # notice this won't work if your data has NaN.
            self.assertTrue(np.array_equal(x, y))

    def test_prepare_stimulus_delays(self):
        delay_0_checked = False
        for delays_this in [
            [0],
            [-1, 1],
            [-2, -1, 0, 1],
        ]:
            shape_this = tuple(test_util.get_random_shape(1, 1))
            num_trial_this = rng_stream_global.randint(1, 5)
            stim_list = []
            for i_trial in range(num_trial_this):
                num_stimulus_this = rng_stream_global.randint(100, 10000)
                stimulus_this = test_util.get_random_stimulus(shape_this, num_stimulus_this)
                stim_list.append(stimulus_this)

            assert len(stim_list) == num_trial_this

            # prepare delayed version
            stim_list_delayed = preprocessing.prepare_stimulus_all_delays_all_trials(stim_list, delays_this,
                                                                                     truncate_pars=(0, 0))

            # check list

            # get my own version.
            self.check_two_list_equal(delay_all_array(stim_list, delays_this), stim_list_delayed)
            if delays_this == [0]:
                delay_0_checked = True
                self.check_two_list_equal(stim_list, stim_list_delayed)

        assert delay_0_checked

    def test_truncate(self):
        for truncate_config in [
            (0, 0),
            (0, 2),
            (2, 0),
        ]:
            for delays_this in [
                [0],
                [-1, 1],
                [-2, -1, 0, 1],
            ]:
                shape_this = tuple(test_util.get_random_shape(1, 1))
                num_trial_this = rng_stream_global.randint(1, 5)
                stim_list = []
                for i_trial in range(num_trial_this):
                    num_stimulus_this = rng_stream_global.randint(100, 10000)
                    stimulus_this = test_util.get_random_stimulus(shape_this, num_stimulus_this)
                    stim_list.append(stimulus_this)

                assert len(stim_list) == num_trial_this

                # prepare delayed version
                stim_list_delayed = preprocessing.prepare_stimulus_all_delays_all_trials(stim_list, delays_this,
                                                                                         truncate_pars=truncate_config)
                stim_list_delayed_2 = [truncate_alternative(stim_this, truncate_config) for stim_this in
                                       delay_all_array(stim_list, delays_this)]
                self.check_two_list_equal(stim_list_delayed_2, stim_list_delayed)

    def generate_a_list_of_kernels(self, ndim, num_neuron, num_filter_per_neuron, delays_this):
        for x in delays_this:
            assert -2 <= x <= 2
        shape_this = tuple(test_util.get_random_shape(ndim, ndim, 1, 10))

        num_kernel_pixel_each = np.product(np.asarray(shape_this)) * len(delays_this)
        result_flat_kernel = np.empty((num_neuron, num_filter_per_neuron, num_kernel_pixel_each))

        ref_structured_kernel = np.empty((num_neuron, num_filter_per_neuron, len(delays_this),) + shape_this)

        for i_neuron in range(num_neuron):
            stim_list = []
            for i_filter in range(num_filter_per_neuron):
                stimulus_this = test_util.get_random_stimulus(shape_this, 5)
                stim_list.append(stimulus_this)
            assert len(stim_list) == num_filter_per_neuron
            # prepare delayed version
            stim_list_flat = preprocessing.flatten_stimulus_list(stim_list)
            stim_list_delayed = preprocessing.prepare_stimulus_all_delays_all_trials(stim_list_flat,
                                                                                     delays_this,
                                                                                     truncate_pars=(2, 2))
            assert len(stim_list_delayed) == num_filter_per_neuron
            for i_filter in range(num_filter_per_neuron):
                assert stim_list_delayed[i_filter].shape == (1, num_kernel_pixel_each)
                # fill in
                result_flat_kernel[i_neuron, i_filter] = stim_list_delayed[i_filter].ravel()

                # construct the structured one
                # fetch data from different delays
                for i_delay, delay in enumerate(delays_this):
                    # -, not +, as delay > 0 means past
                    ref_structured_kernel[i_neuron, i_filter, i_delay] = stim_list[i_filter][2 - delay]
        return result_flat_kernel, ref_structured_kernel, shape_this

    def test_reshape_kernel_single(self):
        for delays_this, ndim, num_neuron, num_filter, multi in product([[0], [-1, 1], [-2, -1, 0, 1], ],
                                                                        (1, 4), (1, 10), (1, 10), (True, False)):
            if num_filter > 1 and not multi:
                continue
            result_flat_kernel, ref_structured_kernel, shape_this = self.generate_a_list_of_kernels(
                ndim, num_neuron, num_filter, delays_this
            )
            print(result_flat_kernel.shape, ref_structured_kernel.shape)

            if multi:
                structured_kernel = preprocessing.reshape_kernel(result_flat_kernel, shape_this,
                                                                 len(delays_this), num_neuron, multi)
                self.assertTrue(np.array_equal(structured_kernel, ref_structured_kernel))
            else:
                assert result_flat_kernel.shape[1] == 1
                structured_kernel = preprocessing.reshape_kernel(result_flat_kernel[:, 0], shape_this,
                                                                 len(delays_this), num_neuron, multi)
                self.assertTrue(np.array_equal(structured_kernel, ref_structured_kernel[:, 0]))

    def test_cov(self):
        # just test computation of cov
        for _ in range(10):
            shape_this = tuple(test_util.get_random_shape(1, 1, low=2, high=100))
            num_stimulus_this = rng_stream_global.randint(100, 10000)
            stimulus_this = test_util.get_random_stimulus(shape_this, num_stimulus_this)
            assert stimulus_this.shape == (num_stimulus_this, shape_this[0])

            cov1 = preprocessing.cov_matrix(stimulus_this)

            stimulus_this_mean = stimulus_this.mean(axis=0, keepdims=True)
            stimulus_this = stimulus_this - stimulus_this_mean
            cov2 = np.matmul(stimulus_this.T, stimulus_this) / (num_stimulus_this - 1)

            self.assertEqual(cov1.shape, shape_this * 2)
            self.assertEqual(cov2.shape, shape_this * 2)
            self.assertTrue(np.allclose(cov1, cov2, atol=1e-6))


if __name__ == '__main__':
    unittest.main(failfast=True)
