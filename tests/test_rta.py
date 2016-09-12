from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import os.path
import numpy as np
from scipy.io import loadmat
from scipy.signal import correlate
from strflab.rta import rta
from strflab import pkg_path
from strflab import test_util

ref_result_path = os.path.join(pkg_path, '..', 'ref_results')


class TestSTA(unittest.TestCase):
    def setUp(self):
        # reset seed for each test case.
        test_util.rng_stream.seed(seed=0)

    def do_one_rta_test_reference(self, filename, do_original=True):
        # load previous result
        ref_result = loadmat(os.path.join(ref_result_path, filename))
        raw_stimulus = ref_result['rawStim'].T
        raw_stimulus = np.transpose(raw_stimulus, (0, 2, 1))

        response = ref_result['resp'].ravel()[:, np.newaxis]
        response = np.array(response)
        delays = np.arange(9)
        ref_fit_kernels = ref_result['w1'].T.reshape(9, 10, 10)
        ref_fit_kernels = np.transpose(ref_fit_kernels, (0, 2, 1))[np.newaxis]
        ref_fit_intercept = ref_result['b1'].reshape(1)

        self.do_one_rta_test([raw_stimulus],
                             [response], ref_fit_kernels, ref_fit_intercept,
                             {'delays': delays})

        if do_original:
            # compare with original, don't normalize response at all.
            # # compare with original gabor filters
            # you can't, actually. because you have normalized the stuff.
            ref_kernels = ref_result['gabor'].T.reshape(-1, 10, 10)
            ref_kernels = np.transpose(ref_kernels, (0, 2, 1))
            n_frame = ref_kernels.shape[0]
            padding = int((9 - n_frame) / 2)
            filler = np.zeros((padding,) + ref_kernels.shape[1:], dtype=ref_kernels.dtype)
            ref_kernels_to_use = np.concatenate([filler, ref_kernels, filler], axis=0)
            self.do_one_rta_test([raw_stimulus], [response], ref_kernels_to_use[np.newaxis], np.array([0]),
                                 {'delays': delays,
                                  'normalize_stimulus_mean': False,
                                  'normalize_stimulus_std': False}
                                 )

    def do_one_rta_test(self, raw_stimulus, response, ref_fit_kernels, ref_fit_intercept, config):
        # ok. run my program
        my_result = rta(raw_stimulus, response, config)
        fit_kernels = my_result['kernels']
        fit_intercept = my_result['intercept']
        self.assertEqual(ref_fit_kernels.shape, fit_kernels.shape)
        self.assertTrue(np.allclose(ref_fit_kernels, fit_kernels, atol=1e-4))
        self.assertEqual(ref_fit_intercept.shape, fit_intercept.shape)
        self.assertTrue(np.allclose(ref_fit_intercept, fit_intercept, atol=1e-4))

    def test_rta_t1_1(self):
        self.do_one_rta_test_reference('t1_1.mat', True)

    def test_rta_t1_2(self):
        self.do_one_rta_test_reference('t1_2.mat', False)

    def one_rta_trial(self, min_stimulus=100, max_stimulus=2000,
                      truncate_front_max=0, truncate_back_max=0, min_trial=1, max_trial=100):
        # test aginast a bunch of different ground truth

        # generate delays.
        num_neuron = test_util.rng_stream.randint(1, 10)
        num_delay = test_util.rng_stream.randint(1, 10)
        shape_this = test_util.get_random_shape()
        delays = test_util.rng_stream.choice(np.arange(-2 * num_delay, 2 * num_delay),
                                             size=num_delay, replace=False)
        # all neurons being estimated have same number of delays.
        kernel_this = test_util.get_random_kernel(shape_this, num_neuron, num_delay)
        bias_this = test_util.rng_stream.randn(num_neuron)
        num_trial = test_util.rng_stream.randint(min_trial, max_trial + 1)

        stimulus_list = []
        response_list = []

        truncate_front = test_util.rng_stream.randint(0, truncate_front_max + 1)
        truncate_back = test_util.rng_stream.randint(0, truncate_back_max + 1)

        for i_trial in range(num_trial):
            if i_trial + 1 % 100 == 0 or i_trial == num_trial - 1:
                print('trial {}/{}'.format(i_trial + 1, num_trial))
            num_stimulus_this = test_util.rng_stream.randint(min_stimulus, max_stimulus)
            # generate stimulus
            stimulus_this = test_util.get_random_stimulus(shape_this, num_stimulus_this)

            # generate the response for different neurons.
            response_all_this_trial = []
            for i_neuron in range(num_neuron):
                response_all_this_neuron = []
                bias_this_this = bias_this[i_neuron]
                for i_delay in range(num_delay):
                    delay_this = delays[i_delay]
                    kernel_this_this = kernel_this[i_neuron, i_delay:i_delay + 1]
                    # compute correlation, which is how I did regression.
                    response_this = correlate(stimulus_this, kernel_this_this, mode='valid')
                    assert response_this.shape == (num_stimulus_this,) + (1,) * len(shape_this)
                    response_this = np.squeeze(response_this)
                    # generate different shifted stimuli by difference shifting.
                    zero_filler = np.zeros(abs(delay_this), dtype=response_this.dtype)
                    if delay_this > 0:
                        response_this = np.concatenate([zero_filler, response_this[:(-delay_this)]])
                    else:
                        response_this = np.concatenate([response_this[(-delay_this):], zero_filler])
                    if truncate_front > 0:
                        response_this[:truncate_front] = test_util.rng_stream.randn(truncate_front)
                    if truncate_back > 0:
                        response_this[(-truncate_back):] = test_util.rng_stream.randn(truncate_back)
                    response_all_this_neuron.append(response_this)
                response_all_this_neuron = np.array(response_all_this_neuron).sum(axis=0) + bias_this_this
                response_all_this_trial.append(response_all_this_neuron)
            stimulus_list.append(stimulus_this)
            response_list.append(np.array(response_all_this_trial).T)
        # ok, then, let's do the real work.
        self.do_one_rta_test(stimulus_list, response_list, kernel_this, bias_this,
                             {'delays': delays, 'truncate': (truncate_front, truncate_back),
                              'normalize_stimulus_mean': False,
                              'normalize_stimulus_std': False})

    def test_rta_no_truncate(self):
        for _ in range(5):
            self.one_rta_trial()

    def test_rta_truncate(self):
        for _ in range(5):
            self.one_rta_trial(min_stimulus=100, max_stimulus=200, truncate_front_max=30, truncate_back_max=30,
                               max_trial=1000, min_trial=500)


if __name__ == '__main__':
    unittest.main(failfast=True)
