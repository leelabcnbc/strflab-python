from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import os.path
import numpy as np
from scipy.io import loadmat
from strflab.sta import sta
from strflab import pkg_path

ref_result_path = os.path.join(pkg_path, '..', 'ref_results')


class TestSTA(unittest.TestCase):
    def do_one_sta_test_reference(self, filename, do_original=True):
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

        self.do_one_sta_test([raw_stimulus],
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
            self.do_one_sta_test([raw_stimulus], [response], ref_kernels_to_use[np.newaxis], np.array([0]),
                                 {'delays': delays,
                                  'normalize_stimulus_mean': False,
                                  'normalize_stimulus_std': False}
                                 )

    def do_one_sta_test(self, raw_stimulus, response, ref_fit_kernels, ref_fit_intercept, config):
        # ok. run my program
        my_result = sta(raw_stimulus, response, config)
        fit_kernels = my_result['kernels']
        fit_intercept = my_result['intercept']
        self.assertEqual(ref_fit_kernels.shape, fit_kernels.shape)
        self.assertTrue(np.allclose(ref_fit_kernels, fit_kernels, atol=1e-4))
        self.assertEqual(ref_fit_intercept.shape, fit_intercept.shape)
        self.assertTrue(np.allclose(ref_fit_intercept, fit_intercept, atol=1e-4))

    def test_sta_t1_1(self):
        self.do_one_sta_test_reference('t1_1.mat', True)

    def test_sta_t1_2(self):
        self.do_one_sta_test_reference('t1_2.mat', False)

    def test_sta_no_penalty(self):
        # test aginast a bunch of different ground truth

        # generate delays.

        # generate stimulus

        # generate different shifted stimuli by difference shifting.

        # generate a random intercept.

        # compute response by a simple matrix multiplication!

        pass


if __name__ == '__main__':
    unittest.main()
