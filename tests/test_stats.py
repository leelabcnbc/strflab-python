import unittest
import numpy as np
from strflab import test_util, stats
import os.path
import h5py

rng_stream_global = test_util.rng_stream
current_file = os.path.abspath(__file__)
current_dir = os.path.split(current_file)[0]


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        rng_stream_global.seed(0)

    def test_cc_max(self):
        with h5py.File(os.path.join(current_dir, 'ref_stats', 'ref_stats_ccnorm.hdf5'), 'r') as f_out:
            y_true = f_out['/y_data_all'][...]
            ccmax_ref = f_out['/cc_max_all'][...].ravel()

        ccmax = stats.cc_max(y_true)
        self.assertEqual(ccmax.shape, ccmax_ref.shape)
        self.assertTrue(np.allclose(ccmax_ref, ccmax))

        # then test scalar.
        ccmax_2 = np.asarray([stats.cc_max(y_true_this) for y_true_this in y_true])
        self.assertEqual(ccmax_2.shape, ccmax_ref.shape)
        self.assertTrue(np.allclose(ccmax_ref, ccmax_2))


if __name__ == '__main__':
    unittest.main(failfast=True)
