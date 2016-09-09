from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def check_input(stimulus, response):
    # for maximum precision, some many procedures only give double output.
    response = np.asarray(response).astype(np.float64, copy=False)
    stimulus = np.asarray(stimulus).astype(np.float64, copy=False)
    assert stimulus.ndim > 1 and response.ndim == 1
    assert stimulus.shape[0] == response.shape[0]
    return stimulus, response


def make_stimulus_with_delay(stimulus_flat, delay):
    assert stimulus_flat.ndim > 1
    assert delay < stimulus_flat.shape[0]
    filler = np.zeros((delay,) + stimulus_flat.shape[1:], dtype=stimulus_flat.dtype)
    if delay >= 0:
        result = np.concatenate([filler, stimulus_flat[delay:]], axis=0)
    else:
        result = np.concatenate([filler, stimulus_flat[:delay]], axis=0)
    return result
