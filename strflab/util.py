from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import time


def check_input_one(stimulus, response, return_kernel_shape=False, return_neuron_number=False):
    # for maximum precision, some many procedures only give double output.
    stimulus = np.asarray(stimulus).astype(np.float64, copy=False)
    response = np.asarray(response).astype(np.float64, copy=False)
    assert stimulus.ndim > 1 and response.ndim == 2
    assert stimulus.shape[0] == response.shape[0]
    assert stimulus.size > 0 and response.size > 0
    ret = stimulus, response
    if return_kernel_shape:
        ret = ret + (stimulus.shape[1:],)
    if return_neuron_number:
        ret = ret + (response.shape[1],)
    return ret


def check_input(stimulus_list, response_list):
    n_trial = len(stimulus_list)
    assert n_trial == len(response_list) and n_trial >= 1
    stimulus_list_good = []
    response_list_good = []

    kernel_shape_first = None
    neuron_number_first = None
    for t in range(n_trial):
        stimulus = stimulus_list[t]
        response = response_list[t]
        stimulus_good, response_good, kernel_shape, neuron_number = check_input_one(stimulus,
                                                                                    response,
                                                                                    True, True)
        if t == 0:
            kernel_shape_first = kernel_shape
            neuron_number_first = neuron_number

        assert kernel_shape_first == kernel_shape and neuron_number_first == neuron_number
        stimulus_list_good.append(stimulus)
        response_list_good.append(response)

    return stimulus_list_good, response_list_good, kernel_shape_first, neuron_number_first


def prepare_stimulus(stimulus_flat, delay, truncate_pars=(0, 0)):
    assert stimulus_flat.ndim > 1
    assert abs(delay) < stimulus_flat.shape[0]
    filler = np.zeros((abs(delay),) + stimulus_flat.shape[1:], dtype=stimulus_flat.dtype)
    if delay > 0:
        # see the past; remove last delay elements
        result = np.concatenate([filler, stimulus_flat[:(-delay)]], axis=0)
    else:
        # see the future; remove first (-delay)
        result = np.concatenate([stimulus_flat[(-delay):], filler], axis=0)
    # then let's truncate.
    result = truncate_array(result, truncate_pars)
    assert not np.may_share_memory(result, stimulus_flat)
    return result


def truncate_array(arr, truncate_config):
    truncate_before, truncate_after = truncate_config
    assert truncate_before >= 0 and truncate_after >= 0
    if truncate_before > 0:
        arr = arr[truncate_before:]
    if truncate_after > 0:
        arr = arr[:-truncate_after]

    return arr


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type_, value, traceback):
        exit_time = time.time()
        if self.name:
            print('[{}]'.format(self.name), end='')
        print('Elapsed: {}'.format(exit_time - self.tstart))
