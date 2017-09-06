"""this file deals with preprocessing, such as creating delay line, whitening"""

import numpy as np


def _prepare_stimulus_one_delay(stimulus_flat, delay, truncate_pars):
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
    result = _truncate_array(result, truncate_pars)
    assert not np.may_share_memory(result, stimulus_flat)
    return result


def flatten_stimulus_list(stimulus_list):
    """make stimulus data all 2D for each trial"""
    return [stimulus_this.reshape(stimulus_this.shape[0], -1) for stimulus_this in stimulus_list]


def _prepare_stimulus_all_delays_one_trial(stimulus_flat, delays, truncate_pars):
    assert stimulus_flat.ndim == 2
    return np.concatenate([_prepare_stimulus_one_delay(stimulus_flat, d, truncate_pars) for d in delays], axis=1)


def prepare_stimulus_all_delays_all_trials(flat_stimulus_list, delays, truncate_pars=(0, 0)):
    return [_prepare_stimulus_all_delays_one_trial(stimulus_flat, delays, truncate_pars) for
            stimulus_flat in flat_stimulus_list]


def _truncate_array(arr, truncate_config):
    arr_old_shape = arr.shape
    truncate_before, truncate_after = truncate_config
    assert truncate_before >= 0 and truncate_after >= 0
    assert truncate_before + truncate_after < len(arr)  # at least you need to keep something.
    if truncate_before > 0:
        arr = arr[truncate_before:]
    if truncate_after > 0:
        arr = arr[:-truncate_after]
    assert arr.shape == (arr_old_shape[0] - (truncate_before + truncate_after),) + arr_old_shape[1:]

    return arr


def check_input(stimulus_list, response_list):
    """check all input is good

    :param stimulus_list: an iterable of trials, each of shape T_i x [S], where T_i is number of time bins for trial i,
                          and [S] is the shape of stimulus.
    :param response_list: an iterable of responses, each of shape T_i x M, where M is number of neurons.
    :return: a tuple of the following 4 elements:
        list version of stimulus_list
        list version of response_list
        shape of stimulus (a tuple)
        number of neurons
    """
    n_trial = len(stimulus_list)
    assert n_trial == len(response_list) and n_trial >= 1
    stimulus_list_good = []
    response_list_good = []

    kernel_shape_first = None
    neuron_number_first = None
    for t, (stimulus, response) in enumerate(zip(stimulus_list, response_list)):
        stimulus_good, response_good, kernel_shape, neuron_number = _check_input_one(stimulus,
                                                                                     response)
        if kernel_shape_first is None and neuron_number_first is None:
            kernel_shape_first = kernel_shape
            neuron_number_first = neuron_number
        else:
            assert kernel_shape_first == kernel_shape and neuron_number_first == neuron_number
        stimulus_list_good.append(stimulus)
        response_list_good.append(response)

    return stimulus_list_good, response_list_good, kernel_shape_first, neuron_number_first


def reshape_kernel(kernels_recovered, stim_shape, num_delay, num_neuron, multiple_kernel_per_neuron=False):
    assert np.all(np.asarray(stim_shape) > 0) and num_delay > 0
    num_kernel_element = num_delay * np.prod(np.asarray(stim_shape))
    assert num_neuron > 0 and num_kernel_element > 0
    if multiple_kernel_per_neuron:
        a, b, c = kernels_recovered.shape
        assert b > 0
        new_shape = (num_neuron, b, num_delay) + stim_shape
    else:
        a, c = kernels_recovered.shape
        new_shape = (num_neuron, num_delay) + stim_shape
    assert (a, c) == (num_neuron, num_kernel_element)

    return np.reshape(kernels_recovered, new_shape)


def _check_input_one(stimulus, response):
    # for maximum precision, some many procedures only give double output.
    stimulus = np.asarray(stimulus)
    response = np.asarray(response)
    assert stimulus.ndim > 1 and response.ndim == 2
    assert stimulus.shape[0] == response.shape[0]
    assert stimulus.size > 0 and response.size > 0
    ret = stimulus, response
    ret = ret + (stimulus.shape[1:],)
    ret = ret + (response.shape[1],)
    return ret


def cov_matrix(stim):
    n, d = stim.shape
    assert n > 1 and d > 1  # when D is 1, returned might be a scalar...
    # compute covariance matrix
    # notice that while np.cov would subtract the mean, you should do it yourself when passing stim into STA and STC.
    # this norms data by N-1
    cov = np.cov(stim, rowvar=False)
    assert cov.shape == (d, d)
    return cov
