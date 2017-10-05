"""statistics relevant for STRF fitting"""

import numpy as np


def cc_max(y_true, ddof=1):
    """computes CC_max according to paper

    Measuring the Performance of Neural Models.
    Schoppe, O.; Harper, N. S; Willmore, B. D B; King, A. J; and Schnupp, J. W H
    Frontiers in Computational Neuroscience, 10: 1929. February 2016.

    based on code from <https://github.com/OSchoppe/CCnorm/blob/master/calc_CCnorm.m>

    :param y_true: a num_neuron x num_trial x num_time or num_trial x num_time ndarray.
    :param ddof: ddof when computing variance. by default 1, same as the behavior of the reference code.
    :return: a result of (num_neuron,), each containing the CC_max for that neuron.

    if input a 2D array, then return a scalar instead.

    """

    y_true = np.asarray(y_true)

    scalar_flag = False

    if y_true.ndim == 2:
        y_true = y_true[np.newaxis]
        scalar_flag = True

    n_neuron, n_trial, n_time = y_true.shape
    assert n_trial > 1 and n_neuron > 0 and n_time > 1

    # check data valid
    assert np.all(np.isfinite(y_true))

    # then compute var_y
    # this always use ddof=0. same as reference code.
    # I believe they did this because var_y is also used to compute CC,
    # and when computing CC, people usually use ddof=0, as they will cancel out regardless, in the end.
    # also, as usually y_true.mean is very long (number of time bins), n-1 or n should not make much difference.
    var_y = np.var(y_true.mean(axis=1), axis=1)
    _cc_max_check(var_y, n_neuron)
    assert np.all(var_y > 0), 'there must be variance across time for mean PSTH'

    # Eq. (29) of the paper
    # then compute cc_max for each neuron.
    # same here, n-1 or n should not make much difference, as n is number of time bins.
    sp_var_sum = np.var(y_true.sum(axis=1), axis=1, ddof=ddof)
    _cc_max_check(sp_var_sum, n_neuron)
    sp_sum_var = np.var(y_true, axis=2, ddof=ddof).sum(axis=1)
    _cc_max_check(sp_sum_var, n_neuron)

    sp = (sp_var_sum - sp_sum_var) / (n_trial * (n_trial - 1))
    _cc_max_check(sp, n_neuron)
    bad_mask = sp <= 0
    sp[bad_mask] = 0

    cc_max_all = np.sqrt(sp / var_y)
    _cc_max_check(cc_max_all, n_neuron)
    cc_max_all[bad_mask] = np.nan
    if scalar_flag:
        assert cc_max_all.shape == (1,)
        cc_max_all = cc_max_all[0]
    return cc_max_all


def _cc_max_check(x, n_neuron):
    assert x.shape == (n_neuron,) and np.all(np.isfinite(x))
