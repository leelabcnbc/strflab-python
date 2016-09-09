from __future__ import absolute_import, division, print_function, unicode_literals
from copy import deepcopy
import numpy as np
from .util import check_input, make_stimulus_with_delay
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression


def sta(stimulus, response, config=None):
    """compute spike triggered average

    Parameters
    ----------
    stimulus
    response
    config

    Returns
    -------

    """
    default_config = {
        'normalize_stimulus_mean': True,  # center each feature.
        'normalize_stimulus_std': True,  # divide each feature by its std to make stimulus unit variance
        'regularization': False,  # regularization.
        'delays': [0],  # what delay line to use. By default, a zero delay kernel.
    }

    config_actual = deepcopy(config)
    config_actual.update(default_config)

    # convert response and stimulus to ndarray
    stimulus, response = check_input(stimulus, response)
    stimulus = scale(stimulus, with_mean=config_actual['normalize_stimulus_mean'],
                     with_std=config_actual['normalize_stimulus_std'])

    # then time to make delay lines.
    stimulus_flat = stimulus.reshape(stimulus.shape[0], -1)
    kernel_shape_each_delay = stimulus.shape[1:]
    delays = config_actual['delays']
    stimulus_flat_all = np.concatenate([make_stimulus_with_delay(stimulus_flat, d) for d in delays])

    # ok. let's do the regression.
    if not config_actual['regularization']:
        classifier = LinearRegression(copy_X=False)
    else:
        raise RuntimeError('not support regularization yet!')

    classifier.fit(stimulus_flat_all, response)

    fit_kernels = classifier.coef_
    fit_intercept = classifier.intercept_

    assert np.isscalar(fit_intercept)
    assert fit_kernels.shape == (kernel_shape_each_delay.size * len(delays),)
    fit_kernels = fit_kernels.reshape(len(delays), *kernel_shape_each_delay)

    return {
        'kernels': fit_kernels,
        'intercept': fit_intercept
    }
