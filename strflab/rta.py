from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from .util import check_input, prepare_stimulus, truncate_array
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression, Ridge, Lasso


def rta(stimulus_list, response_list, config=None):
    """compute spike triggered average

    while this seems to be slower than original strf, I think this is due to algorithm.
    original strf used an iterative approach, and here a precise solution is obtained, using least square.
    the least square implemention used by sklearn is as fast as that in MATLAB, from my testing.

    Parameters
    ----------
    stimulus_list: a sequence of ndarrays, each of shape (N_k, K_1, K_2, ...)
        each element in the sequence stores the stimuli for a particular trial.
    response_list: a sequence of ndarrays, each of shape (N_k, N)
        each element in the sequence stores the response of N neurons.
    config:
        some config in a dict.
    Returns
    -------

    """
    if config is None:
        config = {}

    config_actual = {
        'normalize_stimulus_mean': True,  # center each feature.
        'normalize_stimulus_std': True,  # divide each feature by its std to make stimulus_list unit variance
        'regularization': None,  # regularization.
        'regularization_penalty': 0.1,  # alpha in Ridge Regression, or lasso.
        'solver_pars': {},  # other pars passed into solver.
        'delays': [0],  # what delay line to use. By default, a zero delay kernel,
        'truncate': (0, 0),  # first number is samples to drop for each trial in the front, second in the back.
        'correction': True,  # having correction, by doing linear regression (pseudo inverse).
    }
    config_actual.update(config)
    # TODO: check that your config is reall a valid one.

    # convert response and stimulus_list to ndarray
    stimulus_list, response_list, kernel_shape_each_delay, n_neuron = check_input(stimulus_list, response_list)
    stimulus_flat_list = [scale(stimulus_this.reshape(stimulus_this.shape[0], -1),
                                with_mean=config_actual['normalize_stimulus_mean'],
                                with_std=config_actual['normalize_stimulus_std']) for stimulus_this in stimulus_list]

    # then time to make delay lines.
    delays = config_actual['delays']
    truncate_config = config_actual['truncate']
    stimulus_flat_all = np.concatenate(
        [np.concatenate([prepare_stimulus(stimulus_flat, d, truncate_config) for d in delays], axis=1) for
         stimulus_flat in stimulus_flat_list], axis=0)
    n_stimulus_all, _ = stimulus_flat_all.shape
    response_all = np.concatenate([truncate_array(resp, truncate_config) for resp in response_list], axis=0)

    if config_actual['correction']:
        # ok. let's do the regression.
        additional_pars = config_actual['solver_pars']
        if not config_actual['regularization']:
            classifier = LinearRegression(copy_X=False, **additional_pars)
        else:
            if config_actual['regularization'] == 'ridge':
                alpha = config_actual['regularization_penalty'] * n_stimulus_all
                # multiply to make alpha independent of sample size.
                # see <http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression>
                # I also tested this myself.
                # for example, try duplicate X and y in
                # <http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html>
                # and see the change in plot; if I also make alpha = 2*alpha, then plot stays the same.
                classifier = Ridge(alpha=alpha, copy_X=False, **additional_pars)
            elif config_actual['regularization'] == 'lasso':
                # this is because in lasso, alpha is against every sample already.
                # check <http://scikit-learn.org/stable/modules/linear_model.html#lasso>
                alpha = config_actual['regularization_penalty']
                classifier = Lasso(alpha=alpha, copy_X=False, **additional_pars)
            else:
                raise RuntimeError('not support regularization yet!')

        classifier.fit(stimulus_flat_all, response_all)
        fit_kernels = classifier.coef_
        fit_intercept = classifier.intercept_
    else:
        fit_intercept = np.zeros(n_neuron, dtype=np.float64)
        # this case, the intercept doesn't matter, it's just dummy.
        # then doing naive averaging.
        fit_kernels = np.dot(response_all.T, stimulus_flat_all) / n_stimulus_all

    assert fit_intercept.shape == (n_neuron,)
    assert fit_kernels.shape == (n_neuron, np.product(kernel_shape_each_delay) * len(delays))
    fit_kernels = fit_kernels.reshape(n_neuron, len(delays), *kernel_shape_each_delay)

    return {
        'kernels': fit_kernels,
        'intercept': fit_intercept,
    }
