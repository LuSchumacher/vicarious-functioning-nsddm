from scipy.stats import halfnorm, truncnorm, beta
import numpy as np

from configurations import default_prior_settings, default_lower_bounds, default_upper_bounds

def sample_gamma(shape_1=default_prior_settings['gamma_shape_1'], shape_2=default_prior_settings['gamma_shape_2']):
    """Generates 3 random draws from prior distributions over the
    shared parameters gamma = {v0, b_tau, beta}.
    
    Parameters:
    -----------
    shape_1    : tuple, optional, default: ``configuration.default_prior_settings``
        Shape parameters for the prior distributions.
    shape_2  : tuple, optional, default: ``configuration.default_prior_settings``
        Shape parameters for the prior distributions.

    Returns:
    --------
    gamma : np.array
        Random draws for ganmma.
    """

    v0 = halfnorm.rvs(loc=shape_1[0], scale=shape_2[0])
    b_tau = halfnorm.rvs(loc=shape_1[1], scale=shape_2[1])
    bias = beta.rvs(a=shape_1[2], b=shape_2[2])

    return np.array([v0, b_tau, bias], dtype=np.float32)

def sample_eta(shape_1=default_prior_settings['eta_shape_1'], shape_2=default_prior_settings['eta_shape_2']):
    """Generates 3 random draws from a half-normal prior over the
    global parameters eta = {sigma_b_v, sigma_a, sigma_tau_0}.

    Parameters:
    -----------
    shape_1    : tuple, optional, default: ``configuration.default_prior_settings``
        Shape parameters for the prior distributions.
    shape_2  : tuple, optional, default: ``configuration.default_prior_settings``
        Shape parameters for the prior distributions.

    Returns:
    --------
    eta : np.array
        Random draws for eta.
    """

    return halfnorm.rvs(loc=shape_1, scale=shape_2)

def sample_theta0(lower_bounds=default_lower_bounds, upper_bounds=default_upper_bounds):
    """Generates random draws from from prior distributions over the
    inital values for the theta parameters theta0 = {b_v, a, tau_0}

    Parameters:
    -----------
    lower_bounds    : tuple, optional, default: ``configuration.default_lower_bounds``
        The minimum values the parameters can take.
    upper_bound     : tuple, optional, default: ``configuration.default_upper_bounds``
        The maximum values the parameters can take.

    Returns:
    --------
    ddm_params : np.array
        Random draws for theta0.
    """

    b_v = truncnorm.rvs(
        a=(lower_bounds[0] - 0.0) / 2.5,
        b=(upper_bounds[0] - 0.0) / 2.5,
        loc=0.0, scale=2.5
    )
    a = truncnorm.rvs(
        a=(lower_bounds[1] - 2.5) / 1.0,
        b=(upper_bounds[1] - 2.5) / 1.0,
        loc=2.5, scale=1.0
    )
    tau_0 = truncnorm.rvs(
        a=(lower_bounds[2] - 0.3) / 0.25,
        b=(upper_bounds[2] - 0.3) / 0.25,
        loc=0.3, scale=0.25
    )
    return np.c_[b_v, a, tau_0]

def sample_random_walk(eta, num_steps=246, lower_bounds=default_lower_bounds, upper_bounds=default_upper_bounds, rng=None):
    """Generates a single simulation from a random walk transition model.

    Parameters:
    -----------
    eta             : np.array
        The standard deviations of the random walk process.
    num_steps       : int, optional, default: 300
        The number of time steps to take for the random walk. Default
        corresponds to the maximal number of trials in the yes no dataset.
    lower_bounds    : tuple, optional, default: ``configuration.default_lower_bounds``
        The minimum values the parameters can take.
    upper_bound     : tuple, optional, default: ``configuration.default_upper_bounds``
        The maximum values the parameters can take.
    rng             : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    theta_t : np.ndarray of shape (num_steps, num_params)
        The array of time-varying parameters
    """

    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    # Sample initial parameters
    theta_t = np.zeros((num_steps, 3))
    theta_t[0] = sample_theta0(lower_bounds, upper_bounds)
    # Run random walk from initial
    z = rng.normal(size=(num_steps - 1, 3))
    for t in range(1, num_steps):
        theta_t[t] = np.clip(
            theta_t[t - 1] + eta * z[t - 1], lower_bounds, upper_bounds
        )
    return theta_t.astype(np.float32)