from scipy.stats import halfnorm, truncnorm, beta
import numpy as np

from configurations import default_prior_settings, default_lower_bounds, default_upper_bounds

def sample_eta(loc=default_prior_settings['scale_loc'], scale=default_prior_settings['scale_scale']):
    """Generates 4 random draws from a half-normal prior over the
    scale of the random walk.

    Parameters:
    -----------
    loc    : tuple, optional, default: ``configuration.default_scale_prior_loc``
        The location parameters of the half-normal distribution.
    scale  : tuple, optional, default: ``configuration.default_scale_prior_scale``
        The scale parameters of the half-normal distribution.

    Returns:
    --------
    scales : np.array
        The randomly drawn scale parameters.
    """

    return halfnorm.rvs(loc=loc, scale=scale)

def sample_theta0():
    """Generates random draws from truncated normal and beta priors over the
    local parameters, v0, bv, bias, tau_1, tau_2, tau_3.

    Returns:
    --------
    ddm_params : np.array
        The randomly drawn local parameters, v0, bv, a, bias, tau_1, tau_2, tau_3.
    """

    v0 = truncnorm.rvs(a=0, b=np.inf, loc=0.0, scale=2.5)
    bv = truncnorm.rvs(a=0, b=np.inf, loc=0.0, scale=2.5)
    a = truncnorm.rvs(a=0, b=np.inf, loc=0.0, scale=2.5)
    bias = beta.rvs(a=50, b=50)
    tau_1 = truncnorm.rvs(a=0, b=np.inf, loc=0.0, scale=1)
    tau_2 = truncnorm.rvs(a=0, b=np.inf, loc=0.0, scale=1)
    tau_3 = truncnorm.rvs(a=0, b=np.inf, loc=0.0, scale=1)

    return np.c_[v0, bv, a, bias, tau_1, tau_2, tau_3]

def sample_random_walk(eta, num_steps=246, lower_bounds=default_lower_bounds, upper_bounds=default_upper_bounds, rng=None):
    """Generates a single simulation from a random walk transition model.

    Parameters:
    -----------
    sigma           : np.array
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
    # tau_1, tau_2, and tau_3 share eta
    eta = np.append(eta, [eta[-1], eta[-1]])

    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    # Sample initial parameters
    theta_t = np.zeros((num_steps, 7))
    theta_t[0] = sample_theta0()
    # Run random walk from initial
    z = rng.normal(size=(num_steps - 1, 7))
    for t in range(1, num_steps):
        theta_t[t] = np.clip(
            theta_t[t - 1] + eta * z[t - 1], lower_bounds, upper_bounds
        )
    return theta_t