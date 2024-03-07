import numpy as np
from numba import njit

# @njit
def _sample_diffusion_trial(v, a, bias, tau,  dt=0.001, s=1.0, max_iter=1e5):
    """Generates a single response time from a diffusion decision process.

    Parameters:
    -----------
    v        : float
        The drift rate parameter.
    a        : float
        The boundary separation parameter.
    tau      : float
        The non-decision time parameter.
    bias     : float
        The starting point parameter.
    dt       : float, optional, default: 0.001
        Time resolution of the process. Default corresponds to
        a precision of 1 millisecond.
    s        : float, optional, default: 1
        Scaling factor of the Wiener noise.
    max_iter : int, optional, default: 1e5
        Maximum iterations of the process. Default corresponds to
        100 seconds.

    Returns:
    --------
    rt : float
        A response time sample from the static diffusion decision process.
    resp : int
        A choice sample from the static diffusion decision process.
        Reaching the upper threshold results in 1, and 0 otherwise.

    """
    n_iter = 0
    x = a * bias
    c = np.sqrt(dt * s)
    while x > 0 and x < a and n_iter < max_iter:
        x += v*dt + c * np.random.randn()
        n_iter += 1
    rt = n_iter * dt + tau
    resp = 0 if x <= 0 else 1
    return np.array([rt, resp], dtype=np.float32)

# @njit
def sample_non_stationary_diffusion_process(params, context, dt=0.001, s=1.0, max_iter=1e5):
    """Generates a single simulation from a non-stationary diffusion decision process.

    Parameters:
    -----------
    params  : list, length 2
        The trajectory of the 3 local paramters theta_t and the shared parameters gamma
    context  : np.ndarray of shape (num_steps, 3)
        The experimental context.
    dt       : float, optional, default: 0.001
        Time resolution of the process. Default corresponds to
        a precision of 1 millisecond.
    s        : float, optional, default: 1
        Scaling factor of the Wiener noise.
    max_iter : int, optional, default: 1e5
        Maximum iterations of the process. Default corresponds to
        100 seconds.

    Returns:
    --------
    data : np.array of shape (num_steps, 2)
        Response time and choice samples from the random walk diffusion decision process.
    """
    theta_t = params[0]
    gamma = params[1]
    num_steps = theta_t.shape[0]
    data = np.zeros((num_steps, 2))
    for t in range(num_steps):
        drift = gamma[0] + theta_t[t, 0] * context[t, 2]
        tau = theta_t[t, 2] + gamma[1] * context[t, 1]
        if context[t, 0] == 1:
            data[t] = _sample_diffusion_trial(
                drift, theta_t[t, 1], gamma[2], tau,
                dt=dt, s=s, max_iter=max_iter
            )
        else:
            data[t] = _sample_diffusion_trial(
                -drift, theta_t[t, 1], gamma[2], tau,
                dt=dt, s=s, max_iter=max_iter
            )
    return data