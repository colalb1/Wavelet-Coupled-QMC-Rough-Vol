import numpy as np


def fbm_covariance_matrix(H, T, dt):
    """
    Compute the covariance matrix for fractional Brownian motion.

    Args:
        H: Hurst parameter
        T: Time horizon
        dt: Time step

    Returns:
        np.ndarray: Covariance matrix
    """
    t = np.arange(0, T + dt, dt)
    cov = 0.5 * (
        np.abs(t[:, None]) ** (2 * H)
        + np.abs(t[None, :]) ** (2 * H)
        - np.abs(t[:, None] - t[None, :]) ** (2 * H)
    )
    return cov


def cholesky_fbm(H, T, dt, num_paths):
    """
    Generate exact fBM paths using Cholesky decomposition.

    Args:
        H: Hurst parameter
        T: Time horizon
        dt: Time step
        num_paths: Number of paths to generate

    Returns:
        np.ndarray: Array of fBM paths
    """
    # Compute covariance matrix
    cov = fbm_covariance_matrix(H, T, dt)

    # Cholesky decomposition
    L = np.linalg.cholesky(cov)

    # Generate normal random variables
    Z = np.random.normal(size=(len(cov), num_paths))

    # Compute paths
    return L @ Z


def standard_mc_fbm(H, T, dt, num_paths):
    """
    Generate fBM paths using Euler-Maruyama approximation.

    Args:
        H: Hurst parameter
        T: Time horizon
        dt: Time step
        num_paths: Number of paths to generate

    Returns:
        np.ndarray: Array of fBM paths
    """
    num_steps = int(T / dt) + 1
    paths = np.zeros((num_steps, num_paths))

    for t in range(1, num_steps):
        dW = np.sqrt(dt) * np.random.normal(size=num_paths)
        paths[t] = paths[t - 1] + (dt**H) * dW

    return paths
