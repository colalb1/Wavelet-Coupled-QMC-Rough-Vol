import numba
import numpy as np
from scipy.stats.qmc import Sobol


@numba.njit(nogil=True, fastmath=True)
def haar_wavelet(t, j, k, H):
    """
    Compute Haar wavelet basis function at time t.

    Args:
        t: Time point
        j: Scale parameter
        k: Translation parameter
        H: Hurst parameter

    Returns:
        float: Value of Haar wavelet at time t
    """
    scale = 2.0 ** (-j * (H + 0.5))  # H-dependent normalization
    support_start = k * (2.0 ** (-j))
    support_end = (k + 1) * (2.0 ** (-j))
    if support_start <= t < support_end:
        return scale
    else:
        return -scale


@numba.njit(nogil=True, fastmath=True)
def compute_j_from_index(i):
    """
    Map index to wavelet scale j.
    """
    j = 0
    while i >= 2**j - 1:
        i -= 2**j
        j += 1
    return j


@numba.njit(nogil=True, fastmath=True)
def compute_k_from_index(i):
    """
    Map index to wavelet translation k.
    """
    j = 0
    while i >= 2**j - 1:
        i -= 2**j
        j += 1
    return i


@numba.njit(parallel=True)
def reconstruct_fbm_path(sobol_coeffs, H, T, dt):
    """
    Reconstruct fBM path from Sobol coefficients using Haar wavelets.

    Args:
        sobol_coeffs: Array of Sobol sequence coefficients
        H: Hurst parameter
        T: Time horizon
        dt: Time step

    Returns:
        np.ndarray: Reconstructed fBM path
    """
    num_steps = int(T / dt)
    t_values = np.linspace(0, T, num_steps)
    path = np.zeros_like(t_values)

    for i in numba.prange(len(sobol_coeffs)):
        j = compute_j_from_index(i)
        k = compute_k_from_index(i)
        for t_idx in range(len(t_values)):
            path[t_idx] += sobol_coeffs[i] * haar_wavelet(t_values[t_idx], j, k, H)

    return path


def generate_sobol_wavelet_coeffs(H, j_max, num_paths):
    """
    Generate Sobol sequence coefficients for wavelet reconstruction.

    Args:
        H: Hurst parameter
        j_max: Maximum wavelet scale
        num_paths: Number of paths to generate

    Returns:
        np.ndarray: Array of Sobol coefficients
    """
    sobol_dim = 2**j_max - 1  # Total number of wavelet coefficients
    sobol = Sobol(d=sobol_dim, scramble=True)
    return sobol.random(num_paths)


def rough_fbm_paths(H, T, dt, num_paths, j_max=6):
    """
    Generate rough fBM paths using wavelet-coupled QMC.

    Args:
        H: Hurst parameter
        T: Time horizon
        dt: Time step
        num_paths: Number of paths to generate
        j_max: Maximum wavelet scale

    Returns:
        np.ndarray: Array of fBM paths
    """
    # Generate Sobol coefficients
    sobol_coeffs = generate_sobol_wavelet_coeffs(H, j_max, num_paths)

    # Reconstruct paths
    paths = np.zeros((int(T / dt) + 1, num_paths))
    for i in range(num_paths):
        paths[:, i] = reconstruct_fbm_path(sobol_coeffs[i], H, T, dt)

    return paths
