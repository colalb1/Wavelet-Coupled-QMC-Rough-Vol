import numba
import numpy as np


@numba.njit(nogil=True, fastmath=True)
def haar_basis(t, j, k):
    """Compute Haar basis function at time t.

    Args:
        t: Time point
        j: Scale parameter
        k: Translation parameter

    Returns:
        float: Value of Haar basis at time t
    """
    support_start = k * (2.0 ** (-j))
    support_end = (k + 1) * (2.0 ** (-j))
    if support_start <= t < support_end:
        return 1.0
    return 0.0


def adaptive_j_max(H, tolerance=1e-4):
    """Compute adaptive maximum wavelet scale based on Hurst parameter.

    Args:
        H: Hurst parameter
        tolerance: Error tolerance for wavelet truncation

    Returns:
        int: Maximum wavelet scale
    """
    # For rough volatility (H < 0.5), we need more scales
    if H < 0.3:
        return 8
    elif H < 0.4:
        return 7
    else:
        return 6


def wavelet_heatmap(coeffs, j_max):
    """Create a heatmap visualization of wavelet coefficients.

    Args:
        coeffs: Array of wavelet coefficients
        j_max: Maximum wavelet scale

    Returns:
        np.ndarray: 2D array for heatmap visualization
    """
    heatmap = np.zeros((j_max + 1, 2**j_max))
    idx = 0

    for j in range(j_max + 1):
        for k in range(2**j):
            if idx < len(coeffs):
                heatmap[j, k] = coeffs[idx]
                idx += 1

    return heatmap
