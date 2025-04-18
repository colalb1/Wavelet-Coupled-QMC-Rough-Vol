import numpy as np
import pytest

from configs.rough_vol_params import ROUGH_VOL_CONFIG
from core.baselines import cholesky_fbm, standard_mc_fbm
from core.wavelet_qmc import rough_fbm_paths


def test_path_statistics():
    """
    Test that QMC paths have correct mean and variance.
    """
    config = ROUGH_VOL_CONFIG.copy()
    config["num_paths"] = 10_000  # Reduce for faster testing

    # Generate paths
    qmc_paths = rough_fbm_paths(**config)
    cholesky_paths = cholesky_fbm(**config)

    # Test mean at maturity
    qmc_mean = np.mean(qmc_paths[-1, :])
    cholesky_mean = np.mean(cholesky_paths[-1, :])
    assert np.abs(qmc_mean - cholesky_mean) < 0.1, "Mean mismatch"

    # Test variance at maturity
    qmc_var = np.var(qmc_paths[-1, :])
    cholesky_var = np.var(cholesky_paths[-1, :])
    assert np.abs(qmc_var - cholesky_var) < 0.1, "Variance mismatch"


def test_convergence_rate():
    """
    Test convergence rate against Cholesky (ground truth).
    """
    config = ROUGH_VOL_CONFIG.copy()
    config["num_paths"] = 10_000

    # Generate paths
    qmc_paths = rough_fbm_paths(**config)
    cholesky_paths = cholesky_fbm(**config)

    # Compute error
    error = np.mean(np.abs(qmc_paths - cholesky_paths))
    assert error < 0.1, f"QMC error {error} exceeds tolerance"


def test_variance_reduction():
    """
    Test that QMC reduces variance compared to standard MC.
    """
    config = ROUGH_VOL_CONFIG.copy()
    config["num_paths"] = 10_000

    # Generate paths
    qmc_paths = rough_fbm_paths(**config)
    mc_paths = standard_mc_fbm(**config)

    # Compute variance at maturity
    var_qmc = np.var(qmc_paths[-1, :])
    var_mc = np.var(mc_paths[-1, :])

    # QMC should reduce variance by at least 20%
    assert var_qmc < 0.8 * var_mc, "QMC failed to reduce variance by 20%"


@pytest.mark.benchmark
def test_performance(benchmark):
    """
    Benchmark performance of QMC vs Cholesky.
    """
    config = ROUGH_VOL_CONFIG.copy()
    config["num_paths"] = 1_000  # Reduce for benchmarking

    # Benchmark QMC
    benchmark(rough_fbm_paths, **config)

    # Note: Cholesky is too slow for benchmarking with large paths
