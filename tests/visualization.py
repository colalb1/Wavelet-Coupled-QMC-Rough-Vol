import matplotlib.pyplot as plt

from configs.rough_vol_params import ROUGH_VOL_CONFIG
from core.baselines import cholesky_fbm
from core.wavelet_qmc import rough_fbm_paths
from core.wavelet_utils import wavelet_heatmap


def plot_fbm_paths(qmc_paths, cholesky_paths, num_samples=10):
    """
    Plot sample paths from QMC and Cholesky methods.

    Args:
        qmc_paths: QMC-generated paths
        cholesky_paths: Cholesky-generated paths
        num_samples: Number of sample paths to plot
    """
    plt.figure(figsize=(12, 6))

    # Plot QMC paths
    plt.subplot(1, 2, 1)
    for i in range(min(num_samples, qmc_paths.shape[1])):
        plt.plot(qmc_paths[:, i], alpha=0.5)
    plt.title("QMC Paths")
    plt.xlabel("Time")
    plt.ylabel("Value")

    # Plot Cholesky paths
    plt.subplot(1, 2, 2)
    for i in range(min(num_samples, cholesky_paths.shape[1])):
        plt.plot(cholesky_paths[:, i], alpha=0.5)
    plt.title("Cholesky Paths")
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()


def plot_wavelet_heatmap(coeffs, j_max):
    """
    Plot heatmap of wavelet coefficients.

    Args:
        coeffs: Array of wavelet coefficients
        j_max: Maximum wavelet scale
    """
    heatmap = wavelet_heatmap(coeffs, j_max)

    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, aspect="auto", cmap="viridis")
    plt.colorbar(label="Coefficient Value")
    plt.title("Wavelet Coefficient Heatmap")
    plt.xlabel("Translation (k)")
    plt.ylabel("Scale (j)")
    plt.show()


def main():
    """
    Main function for visualization testing.
    """
    config = ROUGH_VOL_CONFIG.copy()
    config["num_paths"] = 1000  # Reduce for visualization

    # Generate paths
    qmc_paths = rough_fbm_paths(**config)
    cholesky_paths = cholesky_fbm(**config)

    # Plot paths
    plot_fbm_paths(qmc_paths, cholesky_paths)

    # Plot wavelet coefficients for first path
    coeffs = qmc_paths[0, :]
    plot_wavelet_heatmap(coeffs, config["j_max"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=float, default=ROUGH_VOL_CONFIG["H"])
    parser.add_argument("--j_max", type=int, default=ROUGH_VOL_CONFIG["j_max"])
    args = parser.parse_args()

    ROUGH_VOL_CONFIG["H"] = args.H
    ROUGH_VOL_CONFIG["j_max"] = args.j_max

    main()
