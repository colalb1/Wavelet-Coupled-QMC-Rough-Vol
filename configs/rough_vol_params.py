ROUGH_VOL_CONFIG = {
    "H": 0.1,  # Hurst parameter
    "T": 1.0,  # Time horizon
    "dt": 1 / 252,  # Time step (daily)
    "num_paths": 100_000,  # Number of paths
    "j_max": 6,  # Maximum wavelet scale
    "sobol_dim": 2**6 - 1,  # Sobol sequence dimension
    "tolerance": 1e-4,  # Error tolerance
    "seed": 42,  # Random seed for reproducibility
}
